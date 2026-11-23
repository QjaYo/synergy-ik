"""Review behavior groups with O/X/U; Q excludes only this take and its mirror for quality."""

import argparse
import csv
from concurrent.futures import Future
from functools import partial
from pathlib import Path
import sqlite3
import sys
import textwrap
import time

import numpy as np

from motion_review_data import (DEFAULT_REVIEW_DIR, DEFAULT_METADATA, PROJECT_ROOT,
                                MotionReviewStore, MotionDescription, load_motion_descriptions)
from uniform_mesh import (DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG,
                          load_aligned_uniform_mesh, surface_colors)

sys.path.insert(0, str(PROJECT_ROOT / "source"))
from util import load_canonicalized_bvh
from viewer import create_initial_camera


HEADER_HEIGHT = 210


def prepare_review_queue(
    store: MotionReviewStore, selection: str = "all", unit: str = "action",
    action_name: str | None = None, limit: int | None = None,
) -> tuple[list[str], int]:
    """Retain completed entries; start at the first unresolved entry in scope."""
    if unit not in ("action", "clip"):
        raise ValueError(unit)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    queue = store.action_queue if unit == "action" else store.queue
    keys = queue(selection, action_name)
    if limit is not None:
        keys = keys[:limit]
    pending = set(queue("pending", action_name))
    start_index = next((i for i, key in enumerate(keys) if key in pending), 0)
    return keys, start_index


def create_review_viewer(
    store: MotionReviewStore, keys: list[str], speed: float, unit: str = "action",
    start_index: int = 0,
    uniform_root: Path = DEFAULT_UNIFORM_ROOT, uniform_rig: Path = DEFAULT_UNIFORM_RIG,
    show_mesh: bool = True,
    descriptions: dict[str, MotionDescription] | None = None,
    selection: str = "all",
):
    """Keep a stable queue; only a committed human decision changes inclusion."""
    if unit not in ("action", "clip"):
        raise ValueError(unit)
    from motion_view import MotionView

    class MotionReviewViewer(MotionView):
        def __init__(self) -> None:
            super().__init__(b"SynergyIK | Motion inclusion review", speed, show_mesh, surface_colors)
            self.loading: Future | None = None
            self.loading_index = -1
            self.show_details = False
            self.current_key = ""
            self.members: list[str] = []
            self.sample_index = 0
            self.index = -1
            self.row: dict = {}
            self.ready_at = 0.0
            self.remaining = self.remaining_count()
            self.select(start_index)

        def remaining_count(self) -> int:
            return len(store.action_queue() if unit == "action" else store.queue())

        def select(self, index: int) -> None:
            if index < 0 or index > len(keys):
                return
            self.index = index
            self.motion = None
            self.clear_mesh()
            self.play_time = 0.0
            self.paused = False
            self.last_root = None
            if self.loading is not None:
                self.loading.cancel()
            if index == len(keys):
                self.loading = None
                self.row = {}
                self.members = []
                self.current_key = ""
                self.status = "Queue complete. Skipped items remain pending. Previous returns to the last item."
                return
            self.members = store.clips_for_action(keys[index], prefer_quality=selection == "quality-exclude") if unit == "action" else [keys[index]]
            self.load_sample(0)

        def load_sample(self, sample_index: int) -> None:
            if not self.members:
                return
            if self.loading is not None:
                self.loading.cancel()
            self.sample_index = sample_index % len(self.members)
            self.current_key = self.members[self.sample_index]
            self.row = store.get(self.current_key)
            self.mesh_loader = partial(load_aligned_uniform_mesh, uniform_root / self.current_key, uniform_rig)
            self.motion = None
            self.clear_mesh()
            self.play_time = 0.0
            self.paused = False
            self.last_root = None
            self.loading_index = self.index
            self.loading = self.executor.submit(load_canonicalized_bvh, store.file_path(self.current_key), "m")
            self.status = "Loading motion... O/X/U/Q is disabled until the motion is ready."

        def poll_loading(self) -> None:
            if self.loading is None or not self.loading.done():
                return
            future, self.loading = self.loading, None
            if self.loading_index != self.index:
                return
            try:
                self.motion = future.result()
                self.cam_cur = create_initial_camera(self.motion)
                self.play_time = 0.0
                self.last_root = None
                self.ready_at = time.monotonic() + 0.35
                store.set_load_error(self.current_key, None)
                self.status = ("O/X/U applies to ALL originals/mirrors in this group. U saves Unsure for later review. B shows another example."
                               if unit == "action" else "O/X/U saves an individual override for this original and its mirror.")
                self.start_mesh_loading()
            except Exception as error:
                self.motion = None
                self.status = f"Load failed: {type(error).__name__}: {error}. Use B or Skip; no verdict was saved."
                store.set_load_error(self.current_key, str(error))

        def can_decide(self) -> bool:
            return self.motion is not None and time.monotonic() >= self.ready_at

        def saved_verdict(self) -> str | None:
            """Return the saved human verdict at the currently edited scope."""
            field = "action_decision" if unit == "action" else "human_decision"
            return self.row.get(field)

        def act(self, action: str) -> None:
            if self.handle_playback_action(action):
                return
            if action in ("include", "exclude", "uncertain", "quality_exclude"):
                if not self.can_decide():
                    return
                try:
                    if unit == "action" and action != "quality_exclude":
                        store.save_action_decision(keys[self.index], action, self.current_key)
                    else:
                        store.save_decision(self.current_key, action)
                except (sqlite3.Error, OSError) as error:
                    self.status = f"Save failed; staying on this motion: {error}"
                    return
                self.remaining = self.remaining_count()
                if unit == "action" and action == "quality_exclude":
                    pending = set(store.queue("pending", keys[self.index]))
                    next_sample = next((i for i, path in enumerate(self.members) if path in pending), None)
                    if next_sample is not None:
                        self.load_sample(next_sample)
                        return
                self.select(self.index + 1)
            elif action == "clear_quality" and self.row.get("human_decision") == "quality_exclude":
                try:
                    store.save_decision(self.current_key, None)
                    self.row = store.get(self.current_key)
                    self.remaining = self.remaining_count()
                    self.status = "Quality exclusion cleared for this take and its mirror. Group verdict or automatic screening now applies."
                except (sqlite3.Error, OSError) as error:
                    self.status = f"Save failed: {error}"
            elif action == "previous":
                self.select(self.index - 1)
            elif action == "next" and self.index < len(keys):
                self.select(self.index + 1)
            elif action == "another":
                self.load_sample(self.sample_index + 1)
            elif action == "clear" and self.index < len(keys):
                try:
                    if unit == "action":
                        store.save_action_decision(keys[self.index], None, self.current_key)
                    else:
                        store.save_decision(self.current_key, None)
                    self.row = store.get(self.current_key)
                    self.remaining = self.remaining_count()
                    self.status = ("Group verdict cleared. Individual clip overrides remain."
                                   if unit == "action" else "Clip override cleared. Group verdict or automatic screening now applies.")
                except sqlite3.Error as error:
                    self.status = f"Save failed: {error}"
            elif action == "details":
                self.show_details = not self.show_details
            elif action == "export":
                try:
                    store.export()
                    self.status = "Current lists exported beside the review database."
                except OSError as error:
                    self.status = f"DB decisions are saved; export failed: {error}"

        def description_lines(self) -> list[str]:
            description = (descriptions or {}).get(self.current_key)
            text = description.text if description and description.text else "No official description available for this clip."
            return textwrap.wrap(text, max(35, (self.window_size[0] - 32) // 10))

        def header_height(self) -> int:
            return HEADER_HEIGHT + max(0, len(self.description_lines()) - 2) * 22 + (145 if self.show_details else 0)

        def draw_controls(self) -> None:
            width, height = self.window_size
            self.buttons = []
            self.rectangle(0, 0, width, self.header_height(), (0.97, 0.98, 0.99, 1))
            scope = "GROUP" if unit == "action" else "CLIP"
            verdict = self.saved_verdict()
            self.button("O INCLUDE", 12, 12, 115, "include", self.can_decide(), selected=verdict == "include")
            self.button("X EXCLUDE", 137, 12, 115, "exclude", self.can_decide(), selected=verdict == "exclude")
            self.button("U UNSURE", 262, 12, 120, "uncertain", self.can_decide(), selected=verdict == "uncertain")
            self.button("Q QUALITY EXCLUDE", 392, 12, 175, "quality_exclude", self.can_decide(), selected=self.row.get("human_decision") == "quality_exclude")
            self.button("Previous [P]", 577, 12, 105, "previous", self.index > 0)
            self.button("Skip [N]", 692, 12, 85, "next", self.index < len(keys))
            self.button("Another take [B]", 787, 12, 125, "another", len(self.members) > 1)
            self.button("Details [D]", 922, 12, 110, "details", selected=self.show_details)
            self.button("Close [Esc]", 1042, 12, 100, "close")
            self.text(f"Group {min(self.index + 1, len(keys))}/{len(keys)} | Needs decision: {self.remaining} | Variants: {self.row.get('group_type_count', 1)} | Example {self.sample_index + 1}/{len(self.members)}", 12, 64)
            self.text(f"{scope.title()} choice: {verdict.upper().replace('_', ' ') if verdict else 'NOT SET'}", 800, 64)
            lines = self.description_lines()
            for line_index, line in enumerate(lines):
                self.text(line, 12, 94 + line_index * 22, large=True)
            description = (descriptions or {}).get(self.current_key)
            extra = max(0, len(lines) - 2) * 22
            props = description.props if description else "Metadata unavailable"
            self.text(textwrap.shorten(f"Props: {props}", max(40, (width-24)//7), placeholder="..."), 12, 140 + extra)
            if self.row:
                source = {"clip": "clip override", "action": "group decision", "automatic": "automatic suggestion"}.get(self.row.get("decision_source"), "automatic suggestion")
                label = self.row['effective_decision'].upper().replace('_', ' ')
                quality_note = " | Omitted from both training pools" if self.row["effective_decision"] == "quality_exclude" else ""
                self.text(f"Used label: {label} ({source}){quality_note}", 12, 162 + extra)
            if self.show_mesh and self.current_key:
                if self.mesh is not None and self.motion is not None:
                    y_values = self.mesh.frame(self.frame_index()).vertices[:, 1]
                    self.text(f"Lowest surface: {y_values.min()*1000:+.1f} mm | Red: below floor | Yellow: 0-{self.contact_threshold*1000:g} mm | Blue: higher", 12, 189 + extra)
                else:
                    self.text((self.mesh_error or "Loading original Uniform surface...")[:max(30, (width - 24) // 7)], 12, 189 + extra)
            if self.show_details and self.row:
                y_detail = HEADER_HEIGHT + extra
                self.text(f"Group: {self.row['action_name']}", 12, y_detail + 10)
                self.text(f"Motion type: {self.row.get('motion_type', self.row['action_name'])}", 12, y_detail + 28)
                self.text(self.row["original_path"], 12, y_detail + 46)
                self.text(f"Auto: {self.row['auto_decision']} | Group: {self.row['action_decision'] or '-'} | Clip override: {self.row['human_decision'] or '-'}", 12, y_detail + 64)
                self.text(self.row["reason"][:max(30, (width-24)//7)], 12, y_detail + 82)
                if self.show_mesh and self.mesh is not None and self.motion is not None:
                    self.text(f"Below y=0: {np.count_nonzero(y_values < 0)}/{len(y_values)} vertices | USD LBS | Display bands only; no ground correction", 12, y_detail + 100)
                clear_field = "action_decision" if unit == "action" else "human_decision"
                self.button(f"Clear {scope.lower()}", 12, y_detail + 110, 100, "clear", bool(self.row.get(clear_field)))
                self.button("Export [E]", 122, y_detail + 110, 90, "export")
                if self.row["human_decision"] == "quality_exclude":
                    self.button("Clear quality exclusion", 222, y_detail + 110, 170, "clear_quality")
            normal_status = self.status.startswith("O/X/U")
            hint = (f"O/X/U: group ({len(self.members)} takes), keeping quality exclusions. " if unit == "action" else "O/X/U: current take. ") + "Q: quality exclude this take + mirror only."
            self.draw_playback_controls(hint if normal_status else self.status)

        def keyboard_callback(self, key: bytes) -> bool:
            actions = {b"o": "include", b"x": "exclude", b"u": "uncertain", b"q": "quality_exclude",
                       b"p": "previous", b"n": "next", b"e": "export", b"b": "another",
                       b"d": "details", b"\x08": "clear"}
            if key.lower() in actions:
                self.act(actions[key.lower()])
                return True
            return super().keyboard_callback(key)


    return MotionReviewViewer()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_REVIEW_DIR / "review.sqlite3")
    parser.add_argument("--queue", choices=("pending", "all", "auto-include", "auto-exclude", "reviewed", "uncertain", "quality-exclude"), default="all",
                        help="Keep all entries by default and resume at the first unresolved one")
    parser.add_argument("--unit", choices=("action", "clip"), default="action", help="Review a whole behavior group (default) or individual exceptions")
    parser.add_argument("--action", help="Limit to one exact action_name, useful with --unit clip --queue all")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="Open only the first N types (or clips in clip mode)")
    parser.add_argument("--mesh", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the original mesh initially (default); --no-mesh starts hidden")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Official metadata CSV for descriptions; display-only")
    parser.add_argument("--uniform-root", type=Path, default=DEFAULT_UNIFORM_ROOT, help="Original Uniform BVHs, containing date directories")
    parser.add_argument("--uniform-rig", type=Path, default=DEFAULT_UNIFORM_RIG, help="Distributed soma_base_skel_minimal.usd")
    parser.add_argument("--check", action="store_true", help="Load the resume representative without opening the GUI")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("Review DB is missing. Run: python preprocess/motion_review/motion_review_data.py build")
    if not np.isfinite(args.speed) or args.speed <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Speed and limit must be positive")
    store = MotionReviewStore(args.db)
    viewer = None
    try:
        if store.schema_backup:
            print(f"Manual verdict schema updated. Backup: {store.schema_backup}")
        migration = store.adopt_legacy_decisions()
        if migration["backup"]:
            print(f"Adopted {migration['adopted_actions']} existing type verdicts. Backup: {migration['backup']}")
            store.export()
        keys, start_index = prepare_review_queue(store, args.queue, args.unit, args.action, args.limit)
        if not keys:
            print("No motions in this queue. Use --queue all to inspect existing decisions.")
            return
        try:
            descriptions = load_motion_descriptions(args.metadata)
        except (OSError, ValueError, csv.Error) as error:
            print(f"Descriptions unavailable: {error}", file=sys.stderr)
            descriptions = {}
        if args.check:
            first = store.clips_for_action(keys[start_index], prefer_quality=args.queue == "quality-exclude")[0] if args.unit == "action" else keys[start_index]
            motion = load_canonicalized_bvh(store.file_path(first), "m")
            print(f"OK: {len(keys)} {args.unit}s queued | Start {start_index + 1}/{len(keys)} | {first} | {motion.num_frames()} frames | {motion.fps:g} FPS")
            if first in descriptions:
                print(f"Description: {descriptions[first].text}\nProps: {descriptions[first].props}")
            if args.mesh:
                mesh = load_aligned_uniform_mesh(args.uniform_root / first, args.uniform_rig, motion)
                print(f"Mesh: {len(mesh.rest_points)} vertices | {len(mesh.triangles)} triangles | alignment max {mesh.alignment_error_m*1000:.5f} mm | first-frame min Y {mesh.frame(0).vertices[:,1].min()*1000:+.3f} mm")
            return
        viewer = create_review_viewer(store, keys, args.speed, args.unit, start_index,
                                      args.uniform_root, args.uniform_rig, args.mesh, descriptions, args.queue)
        viewer.run()
    finally:
        try:
            if viewer is not None:
                viewer.shutdown()
                store.export()
        finally:
            store.close()


if __name__ == "__main__":
    main()
