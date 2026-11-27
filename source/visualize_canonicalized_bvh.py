"""Play canonical BVH files with the same surface and playback UI as motion review."""

import argparse
import csv
from functools import partial
from pathlib import Path
import sys
import textwrap
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from util import load_canonicalized_bvh
from viewer import create_initial_camera

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "preprocess/motion_review"))
from motion_review_data import DEFAULT_METADATA, load_motion_descriptions, original_key, MotionDescription
from uniform_mesh import (DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG,
                          load_aligned_uniform_mesh, surface_colors)


if TYPE_CHECKING:
    from fairmotion.core.motion import Motion
    from motion_view import MotionView


def parse_viewer_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bvh-file", type=Path)
    inputs.add_argument("--bvh-list", type=Path,
                        help="UTF-8 text: one BVH path per line, relative to the list file")
    parser.add_argument("--unit", choices=("m", "cm"), default="m")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--mesh", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the original mesh initially (default); --no-mesh starts hidden")
    parser.add_argument("--uniform-root", type=Path, default=DEFAULT_UNIFORM_ROOT)
    parser.add_argument("--uniform-rig", type=Path, default=DEFAULT_UNIFORM_RIG)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--check", action="store_true", help="Validate the playlist without opening a window")
    arguments = parser.parse_args(argv)
    if not np.isfinite(arguments.speed) or arguments.speed <= 0:
        parser.error("--speed must be positive and finite")
    return arguments


def read_bvh_paths(arguments: argparse.Namespace) -> list[Path]:
    """Read an ordered playlist; ignore blank lines and # comments."""
    if arguments.bvh_file is not None:
        paths = [arguments.bvh_file.resolve()]
    else:
        playlist = arguments.bvh_list.resolve()
        paths = [(playlist.parent / line.strip()).resolve()
                 for line in playlist.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if not paths:
        raise ValueError("BVH list is empty")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"BVH file does not exist: {path}")
    return paths


def create_canonicalized_bvh_viewer(
    motion: "Motion", playback_speed: float, paths: Sequence[Path], unit: str = "m", *,
    show_mesh: bool = True,
    uniform_root: Path = DEFAULT_UNIFORM_ROOT,
    uniform_rig: Path = DEFAULT_UNIFORM_RIG,
    descriptions: Mapping[str, MotionDescription] | None = None,
) -> "MotionView":
    """A playlist supplies clips; MotionView supplies the shared playback screen."""
    from motion_view import MotionView

    class CanonicalizedBvhViewer(MotionView):
        def __init__(self) -> None:
            super().__init__(b"SynergyIK | Motion viewer", playback_speed, show_mesh, surface_colors)
            self.paths = list(paths)
            self.motion_index = 0
            self.loading = None
            self.motion = motion
            self.cam_cur = create_initial_camera(motion)
            self.configure_mesh_loader()
            self.start_mesh_loading()

        def clip_key(self) -> str:
            path = self.paths[self.motion_index]
            return f"{path.parent.name}/{path.name}"

        def configure_mesh_loader(self) -> None:
            self.mesh_loader = partial(load_aligned_uniform_mesh, uniform_root / self.clip_key(), uniform_rig)

        def navigation_enabled(self, direction: int) -> bool:
            return 0 <= self.motion_index + direction < len(self.paths)

        def select_motion(self, direction: int) -> None:
            if not self.navigation_enabled(direction):
                return
            if self.loading is not None:
                self.loading.cancel()
            self.motion_index += direction
            self.motion = None
            self.clear_mesh()
            self.configure_mesh_loader()
            self.play_time = 0.0
            self.paused = False
            self.last_root = None
            self.loading = self.executor.submit(load_canonicalized_bvh, self.paths[self.motion_index], unit)
            self.status = "Loading motion..."

        def poll_loading(self) -> None:
            if self.loading is None or not self.loading.done():
                return
            future, self.loading = self.loading, None
            try:
                self.motion = future.result()
                self.cam_cur = create_initial_camera(self.motion)
                self.status = ""
                self.start_mesh_loading()
            except Exception as error:
                self.status = f"Load failed: {type(error).__name__}: {error}"

        def act(self, action: str) -> None:
            if action in ("previous", "next"):
                self.select_motion(-1 if action == "previous" else 1)
            else:
                self.handle_playback_action(action)

        def keyboard_callback(self, key: bytes) -> bool:
            if key.lower() in (b"p", b"n"):
                self.select_motion(-1 if key.lower() == b"p" else 1)
                return True
            return super().keyboard_callback(key)

        def description_lines(self) -> list[str]:
            description = (descriptions or {}).get(original_key(self.clip_key()))
            text = description.text if description and description.text else "No official description available for this clip."
            if description and original_key(self.clip_key()) != self.clip_key():
                text = "Original description (mirrored clip): " + text
            return textwrap.wrap(text, max(35, (self.window_size[0] - 24) // 10))

        def header_height(self) -> int:
            return 172 + max(0, len(self.description_lines()) - 1) * 22

        def draw_controls(self) -> None:
            width, _ = self.window_size
            self.buttons = []
            self.rectangle(0, 0, width, self.header_height(), (0.97, 0.98, 0.99, 1))
            self.button("Previous [P]", 12, 12, 115, "previous", self.navigation_enabled(-1))
            self.button("Next [N]", 137, 12, 105, "next", self.navigation_enabled(1))
            self.text(f"Motion {self.motion_index + 1} / {len(self.paths)}", 266, 33, large=True)
            self.button("Close [Esc]", width - 112, 12, 100, "close")
            self.text(self.clip_key()[:max(30, (width - 24) // 7)], 12, 65)
            description_lines = self.description_lines()
            for index, line in enumerate(description_lines):
                self.text(line, 12, 94 + index * 22, large=True)
            extra = (len(description_lines) - 1) * 22
            description = (descriptions or {}).get(original_key(self.clip_key()))
            props = description.props if description else "Metadata unavailable"
            self.text(textwrap.shorten(f"Props: {props}", max(40, (width - 24) // 7), placeholder="..."), 12, 120 + extra)
            if self.show_mesh:
                if self.mesh is not None:
                    y_values = self.mesh.frame(self.frame_index()).vertices[:, 1]
                    message = f"Lowest surface: {y_values.min()*1000:+.1f} mm | Red: below floor | Yellow: 0-{self.contact_threshold*1000:g} mm | Blue: higher"
                else:
                    message = self.mesh_error or ("Loading original Uniform surface..." if self.motion is not None else self.status)
                self.text(message[:max(30, (width - 24) // 7)], 12, 148 + extra)
            self.draw_playback_controls(self.status or "P / N: previous / next motion")

    return CanonicalizedBvhViewer()


def main() -> None:
    arguments = parse_viewer_arguments()
    instance = None
    try:
        paths = read_bvh_paths(arguments)
        motion = load_canonicalized_bvh(paths[0], arguments.unit)
        print(f"Playlist: {len(paths)} motions | First: {paths[0].name}")
        if arguments.check:
            for index, path in enumerate(paths):
                selected = motion if index == 0 else load_canonicalized_bvh(path, arguments.unit)
                if arguments.mesh:
                    key = Path(path.parent.name) / path.name
                    load_aligned_uniform_mesh(arguments.uniform_root / key, arguments.uniform_rig, selected)
            print("OK: playlist loading" + (" and mesh alignment" if arguments.mesh else ""))
            return
        try:
            descriptions = load_motion_descriptions(arguments.metadata)
        except (OSError, ValueError, csv.Error) as error:
            print(f"Descriptions unavailable: {error}", file=sys.stderr)
            descriptions = {}
        instance = create_canonicalized_bvh_viewer(
            motion, arguments.speed, paths, arguments.unit, show_mesh=arguments.mesh,
            uniform_root=arguments.uniform_root, uniform_rig=arguments.uniform_rig, descriptions=descriptions)
        instance.run()
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
    finally:
        if instance is not None:
            instance.shutdown()


if __name__ == "__main__":
    main()
