"""Read-only representative-motion selection and shared contact-viewer arguments."""

from contextlib import closing
import argparse
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "preprocess/motion_review"))
from motion_review_data import DEFAULT_METADATA, load_motion_descriptions, original_key
from uniform_mesh import DEFAULT_UNIFORM_ROOT, DEFAULT_UNIFORM_RIG
from visualize_canonicalized_bvh import read_bvh_paths


@dataclass(frozen=True)
class Entry:
    group: str
    key: str


def catalog(db, search=""):
    """One original per review group, excluding defective recordings only."""
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT action_name, original_path FROM motions "
                                  "WHERE present=1 AND COALESCE(human_decision,'') != 'quality_exclude' "
                                  "ORDER BY action_name, original_path").fetchall()
    grouped = {}
    for group, key in rows:
        if search.lower() in (group + " " + key).lower():
            grouped.setdefault(group, []).append(key)
    entries = []
    for group, keys in grouped.items():
        key = min(keys, key=lambda k: (any(s in k.lower() for s in ("_start", "_stop", "_end_")), k))
        entries.append(Entry(group, key))
    priorities = ("walking", "walk", "jump_and_land", "running", "run", "dancing", "cartwheel")
    def order(entry):
        rank = next((i for i, word in enumerate(priorities) if entry.group.startswith(word)), len(priorities))
        return rank, entry.group
    return sorted(entries, key=order)


def add_viewer_arguments(parser):
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/canonical_bvh")
    parser.add_argument("--db", type=Path, default=ROOT / "preprocess/motion_review/results/review.sqlite3")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--uniform-root", type=Path, default=DEFAULT_UNIFORM_ROOT)
    parser.add_argument("--uniform-rig", type=Path, default=DEFAULT_UNIFORM_RIG)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--bvh-file", type=Path)
    inputs.add_argument("--bvh-list", type=Path, help="BVH paths relative to the list file, one per line")
    inputs.add_argument("--filter", default="", help="Filter review groups and filenames")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check", action="store_true", help="Compute contacts without opening a window")


def select_entries(args):
    if not np.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("Playback speed must be positive and finite")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Playlist limit must be positive")
    if args.bvh_file or args.bvh_list:
        entries = [Entry(path.stem, path.relative_to(args.data_root.resolve()).as_posix())
                   for path in read_bvh_paths(args)]
    else:
        entries = catalog(args.db, args.filter)
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError("No matching nondefective motions")
    return entries
