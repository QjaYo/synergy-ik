#!/usr/bin/env python3
"""Convert SOMA Uniform BVH files from centimeters to meters.

Remove the identity wrapper Root, fingers, jaw and eyes; promote Hips to root
and zero its static OFFSET while preserving its animated translation.
Rename selected body joints and add fixed LeftHandEnd/RightHandEnd markers.
Frame rate and retained local rotations are preserved.

Run in the synergy environment from the project directory::

    python preprocess/canonicalize.py --input-dir PATH --output-dir PATH -j 16

Input subdirectories are reproduced under the output directory. Resume metadata
is stored in OUTPUT/.state/canonical_bvh_log.txt, keyed by absolute source path,
size and modification time. An output directory is tied to its recorded input
directory. Existing nonempty outputs with matching metadata
are skipped. Use --force after changing conversion settings or code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

# Import Fairmotion from the checked-out submodule.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "third_party" / "fairmotion"))

import numpy as np
from fairmotion.core.motion import Joint, Motion, Skeleton
from fairmotion.data import bvh
from fairmotion.ops import conversions
from tqdm.auto import tqdm


OUTPUT_SCALE = 0.01  # Source centimeters -> canonical BVH meters
FINGER_PARTS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
REMOVE = {"Root", "Jaw", "LeftEye", "RightEye"}
RENAME = {
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
}

# Fixed Hand-local endpoint offsets in source centimeters.
# These constants place each marker halfway from the wrist to the mean of the
# four non-thumb MCP offsets used to define this skeleton. They are not
# recalculated per file and are not independent articulated joints.
PALM_OFFSET = {
    "Left": np.array([1.519914875, 0.182887750, 0.051776500]),
    "Right": np.array([-1.521701750, -0.185643625, -0.055050875]),
}


class CanonicalizeError(RuntimeError):
    pass


def fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def is_finger(name: str) -> bool:
    prefixes = tuple(
        f"{side}Hand{finger}"
        for side in ("Left", "Right")
        for finger in FINGER_PARTS
    )
    return name.startswith(prefixes)


def validate_source(motion: Motion, source: Path) -> None:
    root = motion.skel.root_joint
    if root.name != "Root" or len(root.child_joints) != 1:
        raise CanonicalizeError(f"{source.name}: expected Root -> Hips")
    if root.child_joints[0].name != "Hips":
        raise CanonicalizeError(f"{source.name}: Root child must be Hips")
    if not np.allclose(root.xform_from_parent_joint, np.eye(4), rtol=0, atol=1e-6):
        raise CanonicalizeError(f"{source.name}: Root static transform is not identity")
    root_index = motion.skel.get_index_joint("Root")
    for frame, pose in enumerate(motion.poses):
        if not np.allclose(pose.data[root_index], np.eye(4), rtol=0, atol=1e-6):
            raise CanonicalizeError(
                f"{source.name}: Root transform is not identity at frame {frame}"
            )


def canonicalize_motion(old_motion: Motion, source: Path) -> Motion:
    validate_source(old_motion, source)
    old_skel = old_motion.skel
    hips = old_skel.get_joint("Hips")
    removed = REMOVE | {j.name for j in old_skel.joints if is_finger(j.name)}
    retained = [j for j in old_skel.joints if j.name not in removed]

    new_skel = Skeleton(
        name=old_skel.name,
        v_up=old_skel.v_up.copy(),
        v_face=old_skel.v_face.copy(),
        v_up_env=old_skel.v_up_env.copy(),
    )
    old_to_new: dict[str, Joint] = {}
    new_to_old: dict[str, str] = {}

    for old_joint in retained:
        new_name = RENAME.get(old_joint.name, old_joint.name)
        base_transform = old_joint.xform_from_parent_joint.copy()
        if old_joint is hips:
            # Hips becomes the BVH root. Keep its per-frame translation, but
            # canonicalize the root's static OFFSET to the origin.
            base_transform[:3, 3] = 0.0
        new_joint = Joint(
            name=new_name,
            dof=6 if old_joint is hips else 3,
            xform_from_parent_joint=base_transform,
        )
        if old_joint is hips:
            parent = None
        else:
            parent_name = old_joint.parent_joint.name
            if parent_name in removed:
                raise CanonicalizeError(
                    f"cannot retain {old_joint.name} after removing {parent_name}"
                )
            parent = old_to_new[parent_name]
        new_skel.add_joint(new_joint, parent)
        old_to_new[old_joint.name] = new_joint
        new_to_old[new_name] = old_joint.name

        if old_joint.name in ("LeftHand", "RightHand"):
            side = old_joint.name.removesuffix("Hand")
            hand_end = Joint(
                name=f"{side}HandEnd",
                dof=3,
                xform_from_parent_joint=conversions.p2T(PALM_OFFSET[side]),
            )
            new_skel.add_joint(hand_end, new_joint)

    new_motion = Motion(name=old_motion.name, skel=new_skel, fps=old_motion.fps)
    for old_pose in old_motion.poses:
        frame = []
        for new_joint in new_skel.joints:
            if new_joint.name.endswith("HandEnd"):
                # Identity pose means zero local rotation. The static palm
                # position is already stored in xform_from_parent_joint.
                frame.append(np.eye(4))
            else:
                old_name = new_to_old[new_joint.name]
                frame.append(
                    old_pose.data[old_skel.get_index_joint(old_name)].copy()
                )
        new_motion.add_one_frame(frame)
    return new_motion


def canonicalize_file(source: Path, destination: Path) -> None:
    motion = bvh.load(str(source))
    canonical = canonicalize_motion(motion, source)
    bvh.save(canonical, str(destination), scale=OUTPUT_SCALE, rot_order="ZYX")


def worker(
    source_str: str, destination_str: str, source_key: str
) -> tuple[str, str, int, int, str]:
    source, destination = Path(source_str), Path(destination_str)
    size, mtime_ns = fingerprint(source)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        canonicalize_file(source, temporary)
        os.replace(temporary, destination)
        return "OK", source_key, size, mtime_ns, ""
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return "ERROR", source_key, size, mtime_ns, f"{type(exc).__name__}: {exc}"


def load_completed(log_path: Path) -> dict[str, tuple[int, int]]:
    """Read latest per-file status from the tab-separated metadata TXT."""
    latest: dict[str, tuple[int, int] | None] = {}
    if not log_path.exists():
        return {}
    for number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t", 4)
        if len(fields) < 4:
            print(f"warning: ignoring metadata line {number}", file=sys.stderr)
            continue
        status, name, size, mtime_ns = fields[:4]
        try:
            latest[name] = (int(size), int(mtime_ns)) if status == "OK" else None
        except ValueError:
            print(f"warning: ignoring metadata line {number}", file=sys.stderr)
    return {name: value for name, value in latest.items() if value is not None}


def append_log(log_path: Path, result: tuple[str, str, int, int, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as stream:
        if new_file:
            stream.write("# status\tfile\tsize\tmtime_ns\terror\n")
        stream.write("\t".join(map(str, result)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SOMA Uniform BVH (cm) to canonical body BVH (m).",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="directory containing source BVH files in centimeters")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="separate directory for canonical BVH files in meters")
    parser.add_argument(
        "-j", "--workers", type=int, default=os.cpu_count() or 1,
        help="worker processes (default: logical CPU count)",
    )
    parser.add_argument("--force", action="store_true", help="regenerate all files")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    if (input_dir == output_dir or input_dir in output_dir.parents
            or output_dir in input_dir.parents):
        parser.error("input and output directories must be separate, non-nested paths")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"output path is not a directory: {output_dir}")
    log_path = output_dir / ".state" / "canonical_bvh_log.txt"
    sources = sorted(input_dir.rglob("*.bvh"))
    if not sources:
        parser.error(f"no BVH files found: {input_dir}")
    input_header = f"# input-dir\t{input_dir.as_posix()}\n"
    if log_path.exists():
        with log_path.open(encoding="utf-8") as stream:
            if stream.readline() != input_header:
                parser.error("output metadata belongs to another input directory; "
                             "use a separate output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(input_header + "# status\tfile\tsize\tmtime_ns\terror\n",
                            encoding="utf-8")
    completed = {} if args.force else load_completed(log_path)
    pending = []
    skipped_bytes = 0
    for source in sources:
        relative = source.relative_to(input_dir)
        source_key = source.as_posix()
        destination = output_dir / relative
        if (
            completed.get(source_key) == fingerprint(source)
            and destination.is_file()
            and destination.stat().st_size > 0
        ):
            skipped_bytes += source.stat().st_size
        else:
            pending.append((source, destination, source_key))

    skipped = len(sources) - len(pending)
    workers = min(args.workers, max(1, len(pending)))
    total_bytes = sum(source.stat().st_size for source in sources)
    completed_count = 0
    failures = 0
    with tqdm(
        total=total_bytes,
        initial=skipped_bytes,
        desc="Canonicalizing",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(ok=0, failed=0, skipped=skipped, workers=workers)
        if pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        worker, str(source), str(destination), source_key
                    ): source.stat().st_size
                    for source, destination, source_key in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    append_log(log_path, result)
                    if result[0] == "OK":
                        completed_count += 1
                    else:
                        failures += 1
                        tqdm.write(f"FAIL {result[1]}: {result[4]}", file=sys.stderr)
                    progress.update(futures[future])
                    progress.set_postfix(
                        ok=completed_count,
                        failed=failures,
                        skipped=skipped,
                        workers=workers,
                    )
    print(f"done={len(pending) - failures} failed={failures} skipped={skipped}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
