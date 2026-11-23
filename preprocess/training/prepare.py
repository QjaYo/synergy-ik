"""Build grouped manifests, per-clip NPZ samples and training-only statistics."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from functools import partial
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from contact import UnderPressureContact, contact_from_dict, load_contact, detect_sequence
from contact.underpressure import canonical_fingerprint
from features import FeatureSchema, assemble_features, relative_positions
from pose import (SkeletonLayout, forward_kinematics, vectors_in_root, pose_arrays,
                  angular_displacement, ee_rotations_in_root)
from util import load_canonicalized_bvh


CACHE_VERSION = 4

# Cached for later feature comparisons; absent from the baseline predictor input.
COMPARISON_FIELDS = ("joint_relpos", "q_dot_prev", "x_dot_prev")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_records(data_root, database, mirrors, selected=None, source_groups=None):
    """Read only; semantic review labels do not gate the baseline population."""
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT f.path, f.original_path, f.mirrored, m.action_name, m.human_decision "
                          "FROM files f JOIN motions m ON m.original_path=f.original_path "
                          "WHERE f.present=1 AND m.present=1 ORDER BY f.path").fetchall()
    available = {row[0] for row in rows}
    if selected is not None and selected - available:
        raise ValueError(f"Selected clips are absent from review catalog: {sorted(selected - available)[:5]}")
    records = []
    source_groups = source_groups or {}
    for path, original, mirrored, action, decision in rows:
        if decision == "quality_exclude" or (mirrors == "exclude" and mirrored):
            continue
        if selected is not None and path not in selected:
            continue
        key = Path(path)
        if key.is_absolute() or ".." in key.parts:
            raise ValueError(f"Expected a data-root-relative clip key: {path}")
        if not (data_root / key).is_file():
            raise FileNotFoundError(data_root / key)
        records.append(dict(source=key.as_posix(), group=source_groups.get(original, original),
                            mirrored=bool(mirrored), action=action))
    if not records:
        raise ValueError("No eligible clips selected")
    return records


def assign_splits(records, seed):
    """Shuffle whole recording groups once; mirrors/declared derivatives cannot leak."""
    groups = sorted({row["group"] for row in records})
    np.random.default_rng(seed).shuffle(groups)
    train_end = round(0.8 * len(groups))
    valid_end = round(0.9 * len(groups))
    assignment = {group: "train" if i < train_end else "validation" if i < valid_end else "test"
                  for i, group in enumerate(groups)}
    return [dict(row, split=assignment[row["group"]]) for row in records]


def prepare_clip(job, *, contact_variants=None, angular_task=False, ee_rotation_features=False):
    source, destination, contact_data, frame_limit = job
    torch.set_num_threads(1)
    source, destination = Path(source), Path(destination)
    contact = contact_from_dict(contact_data)
    motion = load_canonicalized_bvh(source)
    if frame_limit is not None:
        motion.poses = motion.poses[:frame_limit]
    if motion.num_frames() < 3:
        raise ValueError(f"At least three frames are required: {source}")
    layout = SkeletonLayout.from_skeleton(motion.skel)
    _, _, static = pose_arrays(motion.poses[0])
    data = np.stack([pose.data for pose in motion.poses])
    if np.any(data[:, 1:, :3, 3] != 0):
        raise ValueError(f"Non-root dynamic translations are unsupported: {source}")
    rotations = torch.as_tensor(data[:, :, :3, :3], dtype=torch.float64)
    roots = torch.as_tensor(data[:, 0, :3, 3], dtype=torch.float64)
    world_R, world_p = forward_kinematics(rotations, roots, torch.as_tensor(static), layout.parents)
    dt = 1 / motion.fps
    detectors = ({name: contact_from_dict(value) for name, value in contact_variants.items()}
                 if contact_variants else {"contact": contact})
    if any(detector.method == "underpressure-offline-v1" for detector in detectors.values()):
        raise ValueError("Training contact must not use future frames")
    contacts = {name: detect_sequence(detector, motion, world_R.numpy(), world_p.numpy())[1:-1]
                for name, detector in detectors.items()}
    c = next(iter(contacts.values()))
    current_R = rotations[1:-1, 0]
    ee = world_p[:, list(layout.ee_indices)]
    positions = relative_positions(ee[1:-1], roots[1:-1], current_R, contact.ground_y)
    x_dot = vectors_in_root((ee[2:] - ee[1:-1]) / dt, current_R)
    ee_world_R = world_R[:, list(layout.ee_indices)]
    ee_omega_world = torch.from_numpy(angular_displacement(ee_world_R[1:-1].numpy(),
                                                         ee_world_R[2:].numpy()) / dt)
    ee_angular_velocity = vectors_in_root(ee_omega_world, current_R)
    ee_relrot6d = ee_rotations_in_root(ee_world_R[1:-1], current_R)
    root_velocity = vectors_in_root(((roots[2:] - roots[1:-1]) / dt).unsqueeze(-2), current_R).squeeze(-2)
    active = list(layout.rotation_indices)
    relative_R = (rotations[1:-1, active].transpose(-1, -2) @ rotations[2:, active]).numpy()
    omega = Rotation.from_matrix(relative_R.reshape(-1, 3, 3)).as_rotvec().reshape(-1, len(active), 3) / dt
    q_dot = np.concatenate((root_velocity.numpy(), omega.reshape(len(omega), -1)), axis=1)
    joint_positions = relative_positions(world_p[1:-1], roots[1:-1], current_R, contact.ground_y)
    x_dot_prev = vectors_in_root((ee[1:-1] - ee[:-2]) / dt, current_R)
    root_velocity_prev = vectors_in_root(
        ((roots[1:-1] - roots[:-2]) / dt).unsqueeze(-2), current_R).squeeze(-2)
    previous_relative_R = (rotations[:-2, active].transpose(-1, -2) @ rotations[1:-1, active]).numpy()
    omega_prev = Rotation.from_matrix(previous_relative_R.reshape(-1, 3, 3)).as_rotvec().reshape(-1, len(active), 3) / dt
    q_dot_prev = np.concatenate((root_velocity_prev.numpy(), omega_prev.reshape(len(omega_prev), -1)), axis=1)
    schema = FeatureSchema(len(active), len(layout.ee_indices), len(layout.ee_indices), contact.width,
                           angular_task=angular_task, ee_rotations=ee_rotation_features)
    metadata = dict(version=CACHE_VERSION, layout=layout.to_dict(), schema=schema.to_dict(),
                    contact=contact_data, fps=motion.fps,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), frame_limit=frame_limit)
    if contact_variants:
        metadata.update(contact_variants=contact_variants, default_contact=next(iter(contact_variants)))
    contact_arrays = ({"contact_" + name: value for name, value in contacts.items()}
                      if contact_variants else contacts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, rotations=rotations.numpy().astype(np.float32),
                        root_positions=roots.numpy().astype(np.float32), static=static.astype(np.float32),
                        ee_relpos=positions.numpy().astype(np.float32), x_dot=x_dot.numpy().astype(np.float32),
                        **contact_arrays, q_dot=q_dot.astype(np.float32),
                        joint_relpos=joint_positions.numpy().astype(np.float32),
                        q_dot_prev=q_dot_prev.astype(np.float32), x_dot_prev=x_dot_prev.numpy().astype(np.float32),
                        ee_rotations_world=ee_world_R.numpy().astype(np.float32),
                        ee_angular_velocity=ee_angular_velocity.numpy().astype(np.float32),
                        ee_relrot6d=ee_relrot6d.numpy().astype(np.float32),
                        metadata=json.dumps(metadata))
    return dict(samples=len(c), frames=motion.num_frames(), bytes=destination.stat().st_size,
                source_sha256=metadata["source_sha256"], fps=motion.fps,
                layout=layout.to_dict(), schema=schema.to_dict())


def compute_statistics(output, records, layout, schema, ground_y, contact_key="contact"):
    """Stable streaming population moments; rotation/contact features are not scaled."""
    count = 0
    mean = np.zeros(schema.width, dtype=np.float64)
    m2 = np.zeros_like(mean)
    velocity_sum = np.zeros(2)
    comparison_moments = {}
    for record in records:
        if record["split"] != "train":
            continue
        with np.load(output / record["cache"], allow_pickle=False) as clip:
            task_velocity = torch.from_numpy(clip["x_dot"]).double()
            if schema.angular_task:
                task_velocity = torch.cat((task_velocity, torch.from_numpy(clip["ee_angular_velocity"]).double()), -2)
            ee_rotations = torch.from_numpy(clip["ee_relrot6d"]).double() if schema.ee_rotations else None
            features = assemble_features(
                torch.from_numpy(clip["rotations"][1:-1]).double(),
                torch.from_numpy(clip["root_positions"][1:-1]).double(),
                torch.from_numpy(clip["ee_relpos"]).double(), task_velocity,
                torch.from_numpy(clip[contact_key]), layout.rotation_indices, ground_y,
                ee_rotations=ee_rotations).numpy()
            size = len(features)
            for name in COMPARISON_FIELDS:
                values = clip[name].astype(np.float64).reshape(size, -1)
                current_mean = values.mean(axis=0)
                if name not in comparison_moments:
                    comparison_moments[name] = [np.zeros(values.shape[1]), np.zeros(values.shape[1])]
                running_mean, running_m2 = comparison_moments[name]
                shift = current_mean - running_mean
                running_m2 += ((values - current_mean) ** 2).sum(axis=0) + shift ** 2 * count * size / (count + size)
                running_mean += shift * size / (count + size)
            batch_mean = features.mean(axis=0)
            delta = batch_mean - mean
            m2 += ((features - batch_mean) ** 2).sum(axis=0) + delta ** 2 * count * size / (count + size)
            mean += delta * size / (count + size)
            count += size
            target = clip["q_dot"].astype(np.float64)
            velocity_sum += ((target[:, :3] ** 2).sum(), (target[:, 3:] ** 2).sum())
    if count == 0:
        raise ValueError("The manifest has no training samples")
    scale = np.sqrt(m2 / count)
    scale[scale == 0] = 1
    keep = np.zeros(schema.width, dtype=bool)
    keep[schema.standardized_slice] = True
    mean[~keep], scale[~keep] = 0, 1
    rms = np.sqrt(velocity_sum / (count * np.array([3, layout.dofs - 3])))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError(f"Training velocity RMS is degenerate: {rms}; inspect the selected motions")
    comparisons = {}
    for name, (field_mean, field_m2) in comparison_moments.items():
        field_scale = np.sqrt(field_m2 / count)
        field_scale[field_scale == 0] = 1
        comparisons[name] = dict(mean=field_mean.tolist(), scale=field_scale.tolist())
    return dict(count=count, mean=mean.tolist(), scale=scale.tolist(),
                root_rms=float(rms[0]), angular_rms=float(rms[1]), comparison_inputs=comparisons)


def build_dataset(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Dataset already exists: {manifest_path}; choose a new output directory")
    variants = None
    if args.contact_variants:
        variants = {}
        for item in args.contact_variants:
            name, filename = item.split("=", 1)
            if not name.isidentifier() or name in variants:
                raise ValueError(f"Contact variant names must be unique identifiers: {name}")
            variants[name] = load_contact(filename).to_dict()
        if len({value["ground_y"] for value in variants.values()}) != 1:
            raise ValueError("Contact variants must use the same ground plane")
        contact = contact_from_dict(next(iter(variants.values())))
    else:
        contact = load_contact(args.contact_config)
    contact_data = contact.to_dict()
    reference = None
    if args.statistics_from:
        reference = json.loads((args.statistics_from / "manifest.json").read_text(encoding="utf-8"))
        recorded = reference["contract"]["contact"]
        if isinstance(contact, UnderPressureContact) and "fingerprint" in recorded:
            comparable = dict(recorded, fingerprint=canonical_fingerprint(recorded["fingerprint"]))
            if comparable == contact_data:
                # Keep the reference's exact contract in NPZs and manifests for checkpoint evaluation.
                contact_data = recorded
    selected = None
    if args.bvh_list:
        selected = {line.strip().replace("\\", "/") for line in args.bvh_list.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")}
    groups = json.loads(args.source_groups.read_text(encoding="utf-8")) if args.source_groups else None
    records = assign_splits(select_records(args.data_root, args.review_db, args.mirrors, selected, groups), args.seed)
    split_source = reference
    if args.split_manifest:
        split_source = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split_source is not None:
        assignments = {r["group"]: r["split"] for r in split_source["clips"]}
        for row in records:
            row["split"] = assignments[row["group"]]
    for row in records:
        row["cache"] = "clips/" + str(Path(row["source"]).with_suffix(".npz")).replace("\\", "/")
    selection = dict(version=CACHE_VERSION, seed=args.seed, mirrors=args.mirrors,
                     data_root=str(args.data_root.resolve()), frame_limit=args.frame_limit,
                     grouping="review original_path + optional declared source groups",
                     contact=contact_data, clips=records)
    if variants:
        selection.update(default_contact=next(iter(variants)))
    selection.update(task=args.task, ee_rotation_features=not args.no_ee_rotation_features)
    write_json(args.output / "selection.json", selection)
    if args.manifest_only:
        print(f"Selected {len(records)} clips; written to {args.output / 'selection.json'}")
        return
    jobs = [(args.data_root / row["source"], args.output / row["cache"], contact_data, args.frame_limit)
            for row in records]
    executor = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    prepare = partial(prepare_clip, contact_variants=variants, angular_task=args.task == "pose",
                      ee_rotation_features=not args.no_ee_rotation_features)
    results = executor.map(prepare, jobs) if executor else map(prepare, jobs)
    layout = schema = None
    try:
        for i, (row, result) in enumerate(zip(records, results)):
            current_layout = SkeletonLayout.from_dict(result.pop("layout"))
            current_schema = FeatureSchema(**result.pop("schema"))
            if layout is None:
                layout, schema = current_layout, current_schema
            if current_layout != layout or current_schema != schema:
                raise ValueError(f"Incompatible canonical topology: {row['source']}")
            row.update(result)
            if (i + 1) % 100 == 0 or i + 1 == len(records):
                print(f"Prepared {i+1}/{len(records)} clips", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    contract = dict(version=CACHE_VERSION, layout=layout.to_dict(), schema=schema.to_dict(), contact=contact_data,
                    units=dict(length="m", angle="rad", time="s"),
                    timing="current=t; history=t-1->t; target=t->t+1",
                    comparison_fields=list(COMPARISON_FIELDS) + ["ee_rotations_world"])
    contact_key = "contact"
    if variants:
        contact_key = "contact_" + next(iter(variants))
        contract.update(contact_key=contact_key)
    if reference is not None:
        if digest(reference["contract"]) != digest(contract):
            raise ValueError("Statistics reference uses a different feature/contact contract")
        reference_splits = {r["group"]: r["split"] for r in reference["clips"]}
        if any(reference_splits[r["group"]] != r["split"] for r in records):
            raise ValueError("A statistics reference must preserve its train/validation/test membership")
        statistics = json.loads((args.statistics_from / reference["statistics"]).read_text(encoding="utf-8"))
        selection["statistics_reference"] = str(args.statistics_from.resolve())
    else:
        statistics = compute_statistics(args.output, records, layout, schema, contact.ground_y, contact_key)
        statistics["training_sources_hash"] = digest(dict(
            sources=[(r["source"], r["source_sha256"]) for r in records if r["split"] == "train"],
            contract=contract, frame_limit=args.frame_limit))
    write_json(args.output / "statistics.json", statistics)
    if variants:
        selections = {}
        for name, configuration in variants.items():
            width = contact_from_dict(configuration).width
            variant_schema = dict(schema.to_dict(), contact_width=width)
            variant_contract = dict(contract, schema=variant_schema, contact=configuration,
                                    contact_key="contact_" + name)
            variant_statistics = dict(statistics,
                mean=statistics["mean"][:-schema.contact_width] + [0.0] * width,
                scale=statistics["scale"][:-schema.contact_width] + [1.0] * width)
            variant_statistics["training_sources_hash"] = digest(dict(
                sources=[(r["source"], r["source_sha256"]) for r in records if r["split"] == "train"],
                contract=variant_contract, frame_limit=args.frame_limit))
            statistics_path = "statistics.json" if name == next(iter(variants)) else f"statistics_{name}.json"
            write_json(args.output / statistics_path, variant_statistics)
            selections[name] = dict(contract=variant_contract, statistics=statistics_path)
        selection["contact_variants"] = selections
    selection.update(contract=contract, contract_hash=digest(contract), statistics="statistics.json")
    write_json(manifest_path, selection)
    summary = dict(clips=len(records), groups=len({r["group"] for r in records}),
                   originals=sum(not r["mirrored"] for r in records), mirrors=sum(r["mirrored"] for r in records),
                   samples={split: sum(r["samples"] for r in records if r["split"] == split)
                            for split in ("train", "validation", "test")},
                   cache_bytes=sum(r["bytes"] for r in records), seconds=time.perf_counter()-started,
                   input_width=schema.width, dofs=layout.dofs)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--review-db", type=Path, required=True)
    contact_arguments = parser.add_mutually_exclusive_group()
    contact_arguments.add_argument("--contact-config", "--sole-config", dest="contact_config", type=Path,
                        help="Contact-method JSON; --sole-config remains an alias for existing commands")
    contact_arguments.add_argument("--contact-variants", nargs="+", metavar="NAME=JSON",
                        help="Store named contact arrays in one NPZ; the first is the default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("position", "pose"), default="pose",
                        help="IK task: EE positions alone or positions and world orientations")
    parser.add_argument("--no-ee-rotation-features", action="store_true",
                        help="Omit current root-relative EE rotations from the W input")
    parser.add_argument("--mirrors", choices=("include", "exclude"), default="include")
    parser.add_argument("--bvh-list", type=Path, help="Optional UTF-8 subset of data-root-relative clip keys")
    parser.add_argument("--source-groups", type=Path, help="JSON original-path -> shared source-recording group")
    parser.add_argument("--split-manifest", type=Path, help="Reuse groups' existing split membership")
    parser.add_argument("--statistics-from", type=Path, help="Reuse a prepared reference training pool's input/RMS scales")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-limit", type=int, help="Explicit diagnostic truncation; omitted for full preprocessing")
    args = parser.parse_args(argv)
    if args.contact_config is None and args.contact_variants is None:
        args.contact_config = Path(__file__).with_name("ankle_contact.json")
    if args.contact_variants and args.statistics_from:
        parser.error("Build shared training statistics once for a multi-contact dataset; omit --statistics-from")
    if args.workers < 1 or (args.frame_limit is not None and args.frame_limit < 3):
        parser.error("workers must be positive; frame-limit must be at least three")
    return args


if __name__ == "__main__":
    build_dataset(parse_arguments())
