"""Bounded-memory batches from prepared clips; no BVH or contact recomputation."""

import json
from dataclasses import replace
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from features import FeatureSchema, assemble_features, convert_root_statistics
from pose import SkeletonLayout


def batch_loader(batches: IterableDataset, *, workers: int, pin_memory: bool, seed: int) -> DataLoader:
    """Batches are already grouped by clip; workers prefetch complete batches."""
    return DataLoader(batches, batch_size=None, num_workers=workers, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed))


class PreparedDataset:
    def __init__(self, directory: str | Path, contact_mode: str | None = None,
                 *, root_rotation: str | None = None) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.contract = self.manifest["contract"]
        statistics_path = self.manifest["statistics"]
        if "contact_variants" in self.manifest:
            name = contact_mode or self.manifest["default_contact"]
            variant = self.manifest["contact_variants"][name]
            self.contract, statistics_path = variant["contract"], variant["statistics"]
        elif contact_mode is not None:
            raise ValueError("This dataset stores one contact method; omit --contact-mode")
        if self.contract["version"] not in (1, 2, 3, 4, 5):
            raise ValueError(f"Unsupported cache version: {self.contract['version']}")
        self.layout = SkeletonLayout.from_dict(self.contract["layout"])
        self.schema = FeatureSchema.from_dict(self.contract["schema"])
        if self.schema.positions != "ee":
            raise NotImplementedError("The first cached feature path uses EE positions")
        self.contact_key = self.contract.get("contact_key", "contact")
        self.statistics = json.loads((self.directory / statistics_path).read_text(encoding="utf-8"))
        if root_rotation is not None and root_rotation != self.schema.root_rotation:
            original_schema = self.schema
            self.schema = replace(self.schema, root_rotation=root_rotation)
            self.statistics = convert_root_statistics(self.statistics, original_schema, self.schema)
            self.contract = dict(self.contract, schema=self.schema.to_dict())

    def batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        epoch: int = 0,
        seed: int = 42,
    ) -> 'ClipBatches':
        return ClipBatches(self, split, batch_size, shuffle, epoch, seed)


class ClipBatches(IterableDataset):
    """Shuffle clips and their samples; visit every selected transition once per epoch."""
    def __init__(
        self,
        dataset: PreparedDataset,
        split: str,
        batch_size: int,
        shuffle: bool,
        epoch: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.records = [row for row in dataset.manifest["clips"] if row["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} clips in this dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size, self.shuffle, self.epoch, self.seed = batch_size, shuffle, epoch, seed

    def __len__(self) -> int:
        return sum((r["samples"] + self.batch_size - 1) // self.batch_size for r in self.records)

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.records)) if self.shuffle else np.arange(len(self.records))
        sample_rng = np.random.default_rng(self.seed + self.epoch * 10007 + worker_id)
        for record_index in order[worker_id::worker_count]:
            record = self.records[record_index]
            with np.load(self.dataset.directory / record["cache"], allow_pickle=False) as archive:
                fields = ("rotations", "root_positions", "static", "ee_relpos", "x_dot", "q_dot")
                if self.dataset.schema.angular_task:
                    fields += ("ee_angular_velocity",)
                if self.dataset.schema.ee_rotations:
                    fields += ("ee_relrot6d",)
                data = {key: archive[key] for key in fields}
                data["contact"] = archive[self.dataset.contact_key]
            indices = sample_rng.permutation(record["samples"]) if self.shuffle else np.arange(record["samples"])
            for start in range(0, len(indices), self.batch_size):
                sample = indices[start:start + self.batch_size]
                frame = sample + 1
                rotations = torch.from_numpy(data["rotations"][frame])
                roots = torch.from_numpy(data["root_positions"][frame])
                x_dot = torch.from_numpy(data["x_dot"][sample])
                if self.dataset.schema.angular_task:
                    x_dot = torch.cat((x_dot, torch.from_numpy(data["ee_angular_velocity"][sample])), dim=-2)
                ee_rotations = (torch.from_numpy(data["ee_relrot6d"][sample])
                                if self.dataset.schema.ee_rotations else None)
                contact = torch.from_numpy(data["contact"][sample])
                features = assemble_features(rotations, roots, torch.from_numpy(data["ee_relpos"][sample]),
                                             x_dot, contact, self.dataset.layout.rotation_indices,
                                             self.dataset.contract["contact"]["ground_y"], ee_rotations=ee_rotations,
                                             root_rotation=self.dataset.schema.root_rotation)
                yield dict(features=features, rotations=rotations, root_positions=roots,
                           static=torch.from_numpy(data["static"]), x_dot=x_dot.flatten(-2),
                           target=torch.from_numpy(data["q_dot"][sample]),
                           source=record["source"], frames=frame.tolist(), angular_task=self.dataset.schema.angular_task)
