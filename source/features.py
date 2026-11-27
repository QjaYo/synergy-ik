"""One feature order and standardization path for cached and online samples."""

from dataclasses import dataclass, asdict
from collections.abc import Sequence
from typing import NoReturn

import torch
from torch import nn
import numpy as np

from pose import vectors_in_root


ROOT_ROTATIONS = ("up_vector", "rotation_6d")
DEFAULT_ROOT_ROTATION = "up_vector"


@dataclass(frozen=True)
class FeatureSchema:
    rotation_count: int = 23
    position_count: int = 4
    ee_count: int = 4
    contact_width: int = 2
    positions: str = "ee"
    coordinates: str = "full-root-ground-origin-v1"
    angular_task: bool = True
    ee_rotations: bool = True
    root_rotation: str = DEFAULT_ROOT_ROTATION

    def __post_init__(self) -> None:
        if self.root_rotation not in ROOT_ROTATIONS:
            raise ValueError(f"Unknown root rotation representation: {self.root_rotation}")

    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureSchema':
        # Caches/checkpoints written before root representation selection used 6D.
        legacy = dict(contact_width=4, angular_task=False, ee_rotations=False, root_rotation="rotation_6d")
        return cls(**(legacy | data))

    @property
    def root_width(self) -> int:
        return 3 if self.root_rotation == "up_vector" else 6

    @property
    def rotation_width(self) -> int:
        return self.root_width + 6 * (self.rotation_count - 1)

    @property
    def task_width(self) -> int:
        return 3 * self.ee_count * (2 if self.angular_task else 1)

    @property
    def ee_rotation_width(self) -> int:
        return 6 * self.ee_count if self.ee_rotations else 0

    @property
    def width(self) -> int:
        return (self.rotation_width + 1 + 3 * self.position_count + self.task_width
                + self.ee_rotation_width + self.contact_width)

    @property
    def standardized_slice(self) -> slice:
        return slice(self.rotation_width, self.width - self.contact_width - self.ee_rotation_width)

    def to_dict(self) -> dict:
        return asdict(self)


def convert_root_statistics(statistics: dict, source: FeatureSchema, target: FeatureSchema) -> dict:
    """Both root encodings are unscaled; retain every other training statistic."""
    return dict(statistics,
                mean=[0.0] * target.root_width + statistics["mean"][source.root_width:],
                scale=[1.0] * target.root_width + statistics["scale"][source.root_width:])


def relative_positions(
    world_positions: torch.Tensor,
    root_position: torch.Tensor,
    root_rotation: torch.Tensor,
    ground_y: float = 0.0,
) -> torch.Tensor:
    origin = root_position * root_position.new_tensor([1, 0, 1])
    origin = origin + root_position.new_tensor([0, ground_y, 0])
    return vectors_in_root(world_positions - origin[..., None, :], root_rotation)


def assemble_features(
    rotations: torch.Tensor,
    root_positions: torch.Tensor,
    positions: torch.Tensor,
    x_dot: torch.Tensor,
    contact: torch.Tensor,
    rotation_indices: Sequence[int],
    ground_y: float = 0.0,
    *,
    ee_rotations: torch.Tensor | None = None,
    root_rotation: str = DEFAULT_ROOT_ROTATION,
) -> torch.Tensor:
    """Keep original pose matrices; encode root first, then parent-local joints.

    rotation_indices starts with the root. Positions/x_dot use the full root
    frame. R_root.T @ world_up equals R_root's second row and has no yaw gauge.
    """
    active = rotations[..., list(rotation_indices), :, :]
    if root_rotation == "up_vector":
        joint_6d = active[..., 1:, :, :2].transpose(-1, -2).flatten(-3)
        rotation_features = torch.cat((active[..., 0, 1, :], joint_6d), -1)
    elif root_rotation == "rotation_6d":
        rotation_features = active[..., :, :2].transpose(-1, -2).flatten(-3)
    else:
        raise ValueError(f"Unknown root rotation representation: {root_rotation}")
    fields = [rotation_features, root_positions[..., 1:2] - ground_y, positions.flatten(-2), x_dot.flatten(-2)]
    if ee_rotations is not None:
        fields.append(ee_rotations.flatten(-2))
    fields.append(contact.to(rotations.dtype))
    return torch.cat(fields, -1)


class Standardizer(nn.Module):
    def __init__(
        self,
        mean: Sequence[float] | np.ndarray | torch.Tensor,
        scale: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.scale


def assemble_history_features(*args: object, **kwargs: object) -> NoReturn:
    raise NotImplementedError("Explicit velocity history is a later comparison")
