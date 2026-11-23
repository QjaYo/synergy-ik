"""One feature order and standardization path for cached and online samples."""

from dataclasses import dataclass, asdict

import torch
from torch import nn

from pose import vectors_in_root


@dataclass(frozen=True)
class FeatureSchema:
    rotation_count: int = 23
    position_count: int = 4
    ee_count: int = 4
    contact_width: int = 4
    positions: str = "ee"
    coordinates: str = "full-root-ground-origin-v1"
    angular_task: bool = False
    ee_rotations: bool = False

    @property
    def task_width(self):
        return 3 * self.ee_count * (2 if self.angular_task else 1)

    @property
    def ee_rotation_width(self):
        return 6 * self.ee_count if self.ee_rotations else 0

    @property
    def width(self):
        return (6 * self.rotation_count + 1 + 3 * self.position_count + self.task_width
                + self.ee_rotation_width + self.contact_width)

    @property
    def standardized_slice(self):
        return slice(6 * self.rotation_count, self.width - self.contact_width - self.ee_rotation_width)

    def to_dict(self):
        return asdict(self)


def relative_positions(world_positions, root_position, root_rotation, ground_y=0.0):
    origin = root_position * root_position.new_tensor([1, 0, 1])
    origin = origin + root_position.new_tensor([0, ground_y, 0])
    return vectors_in_root(world_positions - origin[..., None, :], root_rotation)


def assemble_features(rotations, root_positions, positions, x_dot, contact, rotation_indices, ground_y=0.0,
                      *, ee_rotations=None):
    """Positions/x_dot are already in the current root frame; rotations are original."""
    rotation_6d = rotations[..., list(rotation_indices), :, :2].transpose(-1, -2).flatten(-3)
    fields = [rotation_6d, root_positions[..., 1:2] - ground_y, positions.flatten(-2), x_dot.flatten(-2)]
    if ee_rotations is not None:
        fields.append(ee_rotations.flatten(-2))
    fields.append(contact.to(rotations.dtype))
    return torch.cat(fields, -1)


class Standardizer(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, features):
        return (features - self.mean) / self.scale


def assemble_history_features(*args, **kwargs):
    raise NotImplementedError("Explicit velocity history is a later comparison")
