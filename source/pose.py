"""Tensor pose geometry shared by preprocessing, learning and inference."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from util import get_rotation_joint_indices


DEFAULT_EES = ("LeftHandEnd", "RightHandEnd", "LeftToeEnd", "RightToeEnd")


@dataclass(frozen=True)
class SkeletonLayout:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    rotation_indices: tuple[int, ...]
    ee_names: tuple[str, ...] = DEFAULT_EES

    @classmethod
    def from_skeleton(cls, skeleton):
        return cls(
            tuple(j.name for j in skeleton.joints),
            tuple(-1 if j.parent_joint is None else skeleton.get_index_joint(j.parent_joint)
                  for j in skeleton.joints),
            get_rotation_joint_indices(skeleton),
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: tuple(value) for key, value in data.items()})

    def to_dict(self):
        return dict(names=self.names, parents=self.parents,
                    rotation_indices=self.rotation_indices, ee_names=self.ee_names)

    @property
    def ee_indices(self):
        return tuple(self.names.index(name) for name in self.ee_names)

    @property
    def dofs(self):
        return 3 + 3 * len(self.rotation_indices)


def pose_arrays(pose):
    """Extract a canonical pose without losing leaf rotations or static offsets."""
    data = np.asarray(pose.data)
    if np.any(data[1:, :3, 3] != 0):
        raise ValueError("Canonical non-root joints must have rotation-only dynamic transforms")
    static = np.stack([j.xform_from_parent_joint for j in pose.skel.joints])
    if not np.allclose(static[0], np.eye(4), rtol=0, atol=1e-8):
        raise ValueError("Canonical Hips static transform must be identity")
    return data[:, :3, :3].copy(), data[0, :3, 3].copy(), static


def forward_kinematics(rotations, root_positions, static, parents):
    """(..., N, 3, 3), (..., 3), (..., N, 4, 4) -> world rotations/positions.

    Static transforms can be shared or batched. Dynamic non-root translation is
    zero; OFFSET and any static rotations belong to the skeleton transforms.
    """
    world_rotations, world_positions = [], []
    for j, parent in enumerate(parents):
        local_R = static[..., j, :3, :3] @ rotations[..., j, :, :]
        local_p = static[..., j, :3, 3]
        if parent == -1:
            world_R = local_R
            world_p = (static[..., j, :3, :3] @ root_positions[..., None]).squeeze(-1) + local_p
        else:
            world_R = world_rotations[parent] @ local_R
            world_p = (world_rotations[parent] @ local_p[..., None]).squeeze(-1) + world_positions[parent]
        world_rotations.append(world_R)
        world_positions.append(world_p)
    return torch.stack(world_rotations, -3), torch.stack(world_positions, -2)


def vectors_in_root(vectors, root_rotation):
    """Rotate physical vectors (..., K, 3) by the current root R.T."""
    return vectors @ root_rotation


def angular_displacement(current, target):
    """World rotation vectors with target = Exp(error) @ current."""
    relative = target @ np.swapaxes(current, -1, -2)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(*relative.shape[:-2], 3)


def ee_rotations_in_root(world_rotations, root_rotation):
    """Current EE rotations relative to current pelvis; two columns per EE, (..., K, 6)."""
    relative = root_rotation.transpose(-1, -2).unsqueeze(-3) @ world_rotations
    return relative[..., :, :2].transpose(-1, -2).flatten(-2)


def generalized_root_to_world(q_dot, root_rotation):
    return torch.cat(((root_rotation @ q_dot[..., :3, None]).squeeze(-1), q_dot[..., 3:]), -1)
