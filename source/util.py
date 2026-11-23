"""Shared BVH loading, pose helpers, and rotation conversions."""

from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np
import torch

FAIRMOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "third_party" / "fairmotion"
sys.path.insert(0, str(FAIRMOTION_DIRECTORY))

from fairmotion.core.motion import Motion, Pose, Skeleton
from fairmotion.data import bvh


def validate_canonicalized_motion(motion: Motion) -> None:
    """Validate frame data and the canonical skeleton required by the viewers."""
    if motion.num_frames() == 0:
        raise ValueError("The motion has no frames")
    if not np.isfinite(motion.fps) or motion.fps <= 0:
        raise ValueError("The motion must have a positive finite FPS")
    root = motion.skel.root_joint
    if root.name != "Hips" or motion.skel.num_joints() != 28:
        raise ValueError("Expected a canonicalized BVH: Hips root and 28 nodes")
    if root.info.get("dof") != 6:
        raise ValueError("Hips must have six position/rotation channels")
    for side in ("Left", "Right"):
        end = motion.skel.get_joint(f"{side}HandEnd")
        if end.parent_joint.name != f"{side}Hand" or end.child_joints:
            raise ValueError(f"Expected {side}Hand -> {side}HandEnd as a leaf")
    for joint in motion.skel.joints:
        if not np.isfinite(joint.xform_from_parent_joint).all():
            raise ValueError(f"Non-finite static transform: {joint.name}")
    for frame_index, pose in enumerate(motion.poses):
        if not np.isfinite(pose.data).all():
            raise ValueError(f"Non-finite motion values at frame {frame_index}")


def load_canonicalized_bvh(bvh_path: Path, input_unit: str = "m") -> Motion:
    """Load a canonicalized BVH in meters and validate its structure."""
    bvh_path = Path(bvh_path)
    if not bvh_path.is_file():
        raise FileNotFoundError(bvh_path)
    length_scales = {"m": 1.0, "cm": 0.01}
    if input_unit not in length_scales:
        raise ValueError("Input length unit must be 'm' or 'cm'")

    motion = bvh.load(str(bvh_path), scale=length_scales[input_unit])
    validate_canonicalized_motion(motion)
    return motion


def get_joint_world_positions(pose: Pose, names: Sequence[str]) -> np.ndarray:
    """Return world positions in the requested order; an empty selection has shape (0, 3)."""
    return np.array([
        pose.get_transform(name, local=False)[:3, 3] for name in names
    ]).reshape(-1, 3)


def copy_pose(pose: Pose) -> Pose:
    """Copy the pose transforms while sharing the read-only skeleton."""
    return Pose(pose.skel, np.array(pose.data, copy=True))


def get_rotation_joint_indices(skel: Skeleton) -> tuple[int, ...]:
    """Return skeleton indices of optimized rotations, in skeleton order.

    Position IK excludes leaf rotations because they move no joint origins.
    This selection is independent of the currently selected end effectors.
    """
    return tuple(j for j, joint in enumerate(skel.joints) if joint.child_joints)


def get_ik_dof_count(skel: Skeleton) -> int:
    """Root translation (3), then local xyz angular velocity per retained joint."""
    return 3 + 3 * len(get_rotation_joint_indices(skel))


def rotation_vector_to_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert a rotation vector (3,), in radians, to a rotation matrix (3, 3)."""
    x, y, z = rotation_vector.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack([
        zero, -z,    y,
        z,    zero, -x,
        -y,   x,    zero,
    ]).reshape(3, 3)
    return torch.linalg.matrix_exp(skew)
