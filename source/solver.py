"""Position/pose IK, analytic Jacobians, weighted solves and pose integration."""

from dataclasses import dataclass
from enum import IntEnum
from collections.abc import Callable, Sequence

import numpy as np
import torch

from util import (
    get_joint_world_positions, get_rotation_joint_indices,
    rotation_vector_to_matrix,
)
from fairmotion.core.motion import Pose
from fairmotion.ops import conversions
from pose import angular_displacement
from velocity_solvers import DEFAULT_METHOD, solve_weighted_velocity


# The provider returns W in the same reduced velocity order as the Jacobian.
WeightProvider = Callable[[Pose, tuple[str, ...], np.ndarray], np.ndarray]
VelocitySolver = Callable[[np.ndarray, Pose, tuple[str, ...], np.ndarray], np.ndarray]


class SolverMode(IntEnum):
    DIAGONAL = 1  # diagonal W
    SPD = 2       # full SPD W


@dataclass
class IKStepResult:
    """Updated pose and metrics; ee_errors follows the input ee_names order, in meters."""

    pose: Pose
    iterations: int
    max_iterations: int
    converged: bool
    ee_errors: np.ndarray
    orientation_errors: np.ndarray | None = None


def solve_ik_step(
    prev_pose: Pose,
    ee_names: tuple[str, ...],
    target_positions: np.ndarray,
    dt: float,
    *,
    mode: SolverMode = SolverMode.DIAGONAL,
    W: np.ndarray | WeightProvider | None = None,
    velocity_solver: VelocitySolver | None = None,
    method: str = DEFAULT_METHOD,
    pinv_rtol: float = 1e-12,
    target_rotations: np.ndarray | None = None,
) -> IKStepResult:
    """Refine the supplied pose against fixed EE targets and return the final metrics."""
    mode = SolverMode(mode)
    if (W is None) == (velocity_solver is None):
        raise ValueError("Provide either W or velocity_solver")

    pos_tolerance = 1e-3  # m; tolerance for every EE position error
    max_iter = 10
    iters = 0

    def errors() -> tuple[np.ndarray, np.ndarray | None]:
        position = target_positions - get_joint_world_positions(prev_pose, ee_names)
        rotation = None
        if target_rotations is not None and ee_names:
            current = np.stack([prev_pose.get_transform(name, local=False)[:3, :3] for name in ee_names])
            rotation = angular_displacement(current, target_rotations)
        return position, rotation

    for _ in range(max_iter):
        error, rotation_error = errors()
        ee_errors = np.linalg.norm(error, axis=1)
        if np.all(ee_errors <= pos_tolerance) and (rotation_error is None or
                np.all(np.linalg.norm(rotation_error, axis=-1) <= 1e-3)):
            break

        # Desired EE velocity from the position error.
        x_dot = (error / dt).reshape(-1)
        if rotation_error is not None:
            x_dot = np.concatenate((x_dot, (rotation_error / dt).reshape(-1)))

        # Assemble the task Jacobian, then solve for joint velocity.
        J = (compute_jacobian_analytic(prev_pose, ee_names) if target_rotations is None
             else compute_pose_task_jacobian(prev_pose, ee_names))
        if velocity_solver is not None:
            q_dot = velocity_solver(J, prev_pose, ee_names, x_dot)
        else:
            iteration_W = W(prev_pose, ee_names, x_dot) if callable(W) else W
            factor = (np.diag(np.sqrt(np.diag(iteration_W))) if mode == SolverMode.DIAGONAL
                      else np.linalg.cholesky(iteration_W))
            q_dot = solve_weighted_velocity(torch.from_numpy(J), torch.from_numpy(factor),
                                            torch.from_numpy(x_dot), method=method, rtol=pinv_rtol).numpy()

        apply_joint_velocity(prev_pose, q_dot, dt)
        iters += 1

    # Evaluate convergence after the final pose update.
    error, rotation_error = errors()
    ee_errors = np.linalg.norm(error, axis=1)
    orientation_errors = None if rotation_error is None else np.linalg.norm(rotation_error, axis=-1)
    converged = bool(np.all(ee_errors <= pos_tolerance) and
                     (orientation_errors is None or np.all(orientation_errors <= 1e-3)))
    return IKStepResult(
        pose=prev_pose,
        iterations=iters,
        max_iterations=max_iter,
        converged=converged,
        ee_errors=ee_errors,
        orientation_errors=orientation_errors,
    )



def compute_pseudoinverse(
    J: np.ndarray,
    mode: SolverMode,
    W: np.ndarray,
    *, rtol: float = 1e-12,
) -> np.ndarray:
    """J: (task_rows, D), W: (D, D); return (D, task_rows), with 3K or 6K task rows."""
    if W.shape != (J.shape[1], J.shape[1]):
        raise ValueError(f"W must have shape {(J.shape[1], J.shape[1])}, got {W.shape}")
    mode = SolverMode(mode)
    if mode == SolverMode.DIAGONAL:
        return compute_pseudoinverse_diagonal(J, W, rtol=rtol)
    elif mode == SolverMode.SPD:
        return compute_pseudoinverse_spd(J, W, rtol=rtol)
    raise ValueError(f"Unsupported mode: {mode}")


def compute_pseudoinverse_diagonal(J: np.ndarray, W: np.ndarray, *, rtol: float = 1e-12) -> np.ndarray:
    weights = np.diag(W)                    # 1D array of weights
    W_sqrt_inv = np.diag(1.0 / np.sqrt(weights))     # 2D diagonal matrix

    J_scaled = J @ W_sqrt_inv
    J_scaled_pinv = np.linalg.pinv(J_scaled, rcond=rtol)

    return W_sqrt_inv @ J_scaled_pinv # q_dot = W_sqrt_inv @ (J @ W_sqrt_inv)^+ @ x_dot


def compute_pseudoinverse_spd(
    J: np.ndarray, W: np.ndarray, *, rtol: float = 1e-12,
) -> np.ndarray:
    L = np.linalg.cholesky(W)
    J_scaled = np.linalg.solve(L, J.T).T
    return np.linalg.solve(L.T, np.linalg.pinv(J_scaled, rcond=rtol))


def compute_jacobian_tensor(
    world_rotations: torch.Tensor,
    world_positions: torch.Tensor,
    parents: Sequence[int],
    rotation_indices: Sequence[int],
    ee_indices: Sequence[int],
) -> torch.Tensor:
    """Batched analytic world J; local angular columns agree with right multiplication."""
    count = len(ee_indices)
    shape = (*world_positions.shape[:-2], 3 * count, 3 + 3 * len(rotation_indices))
    J = world_positions.new_zeros(shape)
    identity = torch.eye(3, device=J.device, dtype=J.dtype)
    columns = {joint: 3 + 3 * i for i, joint in enumerate(rotation_indices)}
    for k, end in enumerate(ee_indices):
        J[..., 3*k:3*k+3, :3] = identity
        joint = parents[end]
        while joint != -1:
            axes = world_rotations[..., joint, :, :].transpose(-1, -2)
            arm = world_positions[..., end, :] - world_positions[..., joint, :]
            block = torch.linalg.cross(axes, arm.unsqueeze(-2).expand_as(axes), dim=-1).transpose(-1, -2)
            column = columns[joint]
            J[..., 3*k:3*k+3, column:column+3] = block
            joint = parents[joint]
    return J


def jacobian_in_root(J_world: torch.Tensor, root_rotation: torch.Tensor) -> torch.Tensor:
    """Change physical velocity bases; do not differentiate moving-frame features."""
    rows = J_world.reshape(*J_world.shape[:-2], -1, 3, J_world.shape[-1])
    rotated = root_rotation.transpose(-1, -2).unsqueeze(-3) @ rows
    rotated = torch.cat((rotated[..., :3] @ root_rotation.unsqueeze(-3), rotated[..., 3:]), -1)
    return rotated.reshape_as(J_world)


def compute_pose_jacobian_tensor(
    world_rotations: torch.Tensor,
    world_positions: torch.Tensor,
    parents: Sequence[int],
    rotation_indices: Sequence[int],
    ee_indices: Sequence[int],
) -> torch.Tensor:
    """World task rows: all EE linear xyz velocities, then all EE angular xyz velocities."""
    linear = compute_jacobian_tensor(world_rotations, world_positions, parents, rotation_indices, ee_indices)
    angular = torch.zeros_like(linear)
    columns = {joint: 3 + 3 * i for i, joint in enumerate(rotation_indices)}
    for k, end in enumerate(ee_indices):
        joint = end
        while joint != -1:
            if joint in columns:
                column = columns[joint]
                angular[..., 3*k:3*k+3, column:column+3] = world_rotations[..., joint, :, :]
            joint = parents[joint]
    return torch.cat((linear, angular), dim=-2)


def compute_pose_task_jacobian(prev_pose: Pose, ee_names: tuple[str, ...]) -> np.ndarray:
    transforms = np.stack([prev_pose.get_transform(joint, local=False) for joint in prev_pose.skel.joints])
    parents = tuple(-1 if joint.parent_joint is None else prev_pose.skel.get_index_joint(joint.parent_joint)
                    for joint in prev_pose.skel.joints)
    return compute_pose_jacobian_tensor(torch.from_numpy(transforms[:, :3, :3]),
        torch.from_numpy(transforms[:, :3, 3]), parents, get_rotation_joint_indices(prev_pose.skel),
        tuple(prev_pose.skel.get_index_joint(name) for name in ee_names)).numpy()


def compute_jacobian_analytic(prev_pose: Pose, ee_names: tuple[str, ...]) -> np.ndarray:
    """
    Rotation order: get_rotation_joint_indices(prev_pose.skel)
    Columns: Hips world translation, then each retained joint's local xyz rotation.
    Shape: (3*K, 3 + 3*M); M is the number of joints with children.
    Parent relation: joint.parent_joint
    Static transform including OFFSET: joint.xform_from_parent_joint
    World transform: prev_pose.get_transform(joint.name, local=False)
    Selected EE joint names: ee_names
    """
    K = len(ee_names)
    rotation_indices = get_rotation_joint_indices(prev_pose.skel)
    rotation_columns = {
        joint_index: slice(3 + 3*i, 6 + 3*i)
        for i, joint_index in enumerate(rotation_indices)
    }

    J = np.zeros((3*K, 3 + 3*len(rotation_indices)))
    for k, ee_name in enumerate(ee_names):
        rows = slice(3*k, 3*(k+1))
        J[rows, :3] = np.eye(3)
        ee_joint = prev_pose.skel.get_joint(ee_name)
        ee_pos = prev_pose.get_transform(ee_name, local=False)[:3, 3]

        joint = ee_joint.parent_joint
        while joint is not None:
            j = prev_pose.skel.get_index_joint(joint.name)
            T = prev_pose.get_transform(joint.name, local=False)
            p_j = T[:3, 3]      # world position of joint j
            R_j = T[:3, :3]     # world rotation matrix of joint j
            r = ee_pos - p_j    # vector from joint j to end-effector

            block = np.column_stack([
                np.cross(R_j[:, 0], r), # R_j @ e_1
                np.cross(R_j[:, 1], r), # R_j @ e_2
                np.cross(R_j[:, 2], r), # R_j @ e_3
            ])

            J[rows, rotation_columns[j]] = block

            joint = joint.parent_joint

    return J

def compute_jacobian_autograd(prev_pose: Pose, ee_names: tuple[str, ...]) -> np.ndarray:
    """Differentiate EE positions with respect to delta_q. Shape: (3*K, 3 + 3*M).

    delta_q: root translation (3, m), then retained local rotations (3*M, rad).
    Rotation order is get_rotation_joint_indices(prev_pose.skel); leaves stay fixed.
    Requires a canonical pose and at least one EE. Uses CPU float64.
    """
    rotation_indices = get_rotation_joint_indices(prev_pose.skel)
    dtype = torch.float64

    pose_data = torch.tensor(np.asarray(prev_pose.data), dtype=dtype)
    static_transforms = torch.tensor(
        np.stack([
            joint.xform_from_parent_joint
            for joint in prev_pose.skel.joints
        ]),
        dtype=dtype,
    )
    parent_indices = [
        None if joint.parent_joint is None
        else prev_pose.skel.get_index_joint(joint.parent_joint.name)
        for joint in prev_pose.skel.joints
    ]
    ee_indices = [prev_pose.skel.get_index_joint(name) for name in ee_names]

    # argnums=0 differentiates only the first argument, delta_q.
    delta_q = torch.zeros(3 + 3*len(rotation_indices), dtype=dtype)
    J = torch.func.jacrev(ee_positions_from_delta_q, argnums=0)(
        delta_q, pose_data, static_transforms, parent_indices, ee_indices,
        rotation_indices,
    )

    # Return a NumPy array for the current pseudoinverse calculation.
    return J.detach().cpu().numpy()


def ee_positions_from_delta_q(
    delta_q: torch.Tensor,
    pose_data: torch.Tensor,
    static_transforms: torch.Tensor,
    parent_indices: list[int | None],
    ee_indices: list[int],
    rotation_joint_indices: tuple[int, ...],
) -> torch.Tensor:
    """Return EE world positions after applying delta_q, flattened to (3*K,).

    pose_data and static_transforms: (N, 4, 4), including the unchanged leaves.
    delta_q: (3 + 3*M,), with M = len(rotation_joint_indices).
    Parents must precede their children; joint 0 is Hips.
    The canonical Hips static transform must be identity.
    """
    root_translation_delta = delta_q[:3]
    joint_rotation_vectors = delta_q[3:].reshape(-1, 3)
    rotation_rows = {j: i for i, j in enumerate(rotation_joint_indices)}
    world_transforms: list[torch.Tensor] = []

    for j, parent_index in enumerate(parent_indices):
        updated_R = pose_data[j, :3, :3]
        if j in rotation_rows:
            delta_R = rotation_vector_to_matrix(joint_rotation_vectors[rotation_rows[j]])
            updated_R = updated_R @ delta_R

        updated_translation = pose_data[j, :3, 3]
        if j == 0:
            updated_translation = updated_translation + root_translation_delta

        # Combine rotation and translation into a dynamic transform.
        upper = torch.cat([updated_R, updated_translation[:, None]], dim=1)
        dynamic_transform = torch.cat([upper, pose_data[j, 3:4, :]], dim=0)
        local_transform = static_transforms[j] @ dynamic_transform

        if parent_index is None:
            world_transform = local_transform
        else:
            world_transform = world_transforms[parent_index] @ local_transform
        world_transforms.append(world_transform)

    return torch.stack([
        world_transforms[j][:3, 3] for j in ee_indices
    ]).reshape(-1)


def apply_joint_velocity(pose: Pose, q_dot: np.ndarray, dt: float) -> Pose:
    """Apply reduced q_dot in m/s and local rad/s; preserve all leaf transforms.

    q_dot: Hips world translation, then get_rotation_joint_indices(pose.skel).
    pose.data keeps every stored joint, including the endpoints.
    """
    rotation_indices = list(get_rotation_joint_indices(pose.skel))
    expected_shape = (3 + 3*len(rotation_indices),)
    if q_dot.shape != expected_shape:
        raise ValueError(f"q_dot must have shape {expected_shape}, got {q_dot.shape}")
    root_linvel = q_dot[:3]
    joint_angvel = q_dot[3:].reshape(-1, 3)
    pose.data = np.asarray(pose.data)
    pose.data[0, :3, 3] += root_linvel * dt

    delta_R = conversions.A2R(joint_angvel * dt)
    prev_R = pose.data[rotation_indices, :3, :3]
    pose.data[rotation_indices, :3, :3] = prev_R @ delta_R
    return pose
