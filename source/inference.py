"""Checkpoint-backed weights with contact fixed for one physical frame."""

from typing import TYPE_CHECKING
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from contact import contact_from_dict, encode_contact
from contact.underpressure import UnderPressureContact
from features import FeatureSchema, assemble_features, relative_positions
from learning import load_checkpoint
from pose import SkeletonLayout, pose_arrays, forward_kinematics, vectors_in_root, ee_rotations_in_root
from solver import jacobian_in_root
from velocity_solvers import METHODS, solve_weighted_velocity

if TYPE_CHECKING:
    from fairmotion.core.motion import Pose, Skeleton


class LearnedWeights:
    def __init__(
        self,
        checkpoint: str | Path,
        skeleton: "Skeleton",
        device: str | torch.device = "cpu",
        *,
        method: str | None = None,
    ) -> None:
        self.model, self.standardizer, self.saved = load_checkpoint(checkpoint, device)
        self.layout = SkeletonLayout.from_dict(self.saved["contract"]["layout"])
        self.schema = FeatureSchema.from_dict(self.saved["contract"]["schema"])
        if SkeletonLayout.from_skeleton(skeleton) != self.layout:
            raise ValueError("Checkpoint and motion have different joint/EE order or topology")
        self.detector = contact_from_dict(self.saved["contract"]["contact"])
        self.device = device
        self.reset_history()
        self.pinv_rtol = self.saved["solve_config"]["pinv_rtol"]
        self.method = method or self.saved["solve_config"]["method"]
        if self.method not in METHODS:
            raise ValueError(f"Unknown IK solver: {self.method}")

    def geometry(self, pose: "Pose") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rotations, root, static = (torch.from_numpy(v).double() for v in pose_arrays(pose))
        world_R, world_p = forward_kinematics(rotations, root, static, self.layout.parents)
        return rotations, root, world_R, world_p

    def reset_history(self) -> None:
        self.contact = None
        self.position_history = []
        self.contact_frame = None

    def begin_frame(
        self,
        current_pose: "Pose",
        previous_pose: "Pose",
        dt: float,
        *,
        frame_index: int | None = None,
    ) -> None:
        if isinstance(self.detector, UnderPressureContact):
            if frame_index is None:
                raise ValueError("UnderPressure requires a physical frame index and reset_history on seek/reset")
            if frame_index == self.contact_frame:
                return
            if self.contact_frame is not None and frame_index != self.contact_frame + 1:
                raise ValueError("Contact history must advance one frame at a time; reset it after seeking")
            _, _, _, positions = self.geometry(current_pose)
            history = [*self.position_history, positions.numpy()]
            static = pose_arrays(current_pose)[2]
            contact = self.detector.predict_current(np.stack(history), self.layout.names, static, 1 / dt)
            self.position_history = history
            self.contact_frame = frame_index
            self.contact = encode_contact(contact)
            return
        sites = []
        for pose in (current_pose, previous_pose):
            _, _, rotations, positions = self.geometry(pose)
            sites.append(self.detector.points(rotations.numpy(), positions.numpy(), self.layout.names))
        self.contact = encode_contact(self.detector.detect(sites[0], sites[1], dt))

    def features(
        self,
        pose: "Pose",
        x_dot_world: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        if self.contact is None:
            raise RuntimeError("begin_frame must establish contact before predicting weights")
        rotations, root, world_R, positions = self.geometry(pose)
        root_R = rotations[0]
        ee_relative = relative_positions(positions[list(self.layout.ee_indices)], root, root_R, self.detector.ground_y)
        x_dot = vectors_in_root(torch.as_tensor(x_dot_world, dtype=torch.float64).reshape(-1, 3), root_R)
        ee_rotations = (ee_rotations_in_root(world_R[list(self.layout.ee_indices)], root_R)
                        if self.schema.ee_rotations else None)
        values = assemble_features(rotations, root, ee_relative, x_dot, torch.from_numpy(self.contact),
                                   self.layout.rotation_indices, self.detector.ground_y, ee_rotations=ee_rotations,
                                   root_rotation=self.schema.root_rotation)
        return values.to(device=self.device, dtype=torch.float32), root_R.numpy()

    @torch.no_grad()
    def factor(
        self,
        pose: "Pose",
        ee_names: Sequence[str],
        x_dot_world: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the learned lower factor in pelvis velocity bases and root R."""
        if tuple(ee_names) != self.layout.ee_names:
            raise ValueError("This checkpoint requires its four EEs in the trained order")
        features, R = self.features(pose, x_dot_world)
        factor = self.model(self.standardizer(features)).double().cpu()
        return factor, torch.from_numpy(R)

    @torch.no_grad()
    def solve_velocity(
        self,
        J_world: np.ndarray,
        pose: "Pose",
        ee_names: Sequence[str],
        x_dot_world: np.ndarray,
    ) -> np.ndarray:
        """Use L directly; return world root velocity and local joint velocities."""
        factor, root_R = self.factor(pose, ee_names, x_dot_world)
        J = jacobian_in_root(torch.as_tensor(J_world).to(factor), root_R)
        x_dot = vectors_in_root(torch.as_tensor(x_dot_world).to(factor).reshape(-1, 3), root_R).flatten()
        q_dot = solve_weighted_velocity(J, factor, x_dot, method=self.method, rtol=self.pinv_rtol)
        return torch.cat((root_R @ q_dot[:3], q_dot[3:])).numpy()

    @torch.no_grad()
    def __call__(
        self,
        pose: "Pose",
        ee_names: Sequence[str],
        x_dot_world: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Expose world-basis W for inspection and legacy callers, not the IK solve."""
        factor, root_R = self.factor(pose, ee_names, x_dot_world)
        factor, R = factor.numpy(), root_R.numpy()
        # The existing pose integrator uses world root translation and local angles.
        factor_world = factor.copy()
        factor_world[:3] = R @ factor[:3]
        return factor_world @ factor_world.T
