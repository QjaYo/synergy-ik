"""Checkpoint-backed weights with contact fixed for one physical frame."""

import numpy as np
import torch

from contact import contact_from_dict, encode_contact
from contact.underpressure import UnderPressureContact
from features import FeatureSchema, assemble_features, relative_positions
from learning import load_checkpoint
from pose import SkeletonLayout, pose_arrays, forward_kinematics, vectors_in_root, ee_rotations_in_root


class LearnedWeights:
    def __init__(self, checkpoint, skeleton, device="cpu"):
        self.model, self.standardizer, self.saved = load_checkpoint(checkpoint, device)
        self.layout = SkeletonLayout.from_dict(self.saved["contract"]["layout"])
        self.schema = FeatureSchema(**self.saved["contract"]["schema"])
        if SkeletonLayout.from_skeleton(skeleton) != self.layout:
            raise ValueError("Checkpoint and motion have different joint/EE order or topology")
        self.detector = contact_from_dict(self.saved["contract"]["contact"])
        self.device = device
        self.reset_history()
        self.pinv_rtol = self.saved["solve_config"]["pinv_rtol"]

    def geometry(self, pose):
        rotations, root, static = (torch.from_numpy(v).double() for v in pose_arrays(pose))
        world_R, world_p = forward_kinematics(rotations, root, static, self.layout.parents)
        return rotations, root, world_R, world_p

    def reset_history(self):
        self.contact = None
        self.position_history = []
        self.contact_frame = None

    def begin_frame(self, current_pose, previous_pose, dt, *, frame_index=None):
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

    def features(self, pose, x_dot_world):
        if self.contact is None:
            raise RuntimeError("begin_frame must establish contact before predicting weights")
        rotations, root, world_R, positions = self.geometry(pose)
        root_R = rotations[0]
        ee_relative = relative_positions(positions[list(self.layout.ee_indices)], root, root_R, self.detector.ground_y)
        x_dot = vectors_in_root(torch.as_tensor(x_dot_world, dtype=torch.float64).reshape(-1, 3), root_R)
        ee_rotations = (ee_rotations_in_root(world_R[list(self.layout.ee_indices)], root_R)
                        if self.schema.ee_rotations else None)
        values = assemble_features(rotations, root, ee_relative, x_dot, torch.from_numpy(self.contact),
                                   self.layout.rotation_indices, self.detector.ground_y, ee_rotations=ee_rotations)
        return values.to(device=self.device, dtype=torch.float32), root_R.numpy()

    @torch.no_grad()
    def __call__(self, pose, ee_names, x_dot_world):
        if tuple(ee_names) != self.layout.ee_names:
            raise ValueError("This checkpoint requires its four EEs in the trained order")
        features, R = self.features(pose, x_dot_world)
        factor = self.model(self.standardizer(features)).double().cpu().numpy()
        # The existing pose integrator uses world root translation and local angles.
        factor_world = factor.copy()
        factor_world[:3] = R @ factor[:3]
        return factor_world @ factor_world.T
