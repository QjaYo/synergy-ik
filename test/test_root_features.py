"""Yaw invariance, flip continuity, and cached/online root input contracts."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "preprocess/training"))
from test_solver import make_pose, apply_reference_delta
from checkpoint import CHECKPOINT_VERSION
from contact.single_detection import JointGroundContact
from features import FeatureSchema, assemble_features, relative_positions
from inference import LearnedWeights
from learning import SolveConfig
from pose import SkeletonLayout, pose_arrays, forward_kinematics, ee_rotations_in_root, vectors_in_root
from prepare import prepare_clip, compute_statistics
from training_data import PreparedDataset
from weights.learned import ModelConfig, WPrediction
from fairmotion.core.motion import Motion


class RootFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_complete_input_is_invariant_to_world_yaw(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        rotations, root, static = map(torch.from_numpy, pose_arrays(pose))
        rotations[0] = torch.from_numpy(Rotation.from_euler("xyz", [.9, .4, -.7]).as_matrix())
        original = rotations.clone()
        velocity = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 20
        contacts = torch.tensor([True, False])

        def features(local_rotations, root_position, world_velocity):
            world_R, world_p = forward_kinematics(local_rotations, root_position, static, layout.parents)
            ee = list(layout.ee_indices)
            return assemble_features(local_rotations, root_position,
                relative_positions(world_p[ee], root_position, local_rotations[0]),
                vectors_in_root(world_velocity, local_rotations[0]), contacts, layout.rotation_indices,
                ee_rotations=ee_rotations_in_root(world_R[ee], local_rotations[0]))

        expected = features(rotations, root, velocity)
        self.assertEqual(expected.numel(), ModelConfig().input_width)
        self.assertEqual(expected.numel(), 198)
        np.testing.assert_allclose(expected[:3],
            Rotation.from_matrix(rotations[0].numpy()).inv().apply([0, 1, 0]), atol=1e-14)
        for yaw in (0.3, np.pi / 2, np.pi, -2.2):
            Q = torch.from_numpy(Rotation.from_euler("y", yaw).as_matrix())
            rotated = rotations.clone()
            rotated[0] = Q @ rotations[0]
            actual = features(rotated, Q @ root, velocity @ Q.T)
            torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(rotations, original, atol=0, rtol=0)

    def test_full_flip_is_continuous_including_vertical_and_inverted_poses(self):
        angles = np.linspace(0, 2 * np.pi, 1441)
        rotations = torch.eye(3, dtype=torch.float64).repeat(len(angles), 23, 1, 1)
        rotations[:, 0] = torch.from_numpy(Rotation.from_euler("x", angles).as_matrix())
        features = assemble_features(rotations, torch.zeros(len(angles), 3),
            torch.zeros(len(angles), 4, 3), torch.zeros(len(angles), 8, 3),
            torch.zeros(len(angles), 2), tuple(range(23)), ee_rotations=torch.zeros(len(angles), 4, 6))
        reference = np.column_stack((np.zeros_like(angles), np.cos(angles), -np.sin(angles)))
        np.testing.assert_allclose(features[:, :3], reference, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(features[:, :3], axis=1), 1, atol=1e-15)
        steps = torch.linalg.vector_norm(features[1:, :3] - features[:-1, :3], dim=-1)
        self.assertLess(float(steps.max()), 0.005)
        torch.testing.assert_close(features[0], features[-1], atol=1e-14, rtol=0)

    def test_legacy_npz_reuse_and_online_features_for_both_encodings(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        motion = Motion(skel=pose.skel, fps=120)
        for t in range(6):
            motion.add_one_frame(apply_reference_delta(
                pose, np.linspace(.05, .3, 72) * t / 120, layout.rotation_indices).data)
        detector = JointGroundContact()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "clip.bvh"
            source.write_text("synthetic", encoding="utf-8")
            clip_path = directory / "clip.npz"
            with patch("prepare.load_canonicalized_bvh", return_value=motion):
                report = prepare_clip((source, clip_path, detector.to_dict(), None),
                                      angular_task=True, ee_rotation_features=True)
            legacy_schema = FeatureSchema(contact_width=2, angular_task=True, ee_rotations=True,
                                          root_rotation="rotation_6d")
            schema_dict = legacy_schema.to_dict()
            del schema_dict["root_rotation"]  # Actual historical manifests lacked this field.
            self.assertEqual(FeatureSchema.from_dict(schema_dict), legacy_schema)
            records = [dict(source=source.name, cache=clip_path.name, split="train", samples=report["samples"])]
            stats = compute_statistics(directory, records, layout, legacy_schema, 0)
            contract = dict(version=4, layout=layout.to_dict(), schema=schema_dict, contact=detector.to_dict())
            (directory / "manifest.json").write_text(json.dumps(dict(
                contract=contract, statistics="statistics.json", clips=records)), encoding="utf-8")
            (directory / "statistics.json").write_text(json.dumps(stats), encoding="utf-8")
            paths = (clip_path, directory / "manifest.json", directory / "statistics.json")
            original_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            old = PreparedDataset(directory)
            new = PreparedDataset(directory, root_rotation="up_vector")
            self.assertEqual((old.schema.width, new.schema.width), (201, 198))
            self.assertEqual(new.statistics["mean"], [0.] * 3 + old.statistics["mean"][6:])
            self.assertEqual(new.statistics["scale"], [1.] * 3 + old.statistics["scale"][6:])
            old_batch = next(iter(old.batches("train", 4)))
            new_batch = next(iter(new.batches("train", 4)))
            torch.testing.assert_close(new_batch["features"][:, 3:], old_batch["features"][:, 6:], atol=0, rtol=0)
            for name in ("rotations", "root_positions", "static", "x_dot", "target"):
                torch.testing.assert_close(new_batch[name], old_batch[name], atol=0, rtol=0)
            for dataset, batch in ((old, old_batch), (new, new_batch)):
                model = WPrediction(ModelConfig(input_width=dataset.schema.width, hidden_width=8))
                checkpoint = directory / "weights.pt"
                torch.save(dict(version=CHECKPOINT_VERSION, model_config=model.config.to_dict(),
                    model_state=model.state_dict(), contract=dataset.contract, statistics=dataset.statistics,
                    solve_config=SolveConfig().to_dict()), checkpoint)
                online = LearnedWeights(checkpoint, motion.skel)
                for sample, t in enumerate(batch["frames"]):
                    current = motion.poses[t]
                    world_velocity = batch["x_dot"][sample].numpy().reshape(-1, 3) @ current.data[0, :3, :3].T
                    online.begin_frame(current, motion.poses[t-1], 1 / 120)
                    actual, _ = online.features(current, world_velocity)
                    torch.testing.assert_close(actual, batch["features"][sample], atol=2e-6, rtol=2e-6)
            self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths], original_hashes)


if __name__ == "__main__":
    unittest.main()
