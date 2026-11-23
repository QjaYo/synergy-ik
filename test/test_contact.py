"""Contact-method parity, units, channel order and cache/checkpoint integration."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "preprocess/training"))
from contact import FootContact, JointGroundContact, contact_from_dict, load_contact
from inference import LearnedWeights
from pose import SkeletonLayout
from prepare import compute_statistics, parse_arguments, prepare_clip
from features import FeatureSchema
from training_data import PreparedDataset
from weights.learned import ModelConfig, WPrediction
from test_solver import make_pose, apply_reference_delta
from fairmotion.core.motion import Motion


class ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ankles_match_reference_at_30_fps_including_first_frame(self):
        # Independent reference: original cm coordinates, all joints and fixed FPS.
        positions_cm = np.array([
            [[0, 4, 0], [0, 100, 0], [0, 6, 0]],
            [[0, 4, 0], [0, 100, 0], [0, 4, 0]],
            [[2, 4, 0], [0, 100, 0], [0, 4, 0]],
            [[2, 5, 0], [0, 100, 0], [0, 4, 0]],
        ], dtype=float)
        reference = np.zeros(positions_cm.shape[:2])
        reference[0] = positions_cm[0, :, 1] < 5
        velocities = np.linalg.norm(np.diff(positions_cm, axis=0), axis=-1) * 30 / 100
        reference[1:] = (positions_cm[1:, :, 1] < 5) & (velocities < .4)
        detector = JointGroundContact(height_threshold=.05)
        points = detector.points(None, positions_cm / 100, ("RightFoot", "Hips", "LeftFoot"))
        previous = np.concatenate((points[:1], points[:-1]))
        actual = detector.detect(points, previous, 1 / 30)
        np.testing.assert_array_equal(actual, reference[:, [2, 0]])
        self.assertEqual(actual.dtype, np.bool_)
        self.assertEqual(actual.shape, (4, 2))

    def test_elapsed_time_ground_and_strict_thresholds(self):
        detector = JointGroundContact()
        current = np.array([[.01, .02, 0], [.01, .02, 0]])
        previous = current.copy()
        previous[:, 0] = 0
        self.assertTrue(detector.detect(current, previous, 1 / 30).all())
        self.assertFalse(detector.detect(current, previous, 1 / 60).any())
        elevated = JointGroundContact(ground_y=1)
        offset = np.array([0, 1, 0])
        np.testing.assert_array_equal(elevated.detect(current + offset, previous + offset, 1 / 30),
                                      detector.detect(current, previous, 1 / 30))
        current = np.array([[0, .12, 0], [.4, 0, 0]])
        previous = current.copy()
        previous[1, 0] = 0
        np.testing.assert_array_equal(detector.detect(current, previous, 1), [False, False])

    def test_config_dispatch_and_existing_cli(self):
        for detector in (FootContact(((0, -.03, 0),) * 4, "test"), JointGroundContact()):
            serialized = json.loads(json.dumps(detector.to_dict()))
            restored = contact_from_dict(serialized)
            self.assertEqual(restored, detector)
        with self.assertRaisesRegex(ValueError, "Unknown contact method"):
            contact_from_dict(dict(method="unknown"))
        common = ["--data-root", "data", "--review-db", "review.sqlite3", "--output", "prepared"]
        old = parse_arguments(common + ["--sole-config", "sole.json"])
        new = parse_arguments(common + ["--contact-config", "sole.json"])
        self.assertEqual(vars(old), vars(new))
        self.assertEqual(load_contact(ROOT / "preprocess/training/ankle_contact.json"), JointGroundContact())

    def test_cached_and_online_features_match_for_both_methods(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        motion = Motion(skel=pose.skel, fps=60)
        velocity = np.linspace(.05, .3, layout.dofs)
        for t in range(6):
            motion.add_one_frame(apply_reference_delta(pose, velocity * t / 60, layout.rotation_indices).data)
        detectors = (FootContact(((0, -.03, 0),) * 4, "test"), JointGroundContact())
        for detector in detectors:
            with self.subTest(method=detector.method), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                source = directory / "input.bvh"
                source.write_text("synthetic test source", encoding="utf-8")
                with patch("prepare.load_canonicalized_bvh", return_value=motion):
                    report = prepare_clip((source, directory / "clip.npz", detector.to_dict(), None))
                schema = FeatureSchema(**report["schema"])
                self.assertEqual(schema.width, 163 + detector.width)
                records = [dict(source=source.name, cache="clip.npz", split="train", samples=4)]
                stats = compute_statistics(directory, records, layout, schema, detector.ground_y)
                contract = dict(version=1, layout=layout.to_dict(), schema=schema.to_dict(),
                                contact=detector.to_dict())
                manifest = dict(contract=contract, statistics="statistics.json", clips=records)
                (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (directory / "statistics.json").write_text(json.dumps(stats), encoding="utf-8")
                dataset = PreparedDataset(directory)
                batch = next(iter(dataset.batches("train", 4)))
                model = WPrediction(ModelConfig(input_width=dataset.schema.width, hidden_width=8, depth=1))
                checkpoint = directory / "test.pt"
                torch.save(dict(version=1, model_config=model.config.to_dict(), model_state=model.state_dict(),
                                contract=contract, statistics=stats, solve_config=dict(pinv_rtol=1e-12)), checkpoint)
                online = LearnedWeights(checkpoint, motion.skel)
                self.assertEqual(online.detector, detector)
                for sample, t in enumerate(range(1, 5)):
                    current, following = motion.poses[t], motion.poses[t + 1]
                    x_dot = np.stack([(following.get_transform(n, local=False)[:3, 3]
                                       - current.get_transform(n, local=False)[:3, 3]) * 60
                                      for n in layout.ee_names])
                    online.begin_frame(current, motion.poses[t - 1], 1 / 60)
                    features, _ = online.features(current, x_dot)
                    torch.testing.assert_close(features, batch["features"][sample], atol=2e-6, rtol=2e-6)
                W = online(current, layout.ee_names, x_dot)
                np.testing.assert_allclose(W, np.eye(layout.dofs), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
