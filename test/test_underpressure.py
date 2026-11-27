"""No-future contact semantics and preprocessing/live-IK parity with real assets."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "preprocess/training")]
from contact.underpressure import (
    ASSETS, UPSTREAM_FINGERPRINT, VENDORED_FINGERPRINT, UnderPressureContact,
    asset_fingerprint, close_workers, upstream_directory,
)
from contact.sequence import world_transforms, detector_track
from features import FeatureSchema
from inference import LearnedWeights
from pose import SkeletonLayout
from prepare import prepare_clip, compute_statistics, CACHE_VERSION
from training_data import PreparedDataset
from checkpoint import CHECKPOINT_VERSION
from learning import SolveConfig
from weights.learned import ModelConfig, WPrediction
from test_solver import make_pose, apply_reference_delta
from fairmotion.core.motion import Motion


@unittest.skipUnless((ROOT / "third_party/UnderPressure/pretrained.tar").is_file(),
                     "Initialize the pinned UnderPressure submodule")
class FingerprintTests(unittest.TestCase):
    def test_only_the_two_pinned_asset_bundles_are_interchangeable(self):
        upstream = ROOT / "third_party/UnderPressure"
        for crlf, altered in ((False, None), (True, None), (False, "models.py"),
                              (False, "pretrained.tar"), (True, "footskate_samples/0.pt")):
            with self.subTest(crlf=crlf, altered=altered), tempfile.TemporaryDirectory() as tmp:
                for name in ASSETS:
                    data = (upstream / name).read_bytes()
                    if name.endswith(".py"):
                        data = data.replace(b"\r\n", b"\n")
                        if crlf:
                            data = data.replace(b"\n", b"\r\n")
                    path = Path(tmp) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data + b" " if name == altered else data)
                raw = asset_fingerprint(tmp)
                with patch.dict(os.environ, {"SYNERGYIK_UNDERPRESSURE_ROOT": tmp}):
                    if altered is None:
                        self.assertEqual(raw, VENDORED_FINGERPRINT if crlf else UPSTREAM_FINGERPRINT)
                    for saved in (VENDORED_FINGERPRINT, UPSTREAM_FINGERPRINT):
                        detector = UnderPressureContact(fingerprint=saved)
                        if altered is None:
                            self.assertEqual(detector.to_dict()["fingerprint"], UPSTREAM_FINGERPRINT)
                        else:
                            with self.assertRaisesRegex(ValueError, "differ"):
                                detector.to_dict()
                    if altered is not None:
                        self.assertEqual(UnderPressureContact(fingerprint=raw).to_dict()["fingerprint"], raw)


@unittest.skipUnless((upstream_directory() / "pretrained.tar").is_file(), "Install optional UnderPressure assets")
class UnderPressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        pose = make_pose()
        cls.layout = SkeletonLayout.from_skeleton(pose.skel)
        cls.motion = Motion(skel=pose.skel, fps=60)
        for t in range(12):
            cls.motion.add_one_frame(apply_reference_delta(
                pose, np.linspace(.05, .3, 72) * t / 60, cls.layout.rotation_indices).data)
        cls.world = world_transforms(cls.motion)
        cls.static = np.stack([j.xform_from_parent_joint for j in pose.skel.joints])

    @classmethod
    def tearDownClass(cls):
        close_workers()

    def test_future_changes_cannot_change_past_labels(self):
        detector = UnderPressureContact()
        positions = self.world[..., :3, 3]
        prefix = detector.predict(positions[:6], self.layout.names, self.static, 60)
        changed = positions.copy()
        changed[6:, :, 1] += 10
        complete = detector.predict(changed, self.layout.names, self.static, 60)
        np.testing.assert_array_equal(prefix, complete[:6])
        self.assertEqual(complete.shape, (12, 4))
        self.assertEqual(complete.dtype, np.bool_)
        track = detector_track(detector, self.motion, self.world)
        self.assertEqual(track.points.shape, (12, 4, 3))
        expected_points = np.array([
            [pose.get_transform(name, local=False)[:3, 3] for name in detector.joints]
            for pose in self.motion.poses
        ])
        np.testing.assert_allclose(track.points, expected_points, atol=1e-12)
        self.assertIsNone(track.detector)
        self.assertIn("No future input", track.note)
        with self.assertRaisesRegex(ValueError, "Offline"):
            UnderPressureContact(method="underpressure-offline-v1").predict_current(
                positions, self.layout.names, self.static, 60)

    def test_prepared_features_match_realized_history_and_reset(self):
        detector = UnderPressureContact()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.bvh"
            source.write_text("synthetic source", encoding="utf-8")
            with patch("prepare.load_canonicalized_bvh", return_value=self.motion):
                report = prepare_clip((source, directory / "clip.npz", detector.to_dict(), None),
                                          angular_task=False, ee_rotation_features=False)
                with self.assertRaisesRegex(ValueError, "future"):
                    prepare_clip((source, directory / "offline.npz",
                                  UnderPressureContact(method="underpressure-offline-v1").to_dict(), None))
            schema = FeatureSchema(**report["schema"])
            records = [dict(source=source.name, cache="clip.npz", split="train", samples=10)]
            stats = compute_statistics(directory, records, self.layout, schema, 0)
            contract = dict(version=CACHE_VERSION, layout=self.layout.to_dict(), schema=schema.to_dict(),
                            contact=detector.to_dict())
            (directory / "manifest.json").write_text(json.dumps(dict(
                contract=contract, statistics="statistics.json", clips=records)), encoding="utf-8")
            (directory / "statistics.json").write_text(json.dumps(stats), encoding="utf-8")
            dataset = PreparedDataset(directory)
            batch = next(iter(dataset.batches("train", 10)))
            model = WPrediction(ModelConfig(input_width=schema.width, hidden_width=8, depth=2))
            checkpoint = directory / "test.pt"
            torch.save(dict(version=CHECKPOINT_VERSION, model_config=model.config.to_dict(), model_state=model.state_dict(),
                            contract=contract, statistics=stats, solve_config=SolveConfig().to_dict()), checkpoint)
            online = LearnedWeights(checkpoint, self.motion.skel)
            for t in range(11):
                current = self.motion.poses[t]
                online.begin_frame(current, self.motion.poses[max(t-1, 0)], 1 / 60, frame_index=t)
                if t:
                    x_dot = (self.world[t+1, list(self.layout.ee_indices), :3, 3]
                             - self.world[t, list(self.layout.ee_indices), :3, 3]) * 60
                    actual, _ = online.features(current, x_dot)
                    torch.testing.assert_close(actual, batch["features"][t-1], atol=2e-6, rtol=2e-6)
            online.begin_frame(current, current, 1 / 60, frame_index=10)
            self.assertEqual(len(online.position_history), 11)
            online.reset_history()
            self.assertIsNone(online.contact)
            online.begin_frame(current, current, 1 / 60, frame_index=10)
            self.assertEqual(len(online.position_history), 1)


if __name__ == "__main__":
    unittest.main()
