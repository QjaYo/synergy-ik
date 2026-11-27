"""Sample timing, eligibility and train-only normalization contracts."""

from pathlib import Path
from contextlib import closing
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "preprocess/training"))
from test_solver import make_pose, apply_reference_delta
from contact import FootContact
from features import FeatureSchema
from pose import SkeletonLayout
from prepare import prepare_clip, select_records, compute_statistics
from fairmotion.core.motion import Motion


class TrainingDataTests(unittest.TestCase):
    def test_adjacent_samples_and_train_only_statistics(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        motion = Motion(skel=pose.skel, fps=120)
        velocity = np.linspace(.05, .3, 72)
        for t in range(6):
            motion.add_one_frame(apply_reference_delta(pose, velocity * t / 120, layout.rotation_indices).data)
        detector = FootContact(((0, 0, 0),) * 4, "synthetic")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "source.bvh"
            source.write_text("mocked source", encoding="utf-8")
            destination = directory / "clip.npz"
            with patch("prepare.load_canonicalized_bvh", return_value=motion):
                report = prepare_clip((source, destination, detector.to_dict(), None))
            self.assertEqual(report["samples"], 4)
            with np.load(destination) as archive:
                data = {k: archive[k] for k in archive.files}
            for sample in range(4):
                t = sample + 1
                current, following = motion.poses[t], motion.poses[t+1]
                world_delta = np.array([following.get_transform(n, local=False)[:3, 3]
                                        - current.get_transform(n, local=False)[:3, 3] for n in layout.ee_names]) * 120
                R = current.data[0, :3, :3]
                np.testing.assert_allclose(data["x_dot"][sample], world_delta @ R, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(data["q_dot"][sample, :3], velocity[:3] @ R, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(data["q_dot"][sample, 3:], velocity[3:], rtol=1e-6, atol=1e-7)
                previous = motion.poses[t-1]
                previous_delta = np.array([current.get_transform(n, local=False)[:3, 3]
                                            - previous.get_transform(n, local=False)[:3, 3] for n in layout.ee_names]) * 120
                np.testing.assert_allclose(data["x_dot_prev"][sample], previous_delta @ R, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(data["q_dot_prev"][sample, :3], velocity[:3] @ R, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(data["q_dot_prev"][sample, 3:], velocity[3:], rtol=1e-6, atol=1e-7)
                world_positions = np.stack([current.get_transform(n, local=False)[:3, 3] for n in layout.names])
                origin = current.data[0, :3, 3] * [1, 0, 1]
                np.testing.assert_allclose(data["joint_relpos"][sample], (world_positions-origin) @ R, rtol=1e-6, atol=1e-7)
                expected_R = np.stack([current.get_transform(n, local=False)[:3, :3] for n in layout.ee_names])
                np.testing.assert_allclose(data["ee_rotations_world"][t], expected_R, atol=1e-7)
            records = [dict(cache="clip.npz", split="train")]
            schema = FeatureSchema(**report["schema"])
            stats = compute_statistics(directory, records, layout, schema, 0)
            self.assertEqual(stats["count"], 4)
            np.testing.assert_allclose(stats["angular_rms"], np.sqrt(np.mean(velocity[3:] ** 2)), rtol=1e-7)
            np.testing.assert_allclose(stats["mean"][:FeatureSchema().rotation_width], 0)
            np.testing.assert_allclose(stats["scale"][:FeatureSchema().rotation_width], 1)
            np.testing.assert_allclose(stats["mean"][-4:], 0)
            self.assertEqual(set(stats["comparison_inputs"]), {"joint_relpos", "q_dot_prev", "x_dot_prev"})
            data["q_dot"] = data["q_dot"] * 1e9
            np.savez_compressed(directory / "validation.npz", **data)
            held_out = compute_statistics(directory, records + [dict(cache="validation.npz", split="validation")],
                                          layout, schema, 0)
            self.assertEqual(stats, held_out)

    def test_quality_excludes_recording_and_mirror_but_keeps_semantic_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            database = directory / "review.sqlite3"
            with closing(sqlite3.connect(database)) as db:
                db.executescript("CREATE TABLE motions(original_path TEXT, action_name TEXT, human_decision TEXT, present INT);"
                                 "CREATE TABLE files(path TEXT, original_path TEXT, mirrored INT, present INT);")
                for name, decision in (("good", "exclude"), ("bad", "quality_exclude"), ("unsure", "uncertain")):
                    db.execute("INSERT INTO motions VALUES(?,?,?,1)", (name, name, decision))
                    for mirrored in (0, 1):
                        filename = name + ("_M" if mirrored else "") + ".bvh"
                        (directory / filename).touch()
                        db.execute("INSERT INTO files VALUES(?,?,?,1)", (filename, name, mirrored))
                db.commit()
            before = database.read_bytes()
            records = select_records(directory, database, "include")
            self.assertEqual({r["group"] for r in records}, {"good", "unsure"})
            self.assertEqual(len(records), 4)
            self.assertEqual(len(select_records(directory, database, "exclude")), 2)
            self.assertEqual(database.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
