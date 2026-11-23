"""Compare interrupted training against the same uninterrupted optimizer trajectory."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "preprocess/training"))
from test_solver import make_pose, apply_reference_delta
from contact import FootContact
from features import FeatureSchema
from pose import SkeletonLayout
from prepare import prepare_clip, compute_statistics, CACHE_VERSION
from train import parse_arguments, train
from fairmotion.core.motion import Motion


class ResumeTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        motion = Motion(skel=pose.skel, fps=120)
        for t in range(7):
            motion.add_one_frame(apply_reference_delta(
                pose, np.linspace(.01, .03, 72) * t / 120, layout.rotation_indices).data)
        detector = FootContact(((0, 0, 0),) * 4, "synthetic")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dataset = directory / "data"
            dataset.mkdir()
            source = dataset / "source.bvh"
            source.write_text("synthetic", encoding="utf-8")
            with patch("prepare.load_canonicalized_bvh", return_value=motion):
                report = prepare_clip((source, dataset / "clip.npz", detector.to_dict(), None))
            records = [dict(source="clip.bvh", cache="clip.npz", samples=report["samples"], split=s)
                       for s in ("train", "validation")]
            schema = FeatureSchema(**report["schema"])
            statistics = compute_statistics(dataset, records, layout, schema, 0)
            contract = dict(version=CACHE_VERSION, layout=layout.to_dict(), schema=schema.to_dict(), contact=detector.to_dict())
            (dataset / "manifest.json").write_text(json.dumps(dict(
                contract=contract, statistics="statistics.json", clips=records)), encoding="utf-8")
            (dataset / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
            common = ["--dataset", str(dataset), "--epochs", "2", "--width", "8", "--batch-size", "2",
                      "--lr", "0.00001", "--checkpoint-every", "1", "--log-every", "1"]
            whole, partial = directory / "whole", directory / "partial"
            with redirect_stdout(StringIO()):
                train(parse_arguments(common + ["--output", str(whole)]))
                train(parse_arguments(common + ["--output", str(partial), "--max-steps", "2"]))
                saved = torch.load(partial / "last.pt", weights_only=True)
                self.assertEqual(saved["training_state"]["next_batch"], 2)
                self.assertEqual(saved["training_state"]["next_epoch"], 0)
                train(parse_arguments(["--resume", str(partial / "last.pt")]))
            expected = torch.load(whole / "last.pt", weights_only=True)
            actual = torch.load(partial / "last.pt", weights_only=True)
            self.assertEqual(actual["step"], expected["step"])
            self.assertEqual(actual["epoch"], 2)
            for key, value in expected["model_state"].items():
                torch.testing.assert_close(actual["model_state"][key], value, rtol=0, atol=0)
            for key, values in expected["optimizer_state"]["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(actual["optimizer_state"]["state"][key][name], value, rtol=0, atol=0)
            self.assertEqual(expected["best_validation"], actual["best_validation"])
            initialized = directory / "initialized"
            with redirect_stdout(StringIO()):
                train(parse_arguments(common + ["--output", str(initialized), "--initialize-from",
                    str(whole / "last.pt"), "--lr", "0", "--max-steps", "1"]))
            restarted = torch.load(initialized / "last.pt", weights_only=True)
            self.assertEqual(restarted["step"], 1)
            for key, value in expected["model_state"].items():
                torch.testing.assert_close(restarted["model_state"][key], value, rtol=0, atol=0)
            self.assertTrue(all(v["step"].item() == 1 for v in restarted["optimizer_state"]["state"].values()))
            options = parse_arguments(["--resume", str(initialized / "last.pt")])
            self.assertIsNone(options.initialize_from)
            whole_events = EventAccumulator(str(whole / "tensorboard")).Reload()
            resumed_events = EventAccumulator(str(partial / "tensorboard")).Reload()
            for tag in ("loss/train_batch", "loss/validation", "loss/identity_validation", "numerics/gradient_norm"):
                self.assertEqual([(v.step, v.tensor_proto.double_val[0]) for v in whole_events.Tensors(tag)],
                                 [(v.step, v.tensor_proto.double_val[0]) for v in resumed_events.Tensors(tag)])
            for filename in ("metrics.jsonl", "steps.jsonl"):
                a = [json.loads(v) for v in (whole / filename).read_text().splitlines()]
                b = [json.loads(v) for v in (partial / filename).read_text().splitlines()]
                for rows in (a, b):
                    for row in rows:
                        row.pop("elapsed_seconds")
                self.assertEqual(a, b)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                train(parse_arguments(["--resume", str(partial / "last.pt"), "--batch-size", "3"]))
            failed = directory / "failed"
            with patch("learning.solve_weighted_velocity", side_effect=RuntimeError("test solve failure")):
                with self.assertRaisesRegex(RuntimeError, "test solve failure"):
                    train(parse_arguments(common + ["--output", str(failed)]))
            failure = torch.load(failed / "failure_snapshot.pt", weights_only=True)
            context = json.loads((failed / "failure.json").read_text())
            self.assertEqual(context["step"], 1)
            self.assertEqual(context["source"], "clip.bvh")
            self.assertEqual(context["frames"], failure["batch"]["frames"])
            self.assertEqual(failure["L"].shape, (2, 72, 72))
            self.assertEqual(failure["J"].shape, (2, 12, 72))
            self.assertFalse((failed / "last.pt").exists())
            failed_events = EventAccumulator(str(failed / "tensorboard")).Reload()
            self.assertEqual(failed_events.Tensors("status/failed")[0].tensor_proto.double_val[0], 1.)


if __name__ == "__main__":
    unittest.main()
