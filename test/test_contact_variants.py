"""Shared caches preserve geometry/targets and select the correct contact contract."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "preprocess/training"))
from test_solver import make_pose, apply_reference_delta
from contact import FootContact, JointGroundContact
from fairmotion.core.motion import Motion
from pose import SkeletonLayout
from prepare import prepare_clip, build_dataset, parse_arguments, forward_kinematics
from training_data import PreparedDataset
from train import train, parse_arguments as training_arguments


class ContactVariantTests(unittest.TestCase):
    def test_one_geometry_pass_matches_two_independent_caches_and_trains(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        motion = Motion(skel=pose.skel, fps=120)
        for t in range(7):
            motion.add_one_frame(apply_reference_delta(
                pose, np.linspace(.01, .03, 72) * t / 120, layout.rotation_indices).data)
        detectors = {"dual_detection": FootContact(((0, 0, 0),) * 4, "synthetic"),
                     "single_detection": JointGroundContact()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bvh"
            source.write_text("synthetic")
            configs = []
            for name, detector in detectors.items():
                path = root / (name + ".json")
                path.write_text(json.dumps(detector.to_dict()))
                configs.append(name + "=" + str(path))
            output = root / "combined"
            args = parse_arguments(["--data-root", str(root), "--review-db", str(root / "unused.sqlite3"),
                                    "--output", str(output), "--task", "position", "--no-ee-rotation-features",
                                    "--contact-variants", *configs])
            record = dict(source="source.bvh", group="source.bvh", mirrored=False, action="synthetic")
            with patch("prepare.select_records", return_value=[record]), \
                 patch("prepare.load_canonicalized_bvh", return_value=motion) as load, \
                 patch("prepare.forward_kinematics", wraps=forward_kinematics) as fk, redirect_stdout(StringIO()):
                build_dataset(args)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(fk.call_count, 1)
            self.assertEqual(len(list(output.rglob("*.npz"))), 1)
            with np.load(output / "clips/source.npz") as archive:
                combined = {name: archive[name] for name in archive.files}
            self.assertNotIn("contact", combined)
            for name, detector in detectors.items():
                independent = root / (name + ".npz")
                with patch("prepare.load_canonicalized_bvh", return_value=motion):
                    prepare_clip((source, independent, detector.to_dict(), None))
                with np.load(independent) as archive:
                    for key in archive.files:
                        if key != "metadata":
                            actual = combined["contact_" + name if key == "contact" else key]
                            np.testing.assert_array_equal(actual, archive[key])
            dual = PreparedDataset(output, "dual_detection")
            single = PreparedDataset(output, "single_detection")
            self.assertEqual(PreparedDataset(output).contract, dual.contract)
            self.assertEqual((dual.schema.width, single.schema.width), (164, 162))
            a, b = next(iter(dual.batches("train", 3))), next(iter(single.batches("train", 3)))
            torch.testing.assert_close(a["features"][:, :-4], b["features"][:, :-2], rtol=0, atol=0)
            torch.testing.assert_close(a["target"], b["target"], rtol=0, atol=0)
            self.assertEqual(dual.statistics["mean"][:-4], single.statistics["mean"][:-2])
            self.assertEqual(dual.statistics["scale"][:-4], single.statistics["scale"][:-2])
            self.assertEqual(dual.statistics["root_rms"], single.statistics["root_rms"])
            self.assertEqual(dual.statistics["angular_rms"], single.statistics["angular_rms"])
            run = root / "run"
            with redirect_stdout(StringIO()):
                train(training_arguments(["--dataset", str(output), "--contact-mode", "single_detection",
                    "--output", str(run), "--epochs", "1", "--max-steps", "1", "--width", "8",
                    "--batch-size", "2", "--lr", "0.00001"]))
            saved = torch.load(run / "last.pt", weights_only=True)
            self.assertEqual(saved["contract"], single.contract)
            self.assertEqual(saved["settings"]["contact_mode"], "single_detection")


if __name__ == "__main__":
    unittest.main()
