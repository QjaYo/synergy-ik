"""Independent numerical references for EE orientation tasks and cached/online inputs."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "preprocess/training")]
from test_solver import make_pose, apply_reference_delta, finite_difference_jacobian
from fairmotion.core.motion import Motion
from inference import LearnedWeights
from learning import batch_jacobian, SolveConfig, evaluate, load_checkpoint
from playback import IKPlayback
from pose import SkeletonLayout
from prepare import build_dataset, parse_arguments
from solver import compute_pose_task_jacobian, jacobian_in_root, solve_weighted_velocity, solve_ik_step, SolverMode
from train import train, parse_arguments as train_arguments
from training_data import PreparedDataset
from functools import partial


class PoseTaskTests(unittest.TestCase):
    def test_orientation_jacobian_finite_difference_and_weighted_basis(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        J = compute_pose_task_jacobian(pose, layout.ee_names)
        np.testing.assert_allclose(J[:12], finite_difference_jacobian(pose, layout.ee_names,
                                   layout.rotation_indices), atol=1e-8)
        for column in range(72):
            delta = np.eye(72)[column] * 1e-6
            plus = apply_reference_delta(pose, delta, layout.rotation_indices)
            minus = apply_reference_delta(pose, -delta, layout.rotation_indices)
            finite = []
            for name in layout.ee_names:
                a = plus.get_transform(name, local=False)[:3, :3]
                b = minus.get_transform(name, local=False)[:3, :3]
                finite.extend(Rotation.from_matrix(a @ b.T).as_rotvec() / 2e-6)
            np.testing.assert_allclose(J[12:, column], finite, atol=1e-8)
        J = torch.from_numpy(J)
        root_R = torch.from_numpy(pose.data[0, :3, :3])
        P = torch.eye(72, dtype=torch.float64)
        P[:3, :3] = root_R
        S = torch.block_diag(*([root_R.T] * 8))
        torch.testing.assert_close(jacobian_in_root(J, root_R), S @ J @ P)
        rng = np.random.default_rng(4)
        L = torch.from_numpy(np.tril(rng.normal(0, .02, (72, 72))) + np.eye(72))
        W = L @ L.T
        x_dot = torch.from_numpy(rng.normal(0, .1, 24))
        prediction = solve_weighted_velocity(J, L, x_dot)
        kkt = np.block([[W.numpy(), J.numpy().T], [J.numpy(), np.zeros((24, 24))]])
        expected = np.linalg.solve(kkt, np.r_[np.zeros(72), x_dot.numpy()])[:72]
        np.testing.assert_allclose(prediction, expected, atol=1e-10)
        changed = solve_weighted_velocity(S @ J @ P, torch.linalg.cholesky(P.T @ W @ P), S @ x_dot)
        torch.testing.assert_close(P @ changed, prediction, atol=1e-10, rtol=1e-10)

    def test_default_cache_training_inference_and_playback(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        motion = Motion(skel=pose.skel, fps=120)
        for frame in range(7):
            motion.add_one_frame(apply_reference_delta(pose, np.linspace(.01, .05, 72) * frame / 120,
                                                       layout.rotation_indices).data)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.bvh").write_text("synthetic")
            output = root / "cache"
            args = parse_arguments(["--data-root", str(root), "--review-db", str(root / "unused"),
                                    "--output", str(output)])
            record = dict(source="source.bvh", group="source.bvh", mirrored=False, action="test")
            with patch("prepare.select_records", return_value=[record]), \
                 patch("prepare.load_canonicalized_bvh", return_value=motion), redirect_stdout(StringIO()):
                build_dataset(args)
            dataset = PreparedDataset(output)
            self.assertEqual(dataset.schema.width, 201)
            batch = next(iter(dataset.batches("train", 3)))
            self.assertEqual(batch["x_dot"].shape, (3, 24))
            self.assertEqual(batch_jacobian(batch, layout, SolveConfig()).shape, (3, 24, 72))
            run = root / "run"
            with redirect_stdout(StringIO()):
                train(train_arguments(["--dataset", str(output), "--output", str(run), "--epochs", "1",
                    "--max-steps", "1", "--width", "8", "--batch-size", "3"]))
            learned = LearnedWeights(run / "last.pt", motion.skel)
            for index in range(3):
                frame = index + 1
                current, target = motion.poses[frame], motion.poses[frame + 1]
                current_R = current.data[0, :3, :3]
                world_linear, world_angular = [], []
                for name in layout.ee_names:
                    a, b = current.get_transform(name, local=False), target.get_transform(name, local=False)
                    world_linear.extend((b[:3, 3] - a[:3, 3]) * 120)
                    world_angular.extend(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec() * 120)
                velocity = np.r_[world_linear, world_angular]
                learned.begin_frame(current, motion.poses[frame-1], 1/120)
                online, _ = learned.features(current, velocity)
                np.testing.assert_allclose(online, batch["features"][index], atol=2e-7, rtol=1e-6)
                np.testing.assert_allclose(batch["x_dot"][index].reshape(8, 3),
                                           velocity.reshape(8, 3) @ current_R, atol=2e-7)
            model, standardizer, _ = load_checkpoint(run / "last.pt")
            metrics = evaluate(model, standardizer, dataset.batches("train", 3), dataset, SolveConfig())
            self.assertIn("task_angular_rmse_rad_s", metrics["learned"])
            playback = IKPlayback(motion, partial(solve_ik_step, mode=SolverMode.SPD, W=learned),
                panel_label="Weighted", fixed_ee_names=layout.ee_names, constrain_orientation=True,
                prepare_solver=learned.begin_frame, reset_solver=learned.reset_history)
            with redirect_stdout(StringIO()):
                playback.step_once()
            self.assertEqual(playback.frame_index, 1)
            for actual in (playback.identity_pose, playback.ik_pose):
                for name in layout.ee_names:
                    a = actual.get_transform(name, local=False)
                    b = motion.poses[1].get_transform(name, local=False)
                    self.assertLessEqual(np.linalg.norm(a[:3, 3] - b[:3, 3]), .001)
                    self.assertLessEqual(np.linalg.norm(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()), .001)


if __name__ == "__main__":
    unittest.main()
