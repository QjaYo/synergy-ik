"""Independent numerical checks for full-root SPD learning and physical history."""

from contextlib import redirect_stdout
from functools import partial
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "preprocess/training"))
from test_solver import make_pose, finite_difference_jacobian
from contact import FootContact
from features import FeatureSchema, assemble_features, relative_positions
from learning import velocity_loss
from pose import SkeletonLayout, pose_arrays, forward_kinematics
from solver import (compute_jacobian_tensor, jacobian_in_root, solve_weighted_velocity,
                    compute_pseudoinverse_spd, compute_pseudoinverse_diagonal, solve_ik_step)
from playback import IKPlayback
from weights.learned import WPrediction, ModelConfig
from prepare import assign_splits, select_records, compute_statistics
from util import copy_pose
from fairmotion.core.motion import Motion


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def geometry(self):
        pose = make_pose()
        layout = SkeletonLayout.from_skeleton(pose.skel)
        rotation, root, static = map(torch.from_numpy, pose_arrays(pose))
        world_R, world_p = forward_kinematics(rotation, root, static, layout.parents)
        J = compute_jacobian_tensor(world_R, world_p, layout.parents, layout.rotation_indices, layout.ee_indices)
        return pose, layout, rotation, root, world_R, world_p, J

    def test_fk_and_tensor_jacobian_match_fairmotion_and_independent_fd(self):
        pose, layout, _, _, world_R, world_p, J = self.geometry()
        transforms = np.stack([pose.get_transform(n, local=False) for n in layout.names])
        np.testing.assert_allclose(world_p, transforms[:, :3, 3], atol=1e-14)
        np.testing.assert_allclose(world_R, transforms[:, :3, :3], atol=1e-14)
        reference = finite_difference_jacobian(pose, layout.ee_names, layout.rotation_indices)
        np.testing.assert_allclose(J, reference, rtol=1e-7, atol=1e-8)

    def test_spd_against_kkt_and_basis_invariance(self):
        _, layout, rotations, _, _, _, J = self.geometry()
        rng = np.random.default_rng(88)
        raw = rng.normal(size=(layout.dofs, layout.dofs)) * 0.02
        L = torch.from_numpy(np.tril(raw) + np.eye(layout.dofs))
        W = L @ L.T
        x = torch.from_numpy(rng.normal(size=12))
        answer = solve_weighted_velocity(J, L, x)
        # Independent KKT linear system, valid here because J has full row rank.
        kkt = np.block([[W.numpy(), J.numpy().T], [J.numpy(), np.zeros((12, 12))]])
        reference = np.linalg.solve(kkt, np.r_[np.zeros(72), x.numpy()])[:72]
        np.testing.assert_allclose(answer, reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(compute_pseudoinverse_spd(J.numpy(), W.numpy()) @ x.numpy(), reference, atol=1e-10)
        P = torch.eye(72, dtype=torch.float64)
        P[:3, :3] = rotations[0]
        S = torch.block_diag(*([rotations[0].T] * 4))
        J_root = jacobian_in_root(J, rotations[0])
        torch.testing.assert_close(J_root, S @ J @ P)
        torch.testing.assert_close(J_root[:, :3], torch.eye(3, dtype=J.dtype).repeat(4, 1))
        L_root = torch.linalg.cholesky(P.T @ W @ P)
        root_answer = solve_weighted_velocity(J_root, L_root, S @ x)
        torch.testing.assert_close(P @ root_answer, answer, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(root_answer @ (P.T @ W @ P) @ root_answer, answer @ W @ answer)
        torch.testing.assert_close(solve_weighted_velocity(J, L * 7, x), answer)

    def test_rank_deficient_and_diagonal_equivalence(self):
        rng = np.random.default_rng(42)
        J = rng.normal(size=(5, 8))
        J[-1] = J[0]
        w = np.exp(rng.normal(size=8))
        a = compute_pseudoinverse_diagonal(J, np.diag(w))
        b = compute_pseudoinverse_spd(J, np.diag(w))
        np.testing.assert_allclose(a, b, atol=1e-12)
        output = solve_weighted_velocity(torch.tensor(J), torch.diag(torch.tensor(w).sqrt()), torch.ones(5, dtype=torch.float64))
        np.testing.assert_allclose(output, a @ np.ones(5), atol=1e-12)

    def test_solver_gradient_matches_finite_difference(self):
        torch.manual_seed(1)
        J = torch.randn(3, 5, dtype=torch.float64)
        L = torch.tril(torch.randn(5, 5, dtype=torch.float64) * 0.1) + torch.eye(5)
        L.requires_grad_()
        x = torch.randn(3, dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda factor: solve_weighted_velocity(J, torch.tril(factor), x),
                                                 (L,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_predictor_identity_trace_and_end_to_end_gradient(self):
        torch.manual_seed(42)
        model = WPrediction(ModelConfig(input_width=7, dofs=5, hidden_width=16))
        x = torch.randn(4, 7)
        factor = model(x)
        torch.testing.assert_close(factor, torch.eye(5).expand(4, 5, 5))
        with torch.no_grad():
            model.head.weight.normal_(0, 0.05)
        factor = model(x)
        W = factor @ factor.transpose(-1, -2)
        torch.testing.assert_close(W.diagonal(dim1=-2, dim2=-1).sum(-1), torch.full((4,), 5.0))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(W) > 0))
        prediction = solve_weighted_velocity(torch.randn(4, 3, 5).double(), factor.double(), torch.randn(4, 3).double())
        loss = velocity_loss(prediction, torch.randn_like(prediction), dict(root_rms=1, angular_rms=1))
        loss.backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.hidden[0].weight.grad.abs().sum(), 0)

    def test_predictor_preserves_small_positive_diagonals_in_float32(self):
        model = WPrediction(ModelConfig(input_width=1, dofs=5, hidden_width=8, depth=0))
        raw = np.array([-10., -20., -30., -40., -100.])
        with torch.no_grad():
            model.head.bias[model.rows == model.columns] = torch.tensor(raw, dtype=torch.float32)
        factor = model(torch.ones(1, 1))[0]
        # Independent float64 reference avoids subtracting nearly equal large terms.
        positive = np.logaddexp(0, raw) + model.config.diagonal_epsilon
        reference = positive / np.sqrt(np.sum(positive ** 2) / len(raw))
        self.assertTrue(torch.all(factor.diagonal() > 0))
        np.testing.assert_allclose(factor.diagonal().detach(), reference, rtol=1e-6)
        factor.diagonal().log().sum().backward()
        self.assertTrue(torch.isfinite(model.head.bias.grad).all())

    def test_feature_order_and_ground_projected_origin(self):
        _, layout, rotations, root, _, positions, _ = self.geometry()
        rel = relative_positions(positions[list(layout.ee_indices)], root, rotations[0])
        feature = assemble_features(rotations, root, rel, torch.zeros(4, 3), torch.ones(4), layout.rotation_indices)
        self.assertEqual(feature.shape, (FeatureSchema().width,))
        np.testing.assert_array_equal(feature[:6], rotations[0, :, :2].T.flatten())
        np.testing.assert_allclose(rel @ rotations[0].T + np.array([root[0], 0, root[2]]),
                                   positions[list(layout.ee_indices)], atol=1e-14)
        self.assertEqual(feature[138].item(), root[1].item())

    def test_contact_boundaries_and_stationary_reset(self):
        detector = FootContact(((0, 0, 0),) * 4, "test")
        p = np.array([[0, 0, 0], [0, .04, 0], [0, .039, 0], [0, -.01, 0]])
        previous = p.copy()
        previous[2, 0] -= .25
        previous[3, 0] -= .249
        np.testing.assert_array_equal(detector.detect(p, previous, 1), [True, False, False, True])
        np.testing.assert_array_equal(detector.detect(p, p, .01), [True, False, True, True])

    def test_playback_contact_preparation_once_per_frame_and_reset(self):
        pose = make_pose()
        motion = Motion(skel=pose.skel, fps=120)
        for frame in range(3):
            data = pose.data.copy()
            data[0, 0, 3] += frame * .01
            motion.add_one_frame(data)
        observed = []
        def prepare(current, previous, dt, *, frame_index):
            self.assertEqual(frame_index, len(observed))
            observed.append((current.data.copy(), previous.data.copy(), dt))
        playback = IKPlayback(motion, partial(solve_ik_step, W=np.eye(72)), panel_label="test",
                              prepare_solver=prepare, fixed_ee_names=SkeletonLayout.from_skeleton(pose.skel).ee_names)
        with redirect_stdout(StringIO()):
            playback.step_frame()
            realized = playback.ik_pose.data.copy()
            playback.step_frame()
        self.assertEqual(len(observed), 2)
        np.testing.assert_array_equal(observed[0][0], observed[0][1])
        np.testing.assert_array_equal(observed[1][0], realized)
        np.testing.assert_array_equal(observed[1][1], pose.data)
        playback.reset_pose()
        np.testing.assert_array_equal(playback.previous_ik_pose.data, playback.ik_pose.data)

    def test_group_split_preserves_mirror_pairs(self):
        records = [dict(source=f"{group}{suffix}", group=str(group)) for group in range(20) for suffix in ("", "_M")]
        split = assign_splits(records, 42)
        self.assertEqual(split, assign_splits(records, 42))
        self.assertEqual([sum(r["split"] == s for r in split) for s in ("train", "validation", "test")], [32, 4, 4])
        for group in range(20):
            self.assertEqual(len({r["split"] for r in split if r["group"] == str(group)}), 1)


if __name__ == "__main__":
    unittest.main()
