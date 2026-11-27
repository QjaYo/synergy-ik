"""Independent constrained-solve, derivative and checkpoint compatibility checks."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from checkpoint import CHECKPOINT_VERSION, read_checkpoint
from inference import LearnedWeights
from learning import SolveConfig
from train import parse_arguments
from velocity_solvers import METHODS, solve_weighted_velocity


class VelocitySolverTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(137)

    def problem(self):
        J = torch.randn(4, 3, 7, dtype=torch.float64)
        L = torch.tril(torch.randn(4, 7, 7, dtype=torch.float64) * .15) + torch.eye(7)
        x_dot = torch.randn(4, 3, dtype=torch.float64)
        return J, L, x_dot

    def test_both_methods_against_independent_kkt(self):
        J, L, x_dot = self.problem()
        W = L @ L.mT
        zeros = np.zeros((3, 3))
        reference = np.stack([np.linalg.solve(np.block([[w, j.T], [j, zeros]]),
                                             np.r_[np.zeros(7), x])[:7]
                              for w, j, x in zip(W.numpy(), J.numpy(), x_dot.numpy())])
        for method in METHODS:
            q_dot = solve_weighted_velocity(J, L, x_dot, method=method)
            np.testing.assert_allclose(q_dot, reference, atol=2e-12, rtol=2e-12)
            torch.testing.assert_close((J @ q_dot.unsqueeze(-1)).squeeze(-1), x_dot)
            torch.testing.assert_close(solve_weighted_velocity(J, L * 7, x_dot, method=method), q_dot)

    def test_different_ranks_in_one_batch_and_infeasible_targets(self):
        J, L, x_dot = self.problem()
        J[1, -1] = J[1, 0]
        J[2, 1:] = 0
        J[3] = 0
        L.requires_grad_()
        actual = solve_weighted_velocity(J, L, x_dot)
        expected = solve_weighted_velocity(J, L, x_dot, method="cholesky_pinv")
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        residual = J @ actual.unsqueeze(-1) - x_dot.unsqueeze(-1)
        torch.testing.assert_close(J.mT @ residual, torch.zeros(4, 7, 1, dtype=J.dtype), atol=1e-12, rtol=0)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(L.grad).all())
        torch.testing.assert_close(L.grad[3], torch.zeros_like(L.grad[3]))

    def test_no_nullspace_and_broadcasting(self):
        J, L, x_dot = self.problem()
        torch.testing.assert_close(solve_weighted_velocity(J[0], L, x_dot[0]),
                                  torch.stack([solve_weighted_velocity(J[0], l, x_dot[0]) for l in L]))
        J = torch.randn(5, 3, dtype=torch.float64)
        L = torch.eye(3, dtype=J.dtype, requires_grad=True)
        x_dot = torch.randn(5, dtype=J.dtype)
        actual = solve_weighted_velocity(J, L, x_dot)
        torch.testing.assert_close(actual, torch.linalg.lstsq(J, x_dot).solution)
        actual.sum().backward()
        torch.testing.assert_close(L.grad, torch.zeros_like(L))

    def test_rank_threshold_is_applied_to_J(self):
        J = torch.tensor([[1., 0., 0.], [0., 1e-14, 0.]], dtype=torch.float64)
        actual = solve_weighted_velocity(J, torch.eye(3, dtype=J.dtype), torch.ones(2, dtype=J.dtype))
        torch.testing.assert_close(actual, torch.tensor([1., 0., 0.], dtype=J.dtype))

    def test_qr_avoids_overflow_from_inverse_factor(self):
        L = torch.tensor([[1e-160, 0.], [1., 1e-160]], dtype=torch.float64, requires_grad=True)
        J = torch.tensor([[1., 0.]], dtype=L.dtype)
        x_dot = torch.ones(1, dtype=L.dtype)
        scaled = torch.linalg.solve_triangular(L.detach(), J.T, upper=False)
        self.assertFalse(torch.isfinite(scaled).all())
        actual = solve_weighted_velocity(J, L, x_dot)
        torch.testing.assert_close(actual, torch.tensor([1., -1e-160], dtype=L.dtype), atol=0, rtol=1e-12)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(L.grad).all())

    def test_gradients_of_factor_and_target(self):
        J, L, x_dot = self.problem()
        L, x_dot = L[:1].requires_grad_(), x_dot[:1].requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda l, x: solve_weighted_velocity(J[:1], torch.tril(l), x), (L, x_dot)))
        a = solve_weighted_velocity(J[:1], L, x_dot)
        a_grad = torch.autograd.grad(a.square().sum(), (L, x_dot))
        b = solve_weighted_velocity(J[:1], L, x_dot, method="cholesky_pinv")
        b_grad = torch.autograd.grad(b.square().sum(), (L, x_dot))
        # The original implementation reads only the lower triangle of L.
        torch.testing.assert_close(torch.tril(a_grad[0]), b_grad[0], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(a_grad[1], b_grad[1], atol=1e-10, rtol=1e-10)

    def test_changing_J_requires_explicit_derivative_support(self):
        J, L, x_dot = self.problem()
        with self.assertRaisesRegex(ValueError, "fixed J"):
            solve_weighted_velocity(J.requires_grad_(), L, x_dot)
        with self.assertRaisesRegex(ValueError, "Unknown IK solver"):
            solve_weighted_velocity(J.detach(), L, x_dot, method="typo")

    def test_inference_uses_factor_in_the_training_velocity_basis(self):
        J, L, x_dot = self.problem()
        J, L, x_dot = J[0], L[0], x_dot[0]
        root_R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=J.dtype)
        weights = LearnedWeights.__new__(LearnedWeights)
        weights.pinv_rtol = 1e-12
        weights.factor = lambda *_: (L, root_R)
        P = torch.eye(7, dtype=J.dtype)
        P[:3, :3] = root_R
        W = P @ L @ L.T @ P.T
        kkt = np.block([[W.numpy(), J.T.numpy()], [J.numpy(), np.zeros((3, 3))]])
        reference = np.linalg.solve(kkt, np.r_[np.zeros(7), x_dot])[:7]
        for method in METHODS:
            weights.method = method
            actual = weights.solve_velocity(J.numpy(), None, (), x_dot.numpy())
            np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)

    def test_solver_device_is_independent_of_model_device(self):
        args = ["--dataset", "data", "--output", "out", "--device", "cuda"]
        self.assertEqual(SolveConfig().device, "cpu")
        self.assertEqual(parse_arguments(args).solver_device, "cpu")
        self.assertEqual(parse_arguments(args + ["--solver-device", "cuda"]).solver_device, "cuda")

    def test_legacy_checkpoints_keep_the_original_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            for version in (1, 2):
                path = Path(directory) / f"v{version}.pt"
                model = dict(depth=2 if version == 1 else 3)
                saved = dict(version=version, model_config=model, settings=dict(depth=model["depth"]),
                             solve_config=dict(device="cpu", dtype="float64", pinv_rtol=1e-12))
                torch.save(saved, path)
                loaded = read_checkpoint(path)
                self.assertEqual(loaded["version"], CHECKPOINT_VERSION)
                self.assertEqual(SolveConfig(**loaded["solve_config"]).method, "cholesky_pinv")
                self.assertEqual(loaded["settings"]["ik_solver"], "cholesky_pinv")
            args = parse_arguments(["--dataset", directory, "--output", directory, "--epochs", "1"])
            self.assertEqual(args.ik_solver, "nullspace_qr")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_qr_matches_cpu_and_has_finite_gradients(self):
        J, L, x_dot = self.problem()
        expected = solve_weighted_velocity(J, L, x_dot)
        factor = L.cuda().requires_grad_()
        actual = solve_weighted_velocity(J.cuda(), factor, x_dot.cuda())
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-10, rtol=1e-10)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(factor.grad).all())


if __name__ == "__main__":
    unittest.main()
