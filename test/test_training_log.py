"""Check numerical diagnostics against an analytic diagonal-factor reference."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from learning import VelocityMetrics, scalar_values, velocity_errors
from solver import solve_weighted_velocity
from training_log import TrainingLog


class TrainingLogTests(unittest.TestCase):
    def test_diagnostics_preserve_predictions_gradients_and_rng(self):
        factor = torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64)).unsqueeze(0).requires_grad_()
        J = torch.tensor([[[1., 0., 1.], [0., 1., 1.]]], dtype=torch.float64)
        x_dot = torch.tensor([[.2, .1]], dtype=torch.float64)
        reference = solve_weighted_velocity(J, factor, x_dot)
        expected_gradient, = torch.autograd.grad(reference.square().sum(), factor)
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            with TrainingLog(Path(directory), dict(solve_config=dict(method="cholesky_pinv")), diagnostics_every=10) as log:
                log.step = 1
                log.observe_solve(factor, J)
                prediction = solve_weighted_velocity(J, factor, x_dot)
                gradient, = torch.autograd.grad(prediction.square().sum(), factor)
            torch.testing.assert_close(prediction, reference, rtol=0, atol=0)
            torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            for tag, expected in {"L_sigma_min": 1., "L_condition_max": 4., "L_diagonal_min": 1.,
                                  "J_L_inverse_T_abs_max": 1., "L_finite_fraction": 1.,
                                  "J_L_inverse_T_finite_fraction": 1.}.items():
                values = events.Tensors("numerics/" + tag)
                self.assertEqual([(v.step, v.tensor_proto.double_val[0]) for v in values], [(1, expected)])

    def test_metrics_weight_short_batch_by_samples_and_keep_only_detached_totals(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = VelocityMetrics()
        for size, speed in [(3, 1.), (1, 3.)]:
            q = torch.full((size, 6), speed, device=device, requires_grad=True)
            target = torch.zeros_like(q)
            J = torch.eye(6, device=device).expand(size, -1, -1)
            scales = dict(root_rms=1., angular_rms=1.)
            errors = velocity_errors(q, target, scales)
            batch = metrics.update(q, target, J, target, scales, errors)
            self.assertEqual(metrics.report(size, batch)["loss"], speed ** 2)
        self.assertTrue(all(t.device.type == device and not t.requires_grad for t in metrics.sums.values()))
        self.assertEqual(metrics.result()["loss"], (3 * 1. + 1 * 9.) / 4)
        self.assertEqual(scalar_values(metrics.sums)["loss"], 12.)

    def test_preserves_values_outside_float32_range(self):
        with tempfile.TemporaryDirectory() as directory:
            with TrainingLog(Path(directory), {}) as log:
                log.scalar("large_finite", 1e100)
                log.scalar("small_nonzero", 1e-100)
            events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            self.assertEqual(events.Tensors("large_finite")[0].tensor_proto.double_val[0], 1e100)
            self.assertEqual(events.Tensors("small_nonzero")[0].tensor_proto.double_val[0], 1e-100)


if __name__ == "__main__":
    unittest.main()
