"""TensorBoard metrics and reproducible failed training batches."""

from contextlib import contextmanager
import json
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import random_state, write_checkpoint


class TrainingLog:
    def __init__(self, output, metadata, *, diagnostics_every=100, purge_step=None):
        self.output, self.metadata = output, metadata
        self.diagnostics_every = diagnostics_every
        self.writer = SummaryWriter(str(output / "tensorboard"), flush_secs=5, purge_step=purge_step)
        self.step = 0
        self.factor = self.jacobian = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.writer.close()

    def scalar(self, tag, value):
        self.writer.add_scalar(tag, float(value), self.step, new_style=True, double_precision=True)

    @contextmanager
    def batch(self, model, optimizer, batch, *, step, epoch, batch_index):
        self.step = step
        self.factor = self.jacobian = None
        try:
            yield
        except Exception as error:
            context = dict(step=step, epoch=epoch, batch=batch_index, source=batch["source"],
                           frames=batch["frames"], error=type(error).__name__, message=str(error),
                           traceback=traceback.format_exc())
            snapshot = dict(version=1, context=context, **self.metadata,
                            model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                            random_state=random_state(),
                            batch={key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                                   for key, value in batch.items()},
                            L=None if self.factor is None else self.factor.cpu(),
                            J=None if self.jacobian is None else self.jacobian.cpu(),
                            gradients={name: parameter.grad.detach().cpu()
                                       for name, parameter in model.named_parameters()
                                       if parameter.grad is not None})
            write_checkpoint(self.output / "failure_snapshot.pt", snapshot)
            (self.output / "failure.json").write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")
            self.scalar("status/failed", 1)
            self.writer.add_text("failure/context", context["traceback"], step)
            self.writer.flush()
            raise
        finally:
            self.factor = self.jacobian = None

    @torch.no_grad()
    def observe_solve(self, factor, jacobian):
        """Observe detached operands before IK; never modify the numerical solve."""
        self.factor, self.jacobian = factor.detach(), jacobian.detach()
        if self.step != 1 and self.step % self.diagnostics_every:
            return
        L, J = self.factor.cpu(), self.jacobian.cpu()
        finite = torch.isfinite(L)
        self.scalar("numerics/L_finite_fraction", finite.double().mean())
        if finite.all():
            singular = torch.linalg.svdvals(L.double())
            self.scalar("numerics/L_sigma_min", singular[..., -1].min())
            self.scalar("numerics/L_condition_max", (singular[..., 0] / singular[..., -1]).max())
            self.scalar("numerics/L_diagonal_min", L.diagonal(dim1=-2, dim2=-1).min())
            scaled = torch.linalg.solve_triangular(L, J.transpose(-1, -2), upper=False)
            self.scalar("numerics/J_L_inverse_T_abs_max", scaled.abs().max())
            self.scalar("numerics/J_L_inverse_T_finite_fraction", torch.isfinite(scaled).double().mean())
        self.writer.flush()

    def metrics(self, label, values):
        for name, value in values.items():
            if name == "loss":
                self.scalar(f"loss/{label}", value)
            elif name != "samples":
                self.scalar(f"error/{name}/{label}", value)

    @torch.no_grad()
    def training_step(self, model, optimizer, values, epoch):
        self.metrics("train_batch", values)
        norms = [torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64)
                 for p in model.parameters() if p.grad is not None]
        self.scalar("numerics/gradient_norm", torch.linalg.vector_norm(torch.stack(norms)))
        self.scalar("progress/epoch", epoch)
        self.scalar("progress/learning_rate", optimizer.param_groups[0]["lr"])
        self.writer.flush()

    def epoch(self, step, training, validation):
        self.step = step
        self.metrics("train_epoch", training)
        self.metrics("validation", validation["learned"])
        self.metrics("identity_validation", validation["identity"])
        self.writer.flush()
