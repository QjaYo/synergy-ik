"""Differentiable batch prediction, velocity metrics and checkpoint reconstruction."""

from dataclasses import dataclass, asdict, field
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

from checkpoint import read_checkpoint
from features import Standardizer
from pose import forward_kinematics, SkeletonLayout
from solver import compute_jacobian_tensor, compute_pose_jacobian_tensor, jacobian_in_root
from velocity_solvers import DEFAULT_METHOD, solve_weighted_velocity
from weights.learned import ModelConfig, WPrediction
from training_data import PreparedDataset


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class SolveConfig:
    device: str = field(default_factory=default_device)
    dtype: str = "float64"
    pinv_rtol: float = 1e-12
    method: str = DEFAULT_METHOD

    def move(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=getattr(torch, self.dtype),
                         non_blocking=self.device == "cuda")

    def to_dict(self) -> dict:
        return asdict(self)


def batch_jacobian(batch: dict, layout: SkeletonLayout, config: SolveConfig) -> torch.Tensor:
    rotations = config.move(batch["rotations"])
    world_R, world_p = forward_kinematics(rotations, config.move(batch["root_positions"]),
                                         config.move(batch["static"]), layout.parents)
    compute = compute_pose_jacobian_tensor if batch.get("angular_task", False) else compute_jacobian_tensor
    world_J = compute(world_R, world_p, layout.parents, layout.rotation_indices, layout.ee_indices)
    return jacobian_in_root(world_J, rotations[:, 0])


def predict_batch(
    model: WPrediction,
    standardizer: Standardizer,
    batch: dict,
    layout: SkeletonLayout,
    solve_config: SolveConfig,
    *,
    observe_solve: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (q_dot, J, x_dot) with shapes (B, D), (B, T, D), (B, T)."""
    device = next(model.parameters()).device
    features = batch["features"].to(device, non_blocking=device.type == "cuda")
    x_dot = solve_config.move(batch["x_dot"])
    J = batch_jacobian(batch, layout, solve_config)
    factor = model(standardizer(features)).to(J)
    if observe_solve is not None:
        observe_solve(factor, J)
    prediction = solve_weighted_velocity(J, factor, x_dot, method=solve_config.method, rtol=solve_config.pinv_rtol)
    return prediction, J, x_dot


def velocity_errors(prediction: torch.Tensor, target: torch.Tensor, statistics: dict) -> dict:
    """The two physical errors and their normalized training objective."""
    squared = (prediction - target).square()
    root_mse = squared[:, :3].mean()
    angular_mse = squared[:, 3:].mean()
    loss = 0.5 * (root_mse / statistics["root_rms"] ** 2 + angular_mse / statistics["angular_rms"] ** 2)
    return dict(loss=loss, root_mse=root_mse, angular_mse=angular_mse)


def velocity_loss(prediction: torch.Tensor, target: torch.Tensor, statistics: dict) -> torch.Tensor:
    return velocity_errors(prediction, target, statistics)["loss"]


def check_finite_gradients(model: torch.nn.Module) -> None:
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    checks = torch.stack([torch.isfinite(g).all() for g in gradients.values()])
    if not checks.all():
        bad = next(name for name, finite in zip(gradients, checks.tolist()) if not finite)
        raise FloatingPointError(f"Non-finite gradient in {bad}")


def scalar_values(values: dict) -> dict:
    """Read a group of detached metric scalars with one device-to-host transfer."""
    if not values or not isinstance(next(iter(values.values())), torch.Tensor):
        return dict(values)
    numbers = torch.stack(tuple(values.values())).detach().tolist()
    return dict(zip(values, numbers))


class VelocityMetrics:
    def __init__(self, angular_task: bool = False) -> None:
        self.angular_task = angular_task
        self.count = 0
        self.sums = dict(loss=0.0, root_mse=0.0, angular_mse=0.0, task_mse=0.0, gt_task_mse=0.0)
        if angular_task:
            self.sums.update(task_angular_mse=0.0, gt_task_angular_mse=0.0)

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        J: torch.Tensor,
        x_dot: torch.Tensor,
        statistics: dict,
        errors: dict | None = None,
    ) -> dict:
        residual = (J @ prediction.unsqueeze(-1)).squeeze(-1) - x_dot
        gt_residual = (J @ target.unsqueeze(-1)).squeeze(-1) - x_dot
        linear_width = x_dot.shape[-1] // 2 if self.angular_task else x_dot.shape[-1]
        errors = errors if errors is not None else velocity_errors(prediction, target, statistics)
        values = {key: value.detach() for key, value in errors.items()}
        values.update(task_mse=residual[:, :linear_width].square().mean(),
                      gt_task_mse=gt_residual[:, :linear_width].square().mean())
        if self.angular_task:
            values.update(task_angular_mse=residual[:, linear_width:].square().mean(),
                          gt_task_angular_mse=gt_residual[:, linear_width:].square().mean())
        size = len(prediction)
        self.count += size
        for key, value in values.items():
            self.sums[key] = self.sums[key] + value.double() * size
        return values

    def result(self) -> dict:
        if not self.count:
            raise ValueError("Cannot evaluate an empty split")
        return self.report(self.count, {key: value / self.count for key, value in self.sums.items()})

    def report(self, count: int, values: dict) -> dict:
        mean = scalar_values(values)
        result = dict(samples=count, loss=mean["loss"], root_rmse_m_s=mean["root_mse"] ** 0.5,
                    angular_rmse_rad_s=mean["angular_mse"] ** 0.5,
                    task_rmse_m_s=mean["task_mse"] ** 0.5, gt_fd_task_rmse_m_s=mean["gt_task_mse"] ** 0.5)
        if self.angular_task:
            result.update(task_angular_rmse_rad_s=mean["task_angular_mse"] ** 0.5,
                          gt_fd_task_angular_rmse_rad_s=mean["gt_task_angular_mse"] ** 0.5)
        return result


@torch.no_grad()
def evaluate(
    model: WPrediction,
    standardizer: Standardizer,
    batches: Iterable[dict],
    dataset: PreparedDataset,
    solve_config: SolveConfig,
) -> dict[str, dict]:
    model.eval()
    learned, identity = VelocityMetrics(dataset.schema.angular_task), VelocityMetrics(dataset.schema.angular_task)
    for batch in batches:
        target = solve_config.move(batch["target"])
        prediction, J, x_dot = predict_batch(model, standardizer, batch, dataset.layout, solve_config)
        identity_prediction = (torch.linalg.pinv(J, rtol=solve_config.pinv_rtol) @ x_dot.unsqueeze(-1)).squeeze(-1)
        learned.update(prediction, target, J, x_dot, dataset.statistics)
        identity.update(identity_prediction, target, J, x_dot, dataset.statistics)
    return dict(learned=learned.result(), identity=identity.result())


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[WPrediction, Standardizer, dict]:
    saved = read_checkpoint(path)
    model = WPrediction(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    statistics = saved["statistics"]
    standardizer = Standardizer(statistics["mean"], statistics["scale"]).to(device)
    return model, standardizer, saved
