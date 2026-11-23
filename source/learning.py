"""Differentiable batch prediction, velocity metrics and checkpoint reconstruction."""

from dataclasses import dataclass, asdict

import torch

from features import Standardizer
from pose import forward_kinematics
from solver import compute_jacobian_tensor, compute_pose_jacobian_tensor, jacobian_in_root, solve_weighted_velocity
from weights.learned import ModelConfig, WPrediction


@dataclass(frozen=True)
class SolveConfig:
    device: str = "cpu"
    dtype: str = "float64"
    pinv_rtol: float = 1e-12

    def to_dict(self):
        return asdict(self)


def batch_jacobian(batch, layout, config):
    options = dict(device=config.device, dtype=getattr(torch, config.dtype))
    rotations = batch["rotations"].to(**options)
    world_R, world_p = forward_kinematics(rotations, batch["root_positions"].to(**options),
                                         batch["static"].to(**options), layout.parents)
    compute = compute_pose_jacobian_tensor if batch.get("angular_task", False) else compute_jacobian_tensor
    world_J = compute(world_R, world_p, layout.parents, layout.rotation_indices, layout.ee_indices)
    return jacobian_in_root(world_J, rotations[:, 0])


def predict_batch(model, standardizer, batch, layout, solve_config, *, observe_solve=None):
    device = next(model.parameters()).device
    features = batch["features"].to(device)
    factor = model(standardizer(features))
    J = batch_jacobian(batch, layout, solve_config)
    factor = factor.to(device=J.device, dtype=J.dtype)
    x_dot = batch["x_dot"].to(J)
    if observe_solve is not None:
        observe_solve(factor, J)
    prediction = solve_weighted_velocity(J, factor, x_dot, rtol=solve_config.pinv_rtol)
    return prediction, J, x_dot


def velocity_loss(prediction, target, statistics):
    root_mse = (prediction[:, :3] - target[:, :3]).square().mean()
    angular_mse = (prediction[:, 3:] - target[:, 3:]).square().mean()
    return 0.5 * (root_mse / statistics["root_rms"] ** 2
                  + angular_mse / statistics["angular_rms"] ** 2)


class VelocityMetrics:
    def __init__(self, angular_task=False):
        self.angular_task = angular_task
        self.count = 0
        self.sums = dict(loss=0.0, root_mse=0.0, angular_mse=0.0, task_mse=0.0, gt_task_mse=0.0)
        if angular_task:
            self.sums.update(task_angular_mse=0.0, gt_task_angular_mse=0.0)

    def update(self, prediction, target, J, x_dot, statistics):
        residual = (J @ prediction.unsqueeze(-1)).squeeze(-1) - x_dot
        gt_residual = (J @ target.unsqueeze(-1)).squeeze(-1) - x_dot
        linear_width = x_dot.shape[-1] // 2 if self.angular_task else x_dot.shape[-1]
        values = dict(loss=velocity_loss(prediction, target, statistics),
                      root_mse=(prediction[:, :3] - target[:, :3]).square().mean(),
                      angular_mse=(prediction[:, 3:] - target[:, 3:]).square().mean(),
                      task_mse=residual[:, :linear_width].square().mean(),
                      gt_task_mse=gt_residual[:, :linear_width].square().mean())
        if self.angular_task:
            values.update(task_angular_mse=residual[:, linear_width:].square().mean(),
                          gt_task_angular_mse=gt_residual[:, linear_width:].square().mean())
        size = len(prediction)
        self.count += size
        for key, value in values.items():
            self.sums[key] += float(value.detach()) * size

    def result(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty split")
        mean = {key: value / self.count for key, value in self.sums.items()}
        result = dict(samples=self.count, loss=mean["loss"], root_rmse_m_s=mean["root_mse"] ** 0.5,
                    angular_rmse_rad_s=mean["angular_mse"] ** 0.5,
                    task_rmse_m_s=mean["task_mse"] ** 0.5, gt_fd_task_rmse_m_s=mean["gt_task_mse"] ** 0.5)
        if self.angular_task:
            result.update(task_angular_rmse_rad_s=mean["task_angular_mse"] ** 0.5,
                          gt_fd_task_angular_rmse_rad_s=mean["gt_task_angular_mse"] ** 0.5)
        return result


@torch.no_grad()
def evaluate(model, standardizer, batches, dataset, solve_config):
    model.eval()
    learned, identity = VelocityMetrics(dataset.schema.angular_task), VelocityMetrics(dataset.schema.angular_task)
    for batch in batches:
        prediction, J, x_dot = predict_batch(model, standardizer, batch, dataset.layout, solve_config)
        target = batch["target"].to(prediction)
        identity_prediction = (torch.linalg.pinv(J, rtol=solve_config.pinv_rtol) @ x_dot.unsqueeze(-1)).squeeze(-1)
        learned.update(prediction, target, J, x_dot, dataset.statistics)
        identity.update(identity_prediction, target, J, x_dot, dataset.statistics)
    return dict(learned=learned.result(), identity=identity.result())


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != 1:
        raise ValueError("Unsupported learned checkpoint version")
    model = WPrediction(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    statistics = saved["statistics"]
    standardizer = Standardizer(statistics["mean"], statistics["scale"]).to(device)
    return model, standardizer, saved
