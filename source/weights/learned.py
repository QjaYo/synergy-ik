"""State-conditioned Cholesky factor, identity initialized and trace normalized."""

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_width: int = 167
    dofs: int = 72
    hidden_width: int = 512
    depth: int = 2
    diagonal_epsilon: float = 1e-6
    cost_structure: str = "spd"

    def to_dict(self):
        return asdict(self)


class WPrediction(nn.Module):
    """Return L_norm; W = L_norm @ L_norm.T has mean diagonal one."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.cost_structure != "spd":
            raise NotImplementedError("The first learned baseline uses full SPD costs")
        if config.depth < 0 or config.hidden_width <= 0 or not 0 < config.diagonal_epsilon < 1:
            raise ValueError("Invalid predictor dimensions or diagonal epsilon")
        self.config = config
        layers, width = [], config.input_width
        for _ in range(config.depth):
            layers.extend((nn.Linear(width, config.hidden_width), nn.SiLU()))
            width = config.hidden_width
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(width, config.dofs * (config.dofs + 1) // 2)
        rows, columns = torch.tril_indices(config.dofs, config.dofs)
        self.register_buffer("rows", rows)
        self.register_buffer("columns", columns)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[rows == columns] = math.log(math.expm1(1 - config.diagonal_epsilon))

    def forward(self, features):
        raw = self.head(self.hidden(features))
        lower = raw.new_zeros(*raw.shape[:-1], self.config.dofs, self.config.dofs)
        # Assign positive diagonal values directly; subtraction can cancel tiny values.
        lower[..., self.rows, self.columns] = torch.where(
            self.rows == self.columns, F.softplus(raw) + self.config.diagonal_epsilon, raw,
        )
        scale = (lower.square().sum(dim=(-2, -1)) / self.config.dofs).sqrt()
        return lower / scale[..., None, None]


def predict_diagonal_weights(*args, **kwargs):
    raise NotImplementedError("Conditional diagonal weights are a planned comparison")


def predict_constant_weights(*args, **kwargs):
    raise NotImplementedError("Learned constant costs are a planned comparison")
