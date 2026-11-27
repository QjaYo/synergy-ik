"""State-conditioned Cholesky factor, identity initialized and trace normalized."""

from dataclasses import dataclass, asdict
import math
from typing import NoReturn

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_width: int = 198  # Root up 3 + other rotations 22×6 + height 1 + EE position 12 + velocity 24 + rotation 24 + contact 2 = 198
    dofs: int = 72 #  root lin_velocity(3) + joint ang_velocity(23x3)
    hidden_width: int = 512
    depth: int = 3  # Total Linear layers, including the output head.
    diagonal_epsilon: float = 1e-6

    @property
    def output_width(self) -> int:
        return self.dofs * (self.dofs + 1) // 2 # 72 × 73 / 2 = 2,628

    def to_dict(self) -> dict:
        return asdict(self)


class WPrediction(nn.Module):
    """Return L_norm; W = L_norm @ L_norm.T has mean diagonal one."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.depth < 1 or config.hidden_width <= 0 or not 0 < config.diagonal_epsilon < 1:
            raise ValueError("Invalid predictor dimensions or diagonal epsilon")

        self.config = config
        layers = []
        width = config.input_width
        for _ in range(config.depth - 1):
            layers.extend((nn.Linear(width, config.hidden_width), nn.SiLU()))
            width = config.hidden_width
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(width, config.output_width)

        rows, columns = torch.tril_indices(config.dofs, config.dofs)
        self.register_buffer("rows", rows)
        self.register_buffer("columns", columns)
        # make the initial model predict L = I, by controlling the bias of the head layer.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[rows == columns] = math.log(math.expm1(1 - config.diagonal_epsilon)) # only diagonal entries

    # Receives standardized features, returns a lower-triangular matrix with positive diagonal and trace normalized.
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.hidden(features)) # input features --> standardized features (features) --> hidden --> output
        lower = raw.new_zeros(*raw.shape[:-1], self.config.dofs, self.config.dofs)
        # Assign positive diagonal values directly; subtraction can cancel tiny values.
        lower[..., self.rows, self.columns] = torch.where(
            self.rows == self.columns, F.softplus(raw) + self.config.diagonal_epsilon, raw,
        )
        scale = (lower.square().sum(dim=(-2, -1)) / self.config.dofs).sqrt()
        return lower / scale[..., None, None]


def predict_diagonal_weights(*args: object, **kwargs: object) -> NoReturn:
    raise NotImplementedError("Conditional diagonal weights are a planned comparison")


def predict_constant_weights(*args: object, **kwargs: object) -> NoReturn:
    raise NotImplementedError("Learned constant costs are a planned comparison")
