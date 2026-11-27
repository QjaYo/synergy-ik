"""Original solve: L^-T (J L^-T)^+ x_dot, using triangular solves."""

import torch


def solve(J: torch.Tensor, factor: torch.Tensor, x_dot: torch.Tensor, *, rtol: float) -> torch.Tensor:
    """factor is lower triangular; rtol truncates singular values of J L^-T."""
    scaled = torch.linalg.solve_triangular(factor, J.transpose(-1, -2), upper=False).transpose(-1, -2)
    transformed = torch.linalg.pinv(scaled, rtol=rtol) @ x_dot.unsqueeze(-1)
    return torch.linalg.solve_triangular(factor.transpose(-1, -2), transformed, upper=True).squeeze(-1)
