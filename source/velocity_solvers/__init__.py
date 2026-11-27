"""Interchangeable weighted velocity solvers for W = L @ L.T."""

import torch

from . import cholesky_pinv, nullspace_qr


METHODS = ("nullspace_qr", "cholesky_pinv")
DEFAULT_METHOD = "nullspace_qr"


def solve_weighted_velocity(
    J: torch.Tensor,
    factor: torch.Tensor,
    x_dot: torch.Tensor,
    *,
    method: str = DEFAULT_METHOD,
    rtol: float = 1e-12,
) -> torch.Tensor:
    """Return (..., D) velocities from J (..., T, D), L (..., D, D), x_dot (..., T).

    Minimize the task residual first, then q_dot.T @ W @ q_dot among its
    minimizers. For feasible targets this is the equality-constrained W solve.
    Inputs must share velocity bases, dtype and device; leading axes broadcast.
    """
    if method == "nullspace_qr":
        return nullspace_qr.solve(J, factor, x_dot, rtol=rtol)
    if method == "cholesky_pinv":
        return cholesky_pinv.solve(J, factor, x_dot, rtol=rtol)
    raise ValueError(f"Unknown IK solver: {method}; choose from {METHODS}")
