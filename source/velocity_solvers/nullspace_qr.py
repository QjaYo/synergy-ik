"""Solve the original W objective in ker(J), without inverses of L or W.

SVD of J supplies a minimum-norm particular solution and an orthonormal null
basis N. QR solves min_z ||L.T N z + L.T q_dot_0||. Rank is determined on J,
independently of W; no truncation or regularization is applied to L.T N.

Autograd is supported through L and x_dot for fixed J (current supervised
training). Differentiating through a changing J requires a separate derivative
of the constrained solve, not autograd through SVD's arbitrary null vectors.
"""

import torch


def solve(J: torch.Tensor, factor: torch.Tensor, x_dot: torch.Tensor, *, rtol: float) -> torch.Tensor:
    if torch.is_grad_enabled() and J.requires_grad:
        raise ValueError("nullspace_qr supports fixed J; gradients through J are not implemented")
    batch_shape = torch.broadcast_shapes(J.shape[:-2], factor.shape[:-2], x_dot.shape[:-1])
    rows, dofs = J.shape[-2:]
    matrices = J.expand(*batch_shape, rows, dofs).reshape(-1, rows, dofs)
    factors = factor.expand(*batch_shape, dofs, dofs).reshape(-1, dofs, dofs)
    targets = x_dot.expand(*batch_shape, rows).reshape(-1, rows)
    with torch.no_grad():
        U, singular, Vh = torch.linalg.svd(matrices, full_matrices=True)
        ranks = (singular > rtol * singular.amax(-1, keepdim=True)).sum(-1)

    # Rank can differ across samples, so each reduced system has its own size.
    result = targets.new_zeros((len(matrices), dofs))
    for rank in torch.unique(ranks).tolist():
        indices = torch.where(ranks == rank)[0]
        vectors = Vh[indices].transpose(-1, -2)
        coordinates = U[indices, :, :rank].transpose(-1, -2) @ targets[indices].unsqueeze(-1)
        q_dot_0 = vectors[..., :rank] @ (coordinates / singular[indices, :rank].unsqueeze(-1))
        L_transpose = factors[indices].transpose(-1, -2)
        N = vectors[..., rank:]
        if rank == 0 or rank == dofs:
            # In these cases the answer is independent of W, with zero W gradient.
            answer = q_dot_0.squeeze(-1) + factors[indices].sum((-2, -1)).unsqueeze(-1) * 0
        else:
            Q, R = torch.linalg.qr(L_transpose @ N, mode="reduced")
            rhs = -Q.transpose(-1, -2) @ (L_transpose @ q_dot_0)
            z = torch.linalg.solve_triangular(R, rhs, upper=True)
            answer = (q_dot_0 + N @ z).squeeze(-1)
        result = result.index_copy(0, indices, answer)
    return result.reshape(*batch_shape, dofs)
