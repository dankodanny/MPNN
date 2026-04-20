"""
Readout R of an MPNN (Gilmer et al. 2017, eq. 4): pool per-atom states into
one vector per molecule.

Permutation-invariant by construction (sum/mean/max over a set). Batch-aware:
`batch[i]` says which molecule atom i belongs to, and we scatter accordingly.
Single-graph case is just batch = torch.zeros(N, dtype=long).
"""
from typing import Literal, Optional

import torch


Pool = Literal["sum", "mean", "max", "sum_mean"]


def global_pool(
    h: torch.Tensor,                        # (N, hidden_dim)
    batch: Optional[torch.Tensor] = None,   # (N,) long, values in [0, num_graphs)
    reduce: Pool = "sum",
    num_graphs: Optional[int] = None,
) -> torch.Tensor:                          # (num_graphs, hidden_dim) or 2*hidden_dim
    """Pool atom states into a graph-level vector, preserving permutation invariance."""
    N, H = h.shape
    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=h.device)
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1

    if reduce in ("sum", "mean"):
        out = h.new_zeros((num_graphs, H)).index_add_(0, batch, h)
        if reduce == "mean":
            counts = h.new_zeros(num_graphs).index_add_(
                0, batch, torch.ones_like(batch, dtype=h.dtype)
            )
            out = out / counts.clamp_min(1.0).unsqueeze(1)
        return out

    if reduce == "max":
        index = batch.unsqueeze(1).expand_as(h)
        out = h.new_zeros((num_graphs, H))
        return out.scatter_reduce(0, index, h, reduce="amax", include_self=False)

    if reduce == "sum_mean":
        # Concat sum and mean -> gives the head both absolute and normalized info.
        s = global_pool(h, batch, "sum", num_graphs)
        m = global_pool(h, batch, "mean", num_graphs)
        return torch.cat([s, m], dim=-1)

    raise ValueError(f"Unknown pool: {reduce!r}")


def pool_output_dim(pool: Pool, hidden_dim: int) -> int:
    """Dimension of the pooled vector given pool type and hidden_dim."""
    return 2 * hidden_dim if pool == "sum_mean" else hidden_dim
