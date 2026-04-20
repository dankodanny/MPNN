"""
Aggregation step of an MPNN (Gilmer et al. 2017, eq. 2):

    m_v = AGG_{w in N(v)}  m_{vw}

Given per-edge messages and an edge_index, collapse them into one vector per
destination node. Implemented as `scatter_add` / `scatter_mean` / `scatter_max`
so it runs in parallel on CPU or GPU — no Python loop over edges.

Sum is the default; MPNN (Gilmer et al. 2017) uses sum. Mean is degree-
invariant; max picks the strongest neighbor signal.
"""
from typing import Literal

import torch


Reduction = Literal["sum", "mean", "max"]


def aggregate(
    messages: torch.Tensor,    # (E, hidden_dim)
    edge_index: torch.Tensor,  # (2, E)
    num_nodes: int,
    reduce: Reduction = "sum",
) -> torch.Tensor:             # (num_nodes, hidden_dim)
    """Sum/mean/max the messages into their destination nodes."""
    dst = edge_index[1]  # row 1 is destination
    hidden_dim = messages.shape[1]
    out = messages.new_zeros((num_nodes, hidden_dim))

    if reduce == "sum":
        # index_add_(dim, index, source): for each i, out[index[i]] += source[i]
        return out.index_add_(0, dst, messages)

    if reduce == "mean":
        summed = out.index_add_(0, dst, messages)
        # Count how many messages landed in each node (degree of the dst node).
        counts = messages.new_zeros(num_nodes).index_add_(
            0, dst, torch.ones_like(dst, dtype=messages.dtype)
        )
        # Nodes with zero incoming edges (isolated atoms) stay zero; avoid div-by-0.
        counts = counts.clamp_min(1.0).unsqueeze(1)
        return summed / counts

    if reduce == "max":
        # scatter_reduce is the general form; "amax" = elementwise max.
        index = dst.unsqueeze(1).expand_as(messages)
        return out.scatter_reduce(0, index, messages, reduce="amax", include_self=False)

    raise ValueError(f"Unknown reduction: {reduce!r}")
