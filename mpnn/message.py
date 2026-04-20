"""
Message function M of an MPNN (Gilmer et al. 2017, eq. 1):

    m_{vw} = M( h_v , h_w , e_{vw} )

Given node states `h` of shape (N, hidden_dim), an `edge_index` of shape
(2, E), and edge features of shape (E, edge_dim), produce messages of shape
(E, hidden_dim) — one message per directed edge.

The actual aggregation and update happen elsewhere (Steps 4 and 5).
"""
import torch
import torch.nn as nn


class EdgeMessage(nn.Module):
    """Concat [h_v ; h_w ; e_vw] and push through a 2-layer MLP."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        in_dim = 2 * hidden_dim + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,          # (N, hidden_dim)  — current node states
        edge_index: torch.Tensor, # (2, E)           — src row, dst row
        edge_attr: torch.Tensor,  # (E, edge_dim)    — bond features
    ) -> torch.Tensor:            # (E, hidden_dim)  — one message per edge
        src, dst = edge_index[0], edge_index[1]
        # Gather node states at each edge endpoint. No Python loop — this is
        # the vectorized shortcut: h[src] picks rows from h by index, giving
        # (E, hidden_dim) in one shot.
        h_src = h[src]
        h_dst = h[dst]
        inp = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.mlp(inp)
