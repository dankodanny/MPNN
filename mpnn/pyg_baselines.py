"""
Simpler GNN baselines implemented on top of PyTorch Geometric.

  - GCN (Kipf & Welling 2017): mean-aggregation with degree normalization.
  - GAT (Velickovic 2018):     attention-weighted aggregation with multiple heads.
  - GIN (Xu 2019):             sum-aggregation + MLP; provably max WL power.

Common design for a fair head-to-head with MPNN:
  - same atom embedding (raw 21-dim features -> hidden_dim);
  - T stacked conv layers (UNTIED weights; these models are conventionally
    written as stacked layers, not tied-weights unrolled like MPNN);
  - same readout (global_add + global_mean, concatenated);
  - same 2-layer MLP head.

None of GCN / GAT / GIN consume edge features in their vanilla forms,
which is the point of the comparison: MPNN's use of bond features is
part of why it tends to win on molecular property tasks.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    global_add_pool,
    global_mean_pool,
)


def _make_head(hidden_dim: int, out_dim: int, pool: str) -> nn.Module:
    head_in = 2 * hidden_dim if pool == "sum_mean" else hidden_dim
    return nn.Sequential(
        nn.Linear(head_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _pool(h: torch.Tensor, batch: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "sum":
        return global_add_pool(h, batch)
    if pool == "mean":
        return global_mean_pool(h, batch)
    if pool == "sum_mean":
        return torch.cat([global_add_pool(h, batch), global_mean_pool(h, batch)], dim=-1)
    raise ValueError(f"Unknown pool: {pool!r}")


class GCN(nn.Module):
    """Stacked GCNConv layers. h_v' = ReLU( Sum_u (1/sqrt(deg_v deg_u)) W h_u )."""

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 3,
        out_dim: int = 1,
        pool: str = "sum_mean",
    ):
        super().__init__()
        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_steps)]
        )
        self.head = _make_head(hidden_dim, out_dim, pool)
        self.pool = pool

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index))
        return self.head(_pool(h, batch, self.pool))


class GAT(nn.Module):
    """Stacked multi-head GATConv layers with attention over neighbors."""

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 3,
        heads: int = 4,
        out_dim: int = 1,
        pool: str = "sum_mean",
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        per_head = hidden_dim // heads
        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)
        # concat=True -> out shape is heads * per_head = hidden_dim. Keeps dims tidy across layers.
        self.layers = nn.ModuleList(
            [GATConv(hidden_dim, per_head, heads=heads, concat=True) for _ in range(num_steps)]
        )
        self.head = _make_head(hidden_dim, out_dim, pool)
        self.pool = pool

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index))
        return self.head(_pool(h, batch, self.pool))


class GIN(nn.Module):
    """Stacked GINConv layers. h_v' = MLP( (1+eps) h_v + Sum_u h_u ).

    GINConv takes its internal MLP as a constructor argument - this is why
    parameter counts are higher than GCN/GAT at the same hidden_dim.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 3,
        out_dim: int = 1,
        pool: str = "sum_mean",
    ):
        super().__init__()
        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for _ in range(num_steps)
            ]
        )
        self.head = _make_head(hidden_dim, out_dim, pool)
        self.pool = pool

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index))
        return self.head(_pool(h, batch, self.pool))
