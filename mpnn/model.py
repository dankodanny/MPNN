"""
Full MPNN model: encoder -> readout -> MLP head.

Inputs and outputs:
    x          : (N, atom_feature_dim)
    edge_index : (2, E)
    edge_attr  : (E, bond_feature_dim)
    batch      : (N,) long in [0, num_graphs), or None for single-graph
    -> returns : (num_graphs, out_dim)
"""
from typing import Optional

import torch
import torch.nn as nn

from mpnn.aggregate import Reduction
from mpnn.encoder import MPNNEncoder
from mpnn.readout import Pool, global_pool, pool_output_dim


class MPNN(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 3,
        out_dim: int = 1,
        aggregate_reduce: Reduction = "sum",
        pool: Pool = "sum",
        head_hidden: Optional[int] = None,
        tied_weights: bool = True,
    ):
        super().__init__()
        self.pool = pool

        self.encoder = MPNNEncoder(
            atom_feature_dim=atom_feature_dim,
            bond_feature_dim=bond_feature_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            reduce=aggregate_reduce,
            tied_weights=tied_weights,
        )

        head_in = pool_output_dim(pool, hidden_dim)
        head_hidden = head_hidden or hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        h = self.encoder(x, edge_index, edge_attr)
        g = global_pool(h, batch=batch, reduce=self.pool, num_graphs=num_graphs)
        return self.head(g)
