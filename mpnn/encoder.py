"""
MPNN encoder: embed atoms, then run T rounds of (message -> aggregate -> update).

Weight sharing across rounds follows the original MPNN (Gilmer et al. 2017):
one M module and one U module, applied T times. Set `tied_weights=False` to
stack T independent (M, U) pairs instead.

Output is per-atom final states of shape (N, hidden_dim). The readout (pooling
into a graph-level vector) lives in Step 6.
"""
from typing import List

import torch
import torch.nn as nn

from mpnn.aggregate import Reduction, aggregate
from mpnn.message import EdgeMessage
from mpnn.update import NodeUpdate


class MPNNEncoder(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 3,
        reduce: Reduction = "sum",
        tied_weights: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.reduce = reduce
        self.tied_weights = tied_weights

        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)

        if tied_weights:
            self.message_fn: nn.Module = EdgeMessage(hidden_dim, bond_feature_dim)
            self.update_fn: nn.Module = NodeUpdate(hidden_dim)
        else:
            self.message_fn = nn.ModuleList(
                [EdgeMessage(hidden_dim, bond_feature_dim) for _ in range(num_steps)]
            )
            self.update_fn = nn.ModuleList(
                [NodeUpdate(hidden_dim) for _ in range(num_steps)]
            )

    def _message(self, t: int) -> nn.Module:
        return self.message_fn if self.tied_weights else self.message_fn[t]

    def _update(self, t: int) -> nn.Module:
        return self.update_fn if self.tied_weights else self.update_fn[t]

    def forward(
        self,
        x: torch.Tensor,            # (N, atom_feature_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr: torch.Tensor,    # (E, bond_feature_dim)
    ) -> torch.Tensor:              # (N, hidden_dim)
        h = self.atom_embed(x)
        for t in range(self.num_steps):
            messages = self._message(t)(h, edge_index, edge_attr)
            aggregated = aggregate(messages, edge_index, num_nodes=h.shape[0], reduce=self.reduce)
            h = self._update(t)(h, aggregated)
        return h

    @torch.no_grad()
    def forward_with_trace(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Same as forward, but returns the node states AFTER each step.

        Useful for eyeballing how representations evolve round by round.
        Length num_steps + 1: index 0 is the initial embedding, index t is
        after t message-passing rounds.
        """
        h = self.atom_embed(x)
        trace = [h.clone()]
        for t in range(self.num_steps):
            messages = self._message(t)(h, edge_index, edge_attr)
            aggregated = aggregate(messages, edge_index, num_nodes=h.shape[0], reduce=self.reduce)
            h = self._update(t)(h, aggregated)
            trace.append(h.clone())
        return trace
