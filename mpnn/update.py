"""
Update function U of an MPNN (Gilmer et al. 2017, eq. 3):

    h_v^(t+1) = U( h_v^(t) , m_v^(t+1) )

Implemented with nn.GRUCell, matching the original paper. The gates in the GRU
let the network learn how much of the previous state to keep vs overwrite with
the aggregated message.
"""
import torch
import torch.nn as nn


class NodeUpdate(nn.Module):
    """
    GRU-based node update. Treats the N atoms as an N-sized minibatch:
    GRUCell takes (input, hidden) both of shape (batch, hidden_dim) and
    returns a new hidden of the same shape.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim) - previous node states
        aggregated: torch.Tensor,  # (N, hidden_dim) - aggregated messages
    ) -> torch.Tensor:             # (N, hidden_dim) - new node states
        return self.gru(aggregated, h)
