"""
PyTorch Geometric version of the MPNN.

Compare with:
  mpnn/message.py    -> MPNNLayer.message()     (one method instead of a module)
  mpnn/aggregate.py  -> super().__init__(aggr="add")   (one string)
  mpnn/update.py     -> MPNNLayer.update()      (or just done in forward())
  mpnn/encoder.py    -> a for loop
  mpnn/readout.py    -> global_add_pool / global_mean_pool (one import)
  mpnn/model.py      -> MPNNPyG                 (half the lines of model.py)

Same math, same atom/bond featurization. The library handles:
  - The scatter/gather plumbing inside MessagePassing.propagate().
  - Batching via Batch.from_data_list() (PyG's DataLoader calls this).
  - Global pooling with a batch index.

Key convention difference from our scratch code:
  PyG uses x_i = target (central node receiving messages) and x_j = source
  (neighbor sending). Our scratch code used h_dst / h_src. Same thing, and
  PyG derives the i/j split automatically from edge_index.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import (
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_mean_pool,
)


class MPNNLayer(MessagePassing):
    """One round of message + aggregate + update, PyG-style.

    Compared to our scratch code, this single class replaces EdgeMessage
    + aggregate() + NodeUpdate. The super().__init__(aggr='add') is the
    whole aggregation step. propagate() internally calls message(), then
    scatter-sums, then our update() (here we call the GRU inline).
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")  # <- replaces mpnn/aggregate.py
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.gru(aggregated, x)

    def message(self, x_i, x_j, edge_attr):
        # x_j = source node features, x_i = target node features.
        # Our scratch code concatenated [h_src, h_dst, edge_attr],
        # which in PyG notation is [x_j, x_i, edge_attr].
        return self.msg_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))


class MPNNPyG(nn.Module):
    """Full model: atom embed -> T rounds of MPNNLayer (tied weights) -> pool -> head."""

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 3,
        out_dim: int = 1,
        pool: str = "sum_mean",
        set2set_steps: int = 3,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.pool = pool

        self.atom_embed = nn.Linear(atom_feature_dim, hidden_dim)
        self.layer = MPNNLayer(hidden_dim, bond_feature_dim)  # tied weights across T steps

        # Set2Set (Vinyals 2016) is what the original Gilmer MPNN used as readout.
        # It runs an LSTM for `processing_steps` iterations, attending over all
        # atoms each step. Output dim is 2 * hidden_dim.
        if pool == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps)
            head_in = 2 * hidden_dim
        elif pool == "sum_mean":
            self.set2set = None
            head_in = 2 * hidden_dim
        elif pool in ("sum", "mean"):
            self.set2set = None
            head_in = hidden_dim
        else:
            raise ValueError(f"Unknown pool: {pool!r}")

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_embed(x)
        for _ in range(self.num_steps):
            h = self.layer(h, edge_index, edge_attr)

        if self.pool == "sum":
            g = global_add_pool(h, batch)
        elif self.pool == "mean":
            g = global_mean_pool(h, batch)
        elif self.pool == "sum_mean":
            g = torch.cat([global_add_pool(h, batch), global_mean_pool(h, batch)], dim=-1)
        elif self.pool == "set2set":
            g = self.set2set(h, batch)
        else:
            raise ValueError(f"Unknown pool: {self.pool!r}")

        return self.head(g)
