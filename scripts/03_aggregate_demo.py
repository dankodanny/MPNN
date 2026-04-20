"""
Run: python -m scripts.03_aggregate_demo

Two demos:

1. Tiny hand-built graph (4 nodes, known messages) so you can verify by hand
   that scatter_add really sums rows into the right node slots.
2. Full pipeline on aspirin: featurize -> embed -> EdgeMessage -> aggregate.
"""
import torch
import torch.nn as nn

from mpnn.aggregate import aggregate
from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph
from mpnn.message import EdgeMessage


def toy_demo() -> None:
    """
    Toy graph, 4 nodes, 3 undirected bonds -> 6 directed edges:

          (0) -- (1) -- (2)
                  |
                 (3)

    Directed edges: 0->1, 1->0, 1->2, 2->1, 1->3, 3->1
    Messages chosen so the sum is obvious:
      every message is a constant row matching its src node index + 1.
    """
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 1, 3],   # src
         [1, 0, 2, 1, 3, 1]],  # dst
        dtype=torch.long,
    )
    # Message on edge i = [src_value, src_value, src_value]  (3-dim "hidden")
    src_values = (edge_index[0].float() + 1.0).unsqueeze(1)  # (E, 1)
    messages = src_values.expand(-1, 3).contiguous()          # (E, 3)

    print("--- toy graph ---")
    print("edge_index:")
    print(edge_index)
    print("messages (each row = src+1, broadcast to 3 dims):")
    print(messages)

    agg = aggregate(messages, edge_index, num_nodes=4, reduce="sum")
    print("\naggregated (sum) - expected:")
    print("  node 0: only receives from 1 -> [2, 2, 2]")
    print("  node 1: receives from 0, 2, 3 -> [1+3+4, ...] = [8, 8, 8]")
    print("  node 2: only receives from 1 -> [2, 2, 2]")
    print("  node 3: only receives from 1 -> [2, 2, 2]")
    print(agg)

    agg_mean = aggregate(messages, edge_index, num_nodes=4, reduce="mean")
    print("\naggregated (mean) - node 1 should now be 8/3 ~= 2.667:")
    print(agg_mean)


def aspirin_demo() -> None:
    torch.manual_seed(0)
    HIDDEN = 64

    g = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    embed = nn.Linear(ATOM_FEATURE_DIM, HIDDEN)
    msg = EdgeMessage(hidden_dim=HIDDEN, edge_dim=BOND_FEATURE_DIM)

    h = embed(g.x)
    m = msg(h, g.edge_index, g.edge_attr)
    agg = aggregate(m, g.edge_index, num_nodes=g.num_atoms, reduce="sum")

    print("\n--- aspirin pipeline ---")
    print(f"h        : {tuple(h.shape)}    (per-atom state)")
    print(f"messages : {tuple(m.shape)}    (per-directed-edge)")
    print(f"aggregated: {tuple(agg.shape)}    (back to per-atom)")
    print(f"\nShape check: aggregated is ({g.num_atoms}, {HIDDEN}) - ready to feed "
          f"the update function in Step 5.")


if __name__ == "__main__":
    toy_demo()
    aspirin_demo()
