"""
Run: python -m scripts.02_message_demo

Takes one molecule, embeds atom features into a hidden space, and runs the
EdgeMessage module once so you can eyeball the shapes and a sample message.
"""
import torch
import torch.nn as nn

from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph
from mpnn.message import EdgeMessage


HIDDEN_DIM = 64


def main() -> None:
    torch.manual_seed(0)

    g = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin

    # Project raw atom features (21 dims) into hidden space (64 dims).
    # In the full MPNN this projection sits at the very start; the message
    # function always sees hidden-dim node states.
    atom_embed = nn.Linear(ATOM_FEATURE_DIM, HIDDEN_DIM)
    h = atom_embed(g.x)

    msg_fn = EdgeMessage(hidden_dim=HIDDEN_DIM, edge_dim=BOND_FEATURE_DIM)
    messages = msg_fn(h, g.edge_index, g.edge_attr)

    print(f"atoms           : {g.num_atoms}")
    print(f"edges (directed): {g.edge_index.shape[1]}")
    print(f"h (node states) : {tuple(h.shape)}       "
          f"[expected: ({g.num_atoms}, {HIDDEN_DIM})]")
    print(f"messages        : {tuple(messages.shape)}       "
          f"[expected: ({g.edge_index.shape[1]}, {HIDDEN_DIM})]")

    # Sanity: the message on edge (i -> j) should generally differ from
    # (j -> i) because src and dst are swapped in the MLP input.
    print("\n--- first two edges (same bond, opposite directions) ---")
    src0, dst0 = g.edge_index[:, 0].tolist()
    src1, dst1 = g.edge_index[:, 1].tolist()
    print(f"edge 0 ({src0} -> {dst0})  first 5 dims: "
          f"{[round(v, 3) for v in messages[0, :5].tolist()]}")
    print(f"edge 1 ({src1} -> {dst1})  first 5 dims: "
          f"{[round(v, 3) for v in messages[1, :5].tolist()]}")
    diff = (messages[0] - messages[1]).abs().mean().item()
    print(f"mean |diff| between the two directions: {diff:.4f}")


if __name__ == "__main__":
    main()
