"""
Run: python -m scripts.04_encoder_demo

Build an MPNNEncoder and run it on aspirin. Shows:
  - output shape (N, hidden_dim) after T message-passing rounds;
  - how node states evolve step by step (L2 distance from the initial embedding);
  - parameter count for tied vs untied weights.
"""
import torch

from mpnn.encoder import MPNNEncoder
from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    torch.manual_seed(0)

    g = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin, 13 atoms

    enc_tied = MPNNEncoder(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=64,
        num_steps=3,
        reduce="sum",
        tied_weights=True,
    )
    enc_untied = MPNNEncoder(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=64,
        num_steps=3,
        reduce="sum",
        tied_weights=False,
    )

    print("--- encoder configs ---")
    print(f"tied weights  : {count_params(enc_tied):>6d} params")
    print(f"untied weights: {count_params(enc_untied):>6d} params  "
          f"(3x the M + 3x the U)")

    h = enc_tied(g.x, g.edge_index, g.edge_attr)
    print(f"\n--- forward pass ---")
    print(f"output h: {tuple(h.shape)}  (expected: ({g.num_atoms}, 64))")

    print("\n--- state evolution on atom 0 (methyl carbon) ---")
    trace = enc_tied.forward_with_trace(g.x, g.edge_index, g.edge_attr)
    initial = trace[0][0]
    for t, h_t in enumerate(trace):
        dist = (h_t[0] - initial).norm().item()
        first3 = [round(v, 3) for v in h_t[0, :3].tolist()]
        label = "embed" if t == 0 else f"after step {t}"
        print(f"{label:>14s}: first 3 dims {first3}  |  L2 from initial: {dist:.3f}")

    print("\nInterpretation: distance grows across steps as atom 0 receives "
          "information from atoms further away in the molecular graph. After 3 "
          "steps, atom 0 has 'seen' everything within graph distance 3.")


if __name__ == "__main__":
    main()
