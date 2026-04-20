"""
Run: python -m scripts.01_featurize_demo  (from the repo root, env active)

Walks a handful of example SMILES through smiles_to_graph and prints the
shapes so you can eyeball what the tensors look like for real molecules.
"""
from mpnn.featurize import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    smiles_to_graph,
)


EXAMPLES = [
    ("methane", "C"),
    ("ethanol", "CCO"),
    ("benzene", "c1ccccc1"),
    ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("helium (atom only)", "[He]"),
]


def main() -> None:
    print(f"ATOM_FEATURE_DIM = {ATOM_FEATURE_DIM}")
    print(f"BOND_FEATURE_DIM = {BOND_FEATURE_DIM}")
    print()
    for name, smi in EXAMPLES:
        g = smiles_to_graph(smi)
        print(f"{name:20s}  {smi:40s}  {g}")

    # Peek at aspirin's first few rows so the feature layout is concrete.
    print("\n--- aspirin atoms 0..2 feature vectors ---")
    g = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    for i in range(3):
        vec = g.x[i].tolist()
        print(f"atom {i}: {[round(v, 2) for v in vec]}")

    print("\n--- aspirin edge_index first 6 columns (src -> dst) ---")
    ei = g.edge_index[:, :6].tolist()
    for col in range(len(ei[0])):
        print(f"  edge {col}: {ei[0][col]} -> {ei[1][col]}")


if __name__ == "__main__":
    main()
