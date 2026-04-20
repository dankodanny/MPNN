"""
Run: python -m scripts.05_model_demo

Full MPNN model in action. Shows:
  - one prediction for one molecule;
  - permutation invariance: randomly reorder atoms, same output;
  - a hand-built batch of three molecules via the 'disconnected supergraph'
    trick (merge graphs, track which atom belongs to which).
"""
import torch

from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph
from mpnn.model import MPNN


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def single_molecule() -> None:
    torch.manual_seed(0)

    model = MPNN(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=64,
        num_steps=3,
        out_dim=1,
        pool="sum_mean",
    ).eval()
    print(f"--- model ---")
    print(f"params: {count_params(model):,}")

    g = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    with torch.no_grad():
        y = model(g.x, g.edge_index, g.edge_attr)
    print(f"\n--- single molecule (aspirin) ---")
    print(f"output shape: {tuple(y.shape)}  (expected: (1, 1))")
    print(f"prediction  : {y.item():.4f}  (random init, so this is meaningless)")


def permutation_invariance() -> None:
    """
    Re-label atoms 0..N-1 -> random permutation and check prediction is identical.
    This is THE sanity check that readout is truly permutation-invariant.
    """
    torch.manual_seed(0)
    model = MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, pool="sum").eval()

    g = smiles_to_graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
    with torch.no_grad():
        y_orig = model(g.x, g.edge_index, g.edge_attr).item()

    torch.manual_seed(42)
    perm = torch.randperm(g.num_atoms)
    inv = torch.argsort(perm)           # inverse permutation

    x_p = g.x[perm]
    edge_index_p = inv[g.edge_index]    # remap both src and dst with the inverse
    edge_attr_p = g.edge_attr           # edge features unchanged

    with torch.no_grad():
        y_perm = model(x_p, edge_index_p, edge_attr_p).item()

    print(f"\n--- permutation invariance (caffeine) ---")
    print(f"original atom order : y = {y_orig:.6f}")
    print(f"permuted atom order : y = {y_perm:.6f}")
    print(f"difference          : {abs(y_orig - y_perm):.2e}  "
          f"(should be ~0 up to float noise)")


def batch_of_three() -> None:
    """
    The 'disconnected supergraph' batching trick. Stack three molecules into
    one big graph by:
      1. shifting each molecule's edge_index by the running node offset;
      2. concatenating x and edge_attr;
      3. building a batch vector of length N that labels which graph each
         atom came from.
    No padding, no masking. Scales linearly with total atom count.
    """
    torch.manual_seed(0)
    model = MPNN(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, pool="sum_mean", out_dim=1).eval()

    smiles = ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    graphs = [smiles_to_graph(s) for s in smiles]

    xs, eis, eas, batches = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        eas.append(g.edge_attr)
        batches.append(torch.full((g.num_atoms,), i, dtype=torch.long))
        offset += g.num_atoms

    x = torch.cat(xs)
    edge_index = torch.cat(eis, dim=1)
    edge_attr = torch.cat(eas)
    batch = torch.cat(batches)

    with torch.no_grad():
        y = model(x, edge_index, edge_attr, batch=batch, num_graphs=len(graphs))

    print(f"\n--- batch of 3 molecules ---")
    print(f"total atoms: {x.shape[0]}  (ethanol 3 + benzene 6 + aspirin 13 = 22)")
    print(f"total edges: {edge_index.shape[1]}  (4 + 12 + 26 = 42)")
    print(f"output shape: {tuple(y.shape)}  (expected: (3, 1))")
    for i, s in enumerate(smiles):
        print(f"  {s:40s}  y = {y[i, 0].item():+.4f}")

    # Cross-check: each prediction should match what the model gives standalone.
    y_single = torch.stack([model(g.x, g.edge_index, g.edge_attr).view(-1) for g in graphs])
    max_diff = (y - y_single).abs().max().item()
    print(f"\nvs. predicting each molecule separately, max |diff|: {max_diff:.2e}  "
          f"(should be ~0 - batching must not change the output)")


if __name__ == "__main__":
    single_molecule()
    permutation_invariance()
    batch_of_three()
