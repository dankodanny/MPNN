"""
SMILES -> molecular graph tensors (x, edge_index, edge_attr).

Feature choices follow Gilmer et al. (2017) "Neural Message Passing for Quantum
Chemistry", lightly simplified. The output is compatible with PyTorch Geometric
conventions so we can drop it into PyG later without changing the featurizer.
"""
from dataclasses import dataclass
from typing import List

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, BondType, HybridizationType


ATOM_SYMBOLS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.UNSPECIFIED,
]
BOND_TYPES = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]


def _one_hot(value, choices: list) -> List[int]:
    # Last bucket is the catch-all for anything not in `choices`.
    vec = [0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1
    return vec


def atom_features(atom: Atom) -> List[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [atom.GetDegree() / 6.0]          # normalize ordinal counts to ~[0, 1]
        + [atom.GetFormalCharge() / 3.0]
        + [atom.GetTotalNumHs() / 4.0]
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [1.0 if atom.IsInRing() else 0.0]
    )


def bond_features(bond: Bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
    )


ATOM_FEATURE_DIM = len(ATOM_SYMBOLS) + len(HYBRIDIZATIONS) + 5
BOND_FEATURE_DIM = len(BOND_TYPES) + 2


@dataclass
class MolGraph:
    x: torch.Tensor           # (N, ATOM_FEATURE_DIM)
    edge_index: torch.Tensor  # (2, 2E) — both directions per bond
    edge_attr: torch.Tensor   # (2E, BOND_FEATURE_DIM)
    num_atoms: int

    def __repr__(self) -> str:
        return (
            f"MolGraph(atoms={self.num_atoms}, "
            f"edges={self.edge_index.shape[1]}, "
            f"x={tuple(self.x.shape)}, "
            f"edge_attr={tuple(self.edge_attr.shape)})"
        )


def smiles_to_graph(smiles: str) -> MolGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )

    src: List[int] = []
    dst: List[int] = []
    e_attr: List[List[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        # Add both directions so messages can flow either way along the bond.
        src.extend([i, j])
        dst.extend([j, i])
        e_attr.extend([feat, feat])

    if not src:
        # Single-atom molecules (e.g. "[He]") have no bonds.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(e_attr, dtype=torch.float)

    return MolGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_atoms=mol.GetNumAtoms(),
    )
