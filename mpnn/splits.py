"""
Bemis-Murcko scaffold split for molecular datasets.

Random splits leak structural similarity between train and test; the test
set contains molecules whose core ring systems are well-represented in
train. Scaffold split groups molecules by Murcko scaffold and assigns
entire groups to one of {train, val, test}, so the test set contains
structurally NOVEL scaffolds. Generalization numbers are typically worse
(task is harder) but more honest for drug-discovery-style deployment.

Reference: Bemis & Murcko, JMC 1996, "The Properties of Known Drugs.
1. Molecular Frameworks."
"""
from collections import defaultdict
from typing import Hashable, List, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return canonical SMILES of the molecule's Bemis-Murcko scaffold.

    Returns an empty string for acyclic molecules (they all share the same
    'no scaffold' bucket — that's fine for our grouping purposes).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def scaffold_split(
    items: List[Tuple[str, float]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Split (smiles, label) pairs by Murcko scaffold.

    Strategy (matches DeepChem's scaffold splitter):
      1. Group molecules by scaffold SMILES.
      2. Sort groups by size, largest first -> goes to train. This ensures
         train covers the most common scaffolds and val/test get rarer ones.
      3. Greedily fill train up to ~train_frac, then val up to ~val_frac,
         then anything remaining goes to test.
    """
    groups: "defaultdict[Hashable, List[Tuple[str, float]]]" = defaultdict(list)
    for smi, y in items:
        groups[murcko_scaffold(smi)].append((smi, y))

    # Sort by group size desc; tiebreak by scaffold string for determinism.
    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), g[0][0]))

    n = len(items)
    target_train = int(train_frac * n)
    target_val = int(val_frac * n)

    train: List[Tuple[str, float]] = []
    val: List[Tuple[str, float]] = []
    test: List[Tuple[str, float]] = []
    for group in sorted_groups:
        if len(train) + len(group) <= target_train:
            train.extend(group)
        elif len(val) + len(group) <= target_val:
            val.extend(group)
        else:
            test.extend(group)

    return train, val, test
