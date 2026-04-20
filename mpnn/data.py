"""
ESOL dataset loader and batching for molecular graphs.

ESOL (Delaney 2004): ~1128 molecules with measured water solubility
(log mol/L). Small regression dataset that fits comfortably on CPU.
Downloaded from the DeepChem mirror on first use and cached under `data/`.

The collate_mols function merges a batch of MolGraphs into one disconnected
supergraph and emits a `batch` vector so the model can pool per-graph at the
readout. No padding, no masking.
"""
import csv
import os
import urllib.request
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from mpnn.featurize import MolGraph, smiles_to_graph


ESOL_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/"
    "datasets/delaney-processed.csv"
)
ESOL_SMILES_COL = "smiles"
ESOL_LABEL_COL = "measured log solubility in mols per litre"


def download_esol(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(ESOL_URL, path)


def load_esol(cache_dir: str = "data") -> List[Tuple[str, float]]:
    """Return a list of (smiles, log_solubility) pairs."""
    path = os.path.join(cache_dir, "esol.csv")
    download_esol(path)
    rows: List[Tuple[str, float]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row[ESOL_SMILES_COL], float(row[ESOL_LABEL_COL])))
    return rows


class MolDataset(Dataset):
    """Holds pre-computed MolGraphs + labels so we don't re-featurize every epoch."""

    def __init__(self, items: List[Tuple[str, float]]):
        self.graphs: List[MolGraph] = []
        self.ys: List[float] = []
        self.smiles: List[str] = []
        for smi, y in items:
            self.graphs.append(smiles_to_graph(smi))
            self.ys.append(y)
            self.smiles.append(smi)

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int) -> Tuple[MolGraph, float]:
        return self.graphs[idx], self.ys[idx]


def collate_mols(batch: List[Tuple[MolGraph, float]]) -> Dict[str, torch.Tensor]:
    """
    Merge a list of (MolGraph, y) into one supergraph dict:

        x          : (sum_N, F_atom)
        edge_index : (2, sum_2E)            shifted by running node offset
        edge_attr  : (sum_2E, F_bond)
        batch      : (sum_N,) long, atom -> graph index
        y          : (B, 1)
        num_graphs : int
    """
    xs: List[torch.Tensor] = []
    eis: List[torch.Tensor] = []
    eas: List[torch.Tensor] = []
    batches: List[torch.Tensor] = []
    ys: List[float] = []
    offset = 0
    for i, (g, y) in enumerate(batch):
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        eas.append(g.edge_attr)
        batches.append(torch.full((g.num_atoms,), i, dtype=torch.long))
        ys.append(y)
        offset += g.num_atoms

    return {
        "x": torch.cat(xs),
        "edge_index": torch.cat(eis, dim=1),
        "edge_attr": torch.cat(eas),
        "batch": torch.cat(batches),
        "y": torch.tensor(ys, dtype=torch.float).unsqueeze(1),
        "num_graphs": len(batch),
    }
