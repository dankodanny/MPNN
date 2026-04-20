"""
Run: python -m scripts.09_scaffold_attention

2x2 comparison on ESOL, varying two axes:
  split   : random vs scaffold (Bemis-Murcko)
  readout : sum_mean vs set2set (attention-based)

Same MPNN architecture, same hyperparameters, same seed. The only things
that change are the split and the readout.

Expected directions:
  - Scaffold split harder than random -> RMSE UP (more honest number)
  - Set2Set more expressive than sum_mean -> RMSE DOWN
  - Combined: set2set should partially compensate for scaffold difficulty
"""
import os
import random
import time
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mpnn.data import load_esol
from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph
from mpnn.pyg_model import MPNNPyG
from mpnn.splits import scaffold_split


SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
HIDDEN_DIM = 64
NUM_STEPS = 3


def to_pyg_data(smi: str, y: float) -> Data:
    g = smiles_to_graph(smi)
    return Data(
        x=g.x,
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
        y=torch.tensor([[y]], dtype=torch.float),
    )


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def random_split(items, train_frac=0.8, val_frac=0.1):
    items = list(items)
    random.seed(SEED)
    random.shuffle(items)
    n = len(items)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def train_epoch(model, loader, optimizer, loss_fn, y_mean, y_std, device):
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y_norm = (batch.y - y_mean) / y_std
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(pred, y_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_n += batch.num_graphs
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, y_mean, y_std, device) -> Tuple[float, float]:
    model.eval()
    sq_err, abs_err, n = 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = pred_norm * y_std + y_mean
        diff = pred - batch.y
        sq_err += (diff ** 2).sum().item()
        abs_err += diff.abs().sum().item()
        n += batch.num_graphs
    return (sq_err / n) ** 0.5, abs_err / n


def run_one(
    name: str,
    train_items, val_items, test_items,
    pool: str,
    device: str,
) -> Dict[str, float]:
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_data = [to_pyg_data(s, y) for s, y in train_items]
    val_data = [to_pyg_data(s, y) for s, y in val_items]
    test_data = [to_pyg_data(s, y) for s, y in test_items]

    y_train = torch.tensor([y for _, y in train_items])
    y_mean, y_std = y_train.mean().item(), y_train.std().item()

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = MPNNPyG(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_steps=NUM_STEPS,
        pool=pool,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_epoch = float("inf"), 0
    times: List[float] = []
    print(f"\n=== {name} ({n_params:,} params) ===")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, y_mean, y_std, device)
        val_rmse, val_mae = evaluate(model, val_loader, y_mean, y_std, device)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if val_rmse < best_val:
            best_val, best_epoch = val_rmse, epoch
        if epoch % 10 == 0 or epoch in (1, EPOCHS):
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val_rmse {val_rmse:.4f}")

    test_rmse, test_mae = evaluate(model, test_loader, y_mean, y_std, device)
    mean_epoch = sum(times) / len(times)
    print(f"  --> best val RMSE {best_val:.4f} (epoch {best_epoch}), "
          f"test RMSE {test_rmse:.4f}, test MAE {test_mae:.4f}, mean {mean_epoch:.2f}s/epoch")

    return {
        "params": n_params,
        "best_val_rmse": best_val,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "mean_epoch_time": mean_epoch,
    }


def report_split_stats(name, train, val, test, items):
    print(f"\n--- {name} split ---")
    print(f"  {len(train)} train / {len(val)} val / {len(test)} test")
    # Count unique scaffolds per split so we can see the structural overlap.
    from mpnn.splits import murcko_scaffold
    def scaf_set(rows):
        return set(murcko_scaffold(s) for s, _ in rows)
    s_train, s_val, s_test = scaf_set(train), scaf_set(val), scaf_set(test)
    print(f"  unique scaffolds: {len(s_train)} train, {len(s_val)} val, {len(s_test)} test")
    print(f"  train/test scaffold overlap: {len(s_train & s_test)}  "
          f"(scaffold split -> should be 0 or tiny)")


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")

    print("\nLoading ESOL...")
    items = load_esol()

    rand_train, rand_val, rand_test = random_split(items)
    scaf_train, scaf_val, scaf_test = scaffold_split(items)
    report_split_stats("RANDOM", rand_train, rand_val, rand_test, items)
    report_split_stats("SCAFFOLD", scaf_train, scaf_val, scaf_test, items)

    configs = [
        ("random + sum_mean",   rand_train, rand_val, rand_test, "sum_mean"),
        ("random + set2set",    rand_train, rand_val, rand_test, "set2set"),
        ("scaffold + sum_mean", scaf_train, scaf_val, scaf_test, "sum_mean"),
        ("scaffold + set2set",  scaf_train, scaf_val, scaf_test, "set2set"),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for name, tr, va, te, pool in configs:
        results[name] = run_one(name, tr, va, te, pool, device)

    print("\n" + "=" * 82)
    print(f"{'Config':<24} {'Params':>8} {'Val RMSE':>10} {'Test RMSE':>11} "
          f"{'Test MAE':>10} {'s/epoch':>9}")
    print("-" * 82)
    for name, r in results.items():
        print(f"{name:<24} {r['params']:>8,} {r['best_val_rmse']:>10.4f} "
              f"{r['test_rmse']:>11.4f} {r['test_mae']:>10.4f} {r['mean_epoch_time']:>9.2f}")
    print("=" * 82)

    # Highlight the two axes.
    r_sm = results["random + sum_mean"]["test_rmse"]
    s_sm = results["scaffold + sum_mean"]["test_rmse"]
    r_s2 = results["random + set2set"]["test_rmse"]
    s_s2 = results["scaffold + set2set"]["test_rmse"]
    print(f"\nAxis 1 (readout):  sum_mean -> set2set")
    print(f"  on random  split: {r_sm:.3f} -> {r_s2:.3f}  ({r_s2-r_sm:+.3f})")
    print(f"  on scaffold split: {s_sm:.3f} -> {s_s2:.3f}  ({s_s2-s_sm:+.3f})")
    print(f"Axis 2 (split):    random -> scaffold (task gets harder)")
    print(f"  with sum_mean: {r_sm:.3f} -> {s_sm:.3f}  ({s_sm-r_sm:+.3f})")
    print(f"  with set2set:  {r_s2:.3f} -> {s_s2:.3f}  ({s_s2-r_s2:+.3f})")

    _save_plot(results)


def _save_plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs("results", exist_ok=True)
    names = list(results.keys())
    vals = [results[n]["test_rmse"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=["C0", "C1", "C2", "C3"])
    ax.set_ylabel("test RMSE (log mol/L)")
    ax.set_title("ESOL: scaffold split vs random split, sum_mean vs set2set")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    out = "results/scaffold_attention_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
