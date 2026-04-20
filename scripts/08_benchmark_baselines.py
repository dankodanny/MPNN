"""
Run: python -m scripts.08_benchmark_baselines

Head-to-head on ESOL: MPNN (edge-aware) vs GCN, GAT, GIN (no edge features).

Controls for fairness:
  - identical train/val/test split (seeded shuffle, computed once);
  - identical target normalization (train-set mean/std);
  - identical optimizer / LR / batch size / epochs for all models;
  - same hidden_dim and num_steps across all models.

Parameter counts will differ because GIN has an MLP per layer, MPNN uses
GRUCell, GAT has attention weights, GCN is the leanest. That's honest.
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
from mpnn.pyg_baselines import GAT, GCN, GIN
from mpnn.pyg_model import MPNNPyG


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
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    build_model: Callable[[], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_mean: float,
    y_std: float,
    device: str,
) -> Dict[str, float]:
    # Re-seed BEFORE building the model so weight init matches run to run.
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_epoch = float("inf"), 0
    epoch_times: List[float] = []
    print(f"\n=== {name} ({n_params:,} params) ===")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, y_mean, y_std, device,
        )
        val_rmse, val_mae = evaluate(model, val_loader, y_mean, y_std, device)
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        if val_rmse < best_val:
            best_val, best_epoch = val_rmse, epoch

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val_rmse {val_rmse:.4f}  "
                  f"val_mae {val_mae:.4f}")

    test_rmse, test_mae = evaluate(model, test_loader, y_mean, y_std, device)
    mean_epoch = sum(epoch_times) / len(epoch_times)
    print(f"  --> best val RMSE {best_val:.4f} (epoch {best_epoch}), "
          f"test RMSE {test_rmse:.4f}, test MAE {test_mae:.4f}, "
          f"mean epoch {mean_epoch:.2f}s")

    return {
        "params": n_params,
        "best_val_rmse": best_val,
        "best_epoch": best_epoch,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "mean_epoch_time": mean_epoch,
    }


def main() -> None:
    random.seed(SEED)
    device = pick_device()
    print(f"Device: {device}  "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    print("\nLoading ESOL...")
    items = load_esol()
    random.shuffle(items)
    n = len(items)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    print(f"  split: {len(train_items)} / {len(val_items)} / {len(test_items)}")

    train_data = [to_pyg_data(s, y) for s, y in train_items]
    val_data = [to_pyg_data(s, y) for s, y in val_items]
    test_data = [to_pyg_data(s, y) for s, y in test_items]

    y_train = torch.tensor([y for _, y in train_items])
    y_mean, y_std = y_train.mean().item(), y_train.std().item()

    # Use same seeds for loader shuffling across models.
    def make_loaders():
        g = torch.Generator()
        g.manual_seed(SEED)
        tl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        vl = DataLoader(val_data, batch_size=BATCH_SIZE)
        ttl = DataLoader(test_data, batch_size=BATCH_SIZE)
        return tl, vl, ttl

    models_to_run = [
        ("MPNN (edge-aware)", lambda: MPNNPyG(
            atom_feature_dim=ATOM_FEATURE_DIM,
            bond_feature_dim=BOND_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM, num_steps=NUM_STEPS, pool="sum_mean",
        )),
        ("GCN", lambda: GCN(
            atom_feature_dim=ATOM_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM, num_steps=NUM_STEPS, pool="sum_mean",
        )),
        ("GAT (4 heads)", lambda: GAT(
            atom_feature_dim=ATOM_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM, num_steps=NUM_STEPS, heads=4, pool="sum_mean",
        )),
        ("GIN", lambda: GIN(
            atom_feature_dim=ATOM_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM, num_steps=NUM_STEPS, pool="sum_mean",
        )),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for name, builder in models_to_run:
        train_loader, val_loader, test_loader = make_loaders()
        results[name] = run_one(
            name, builder, train_loader, val_loader, test_loader,
            y_mean, y_std, device,
        )

    # Final comparison table.
    print("\n" + "=" * 78)
    print(f"{'Model':<22} {'Params':>9} {'Val RMSE':>10} {'Test RMSE':>11} "
          f"{'Test MAE':>10} {'s/epoch':>9}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:<22} {r['params']:>9,} {r['best_val_rmse']:>10.4f} "
              f"{r['test_rmse']:>11.4f} {r['test_mae']:>10.4f} {r['mean_epoch_time']:>9.2f}")
    print("=" * 78)
    print("Edge-awareness = MPNN column uses bond features; others do not.")

    _save_plot(results)


def _save_plot(results: Dict[str, Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs("results", exist_ok=True)

    names = list(results.keys())
    test_rmse = [results[n]["test_rmse"] for n in names]
    params = [results[n]["params"] for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(names, test_rmse, color=["C0", "C1", "C2", "C3"])
    ax[0].set_ylabel("test RMSE (log mol/L)")
    ax[0].set_title("ESOL test RMSE: lower is better")
    ax[0].tick_params(axis="x", rotation=20)

    ax[1].bar(names, params, color=["C0", "C1", "C2", "C3"])
    ax[1].set_ylabel("parameters")
    ax[1].set_title("Parameter count")
    ax[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out = "results/baselines_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved comparison plot to {out}")


if __name__ == "__main__":
    main()
