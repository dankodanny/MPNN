"""
Run: python -m scripts.07_train_esol_pyg

Same ESOL training as scripts/06_train_esol.py, but using the PyG-based
model (mpnn/pyg_model.py). Compare:
  - lines of code (this file + pyg_model.py vs 06_train_esol.py + model.py
    + encoder.py + message.py + aggregate.py + update.py + readout.py);
  - epoch time;
  - final test RMSE (should land in the same ballpark).
"""
import os
import random
import time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mpnn.data import load_esol
from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph
from mpnn.pyg_model import MPNNPyG


SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
HIDDEN_DIM = 64
NUM_STEPS = 3


def to_pyg_data(smi: str, y: float) -> Data:
    """Wrap smiles_to_graph output in a PyG Data object."""
    g = smiles_to_graph(smi)
    return Data(
        x=g.x,
        edge_index=g.edge_index,
        edge_attr=g.edge_attr,
        y=torch.tensor([[y]], dtype=torch.float),
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("Using CPU (CUDA not available)")
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
def evaluate(model, loader, y_mean, y_std, device):
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


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = pick_device()

    print("Loading ESOL (uses data/esol.csv if already cached)...")
    items = load_esol()
    print(f"  {len(items)} molecules")

    random.shuffle(items)
    n = len(items)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    print(f"  split: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

    # Convert to PyG Data objects. PyG's DataLoader handles the batching.
    train_data = [to_pyg_data(smi, y) for smi, y in train_items]
    val_data = [to_pyg_data(smi, y) for smi, y in val_items]
    test_data = [to_pyg_data(smi, y) for smi, y in test_items]

    y_train = torch.tensor([y for _, y in train_items])
    y_mean = y_train.mean().item()
    y_std = y_train.std().item()
    print(f"  train target stats: mean = {y_mean:.3f}, std = {y_std:.3f}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = MPNNPyG(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_steps=NUM_STEPS,
        pool="sum_mean",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "val_mae": [], "epoch_time": []}
    best_val, best_epoch = float("inf"), 0
    print("\nepoch | train_loss |  val_rmse |  val_mae |   time |")
    print("------+------------+-----------+----------+--------+------")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, y_mean, y_std, device)
        val_rmse, val_mae = evaluate(model, val_loader, y_mean, y_std, device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        history["epoch_time"].append(dt)

        is_best = val_rmse < best_val
        if is_best:
            best_val, best_epoch = val_rmse, epoch
        marker = "  <- best" if is_best else ""
        print(f" {epoch:4d} | {train_loss:10.4f} | {val_rmse:9.4f} | {val_mae:8.4f} | {dt:5.2f}s |{marker}")

    mean_epoch = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"\nmean epoch time: {mean_epoch:.2f}s on {device}")

    test_rmse, test_mae = evaluate(model, test_loader, y_mean, y_std, device)
    print(f"best val RMSE (epoch {best_epoch}): {best_val:.4f}")
    print(f"test RMSE: {test_rmse:.4f} log mol/L")
    print(f"test MAE : {test_mae:.4f} log mol/L")

    _save_plot(history)


def _save_plot(history: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"])
    ax[0].set_title("PyG MPNN: train loss (normalized MSE)")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_rmse"], label="val RMSE")
    ax[1].plot(history["val_mae"], label="val MAE")
    ax[1].set_title("PyG MPNN: validation metrics (log mol/L)")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    out = "results/esol_training_pyg.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved training curves to {out}")


if __name__ == "__main__":
    main()
