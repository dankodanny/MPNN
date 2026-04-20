"""
Run: python -m scripts.06_train_esol

Train an MPNN on ESOL (water solubility regression).
Takes ~1-3 minutes on CPU. Saves a loss-curve plot to results/esol_training.png.
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mpnn.data import MolDataset, collate_mols, load_esol
from mpnn.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from mpnn.model import MPNN
from mpnn.train import evaluate, train_epoch


SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
HIDDEN_DIM = 64
NUM_STEPS = 3


def split_items(items, train_frac=0.8, val_frac=0.1):
    n = len(items)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return (
        items[:n_train],
        items[n_train:n_train + n_val],
        items[n_train + n_val:],
    )


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading ESOL (downloads on first run)...")
    items = load_esol()
    print(f"  {len(items)} molecules")

    random.shuffle(items)
    train_items, val_items, test_items = split_items(items)
    print(f"  split: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

    train_ds = MolDataset(train_items)
    val_ds = MolDataset(val_items)
    test_ds = MolDataset(test_items)

    # Target normalization using ONLY train-set stats (no leakage).
    y_train = torch.tensor([y for _, y in train_items])
    y_mean = y_train.mean().item()
    y_std = y_train.std().item()
    print(f"  train target stats: mean = {y_mean:.3f}, std = {y_std:.3f}  (log mol/L)")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_mols,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_mols)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_mols)

    model = MPNN(
        atom_feature_dim=ATOM_FEATURE_DIM,
        bond_feature_dim=BOND_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_steps=NUM_STEPS,
        pool="sum_mean",
        out_dim=1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "val_mae": []}
    best_val_rmse = float("inf")
    best_epoch = 0
    print("\nepoch | train_loss |  val_rmse |  val_mae |")
    print("------+------------+-----------+----------+------")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, y_mean=y_mean, y_std=y_std,
        )
        val_rmse, val_mae = evaluate(model, val_loader, y_mean=y_mean, y_std=y_std)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            best_epoch = epoch
        marker = "  <- best" if is_best else ""
        print(f" {epoch:4d} | {train_loss:10.4f} | {val_rmse:9.4f} | {val_mae:8.4f} |{marker}")

    test_rmse, test_mae = evaluate(model, test_loader, y_mean=y_mean, y_std=y_std)
    print(f"\nbest val RMSE (epoch {best_epoch}): {best_val_rmse:.4f}")
    print(f"test RMSE: {test_rmse:.4f} log mol/L")
    print(f"test MAE : {test_mae:.4f} log mol/L")
    print("(ESOL benchmark: strong MPNN variants reach ~0.55-0.65 test RMSE.)")

    print("\n--- a few test predictions ---")
    test_loader_single = DataLoader(
        test_ds, batch_size=1, collate_fn=collate_mols, shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader_single):
            if i >= 6:
                break
            pred_norm = model(
                batch["x"], batch["edge_index"], batch["edge_attr"],
                batch=batch["batch"], num_graphs=batch["num_graphs"],
            )
            pred = pred_norm * y_std + y_mean
            true = batch["y"].item()
            smi = test_ds.smiles[i]
            print(f"  {smi:40s}  true={true:+.3f}  pred={pred.item():+.3f}  "
                  f"err={pred.item()-true:+.3f}")

    _save_plot(history)


def _save_plot(history: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)")
        return

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].set_title("Train loss (MSE on normalized targets)")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["val_rmse"], label="val RMSE")
    ax[1].plot(history["val_mae"], label="val MAE")
    ax[1].set_title("Validation metrics (log mol/L)")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    out = "results/esol_training.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved training curves to {out}")


if __name__ == "__main__":
    main()
