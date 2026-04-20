"""
Training / evaluation helpers for regression on molecular graphs.

Targets are normalized (zero mean, unit std using train-set stats only) before
passing to the loss; evaluation de-normalizes so metrics are reported in the
original target units (e.g., log mol/L for ESOL).
"""
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _move(batch: dict, device: str) -> dict:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    device: str = "cpu",
) -> float:
    """Return the mean per-molecule training loss (on normalized targets)."""
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        batch = _move(batch, device)
        y_norm = (batch["y"] - y_mean) / y_std
        pred = model(
            batch["x"], batch["edge_index"], batch["edge_attr"],
            batch=batch["batch"], num_graphs=batch["num_graphs"],
        )
        loss = loss_fn(pred, y_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch["num_graphs"]
        total_n += batch["num_graphs"]
    return total_loss / total_n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Return (RMSE, MAE) in the original target units."""
    model.eval()
    sq_err = 0.0
    abs_err = 0.0
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        pred_norm = model(
            batch["x"], batch["edge_index"], batch["edge_attr"],
            batch=batch["batch"], num_graphs=batch["num_graphs"],
        )
        pred = pred_norm * y_std + y_mean
        diff = pred - batch["y"]
        sq_err += (diff ** 2).sum().item()
        abs_err += diff.abs().sum().item()
        n += batch["num_graphs"]
    return (sq_err / n) ** 0.5, abs_err / n
