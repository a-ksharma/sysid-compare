"""Training loop for Neural ODE system identification models.

Implements horizon curriculum, cosine learning rate scheduling, early stopping
on validation loss, and reproducible seed initialization.
"""

from __future__ import annotations

import copy
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
import torch.nn as nn
import torchdiffeq

from sysid.data_io import compute_file_sha256, load_dataset
from sysid.neural.export import export_neural_ode
from sysid.neural.model import NeuralODE

logger = logging.getLogger(__name__)


def compute_fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the normalized goodness-of-fit metric in percent.

    Formula:
        fit = 100 * (1 - ||y - y_hat||_2 / ||y - mean(y)||_2)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth output array.
    y_pred : np.ndarray
        Model predicted output array.

    Returns
    -------
    float
        Fit percentage.
    """
    diff_norm = float(np.linalg.norm(y_true - y_pred))
    denom = float(np.linalg.norm(y_true - np.mean(y_true)))
    if denom < 1e-12:
        return 0.0
    return float(100.0 * (1.0 - diff_norm / denom))


def get_git_commit() -> str:
    """Retrieve the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_library_versions() -> dict[str, str]:
    """Retrieve runtime versions of key libraries."""
    return {
        "torch": str(torch.__version__),
        "torchdiffeq": str(torchdiffeq.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
    }


def train_neural_ode(
    plant_id: str,
    train_data_path: Path,
    val_data_path: Path,
    seed: int = 0,
    n_x: int = 4,
    hidden_dim: int = 64,
    n_hidden: int = 2,
    lr_max: float = 0.01,
    lr_min: float = 0.0001,
    weight_decay: float = 0.00001,
    max_epochs: int = 1500,
    batch_size: int = 12,
    patience: int = 100,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> tuple[NeuralODE, dict[str, Any]]:
    """Train a Neural ODE on train split with early stopping on val split.

    Parameters
    ----------
    plant_id : str
        Target plant id.
    train_data_path : Path
        Path to train split prefix (without extension).
    val_data_path : Path
        Path to val split prefix (without extension).
    seed : int, optional
        Pseudorandom seed, by default 0.
    n_x : int, optional
        Latent dimension, by default 4.
    hidden_dim : int, optional
        MLP width, by default 64.
    n_hidden : int, optional
        Number of hidden layers, by default 2.
    lr_max : float, optional
        Initial learning rate, by default 0.01.
    lr_min : float, optional
        Final learning rate after cosine decay, by default 0.0001.
    weight_decay : float, optional
        L2 weight decay, by default 1e-5.
    max_epochs : int, optional
        Maximum training epochs, by default 1500.
    batch_size : int, optional
        Batch size, by default 12.
    patience : int, optional
        Early stopping patience epochs, by default 100.
    device : str, optional
        Torch device, by default 'cpu'.
    output_dir : Path | None, optional
        Directory to export trained artifacts.

    Returns
    -------
    tuple[NeuralODE, dict[str, Any]]
        (best_model, training_metadata)
    """
    # Deterministic seeding on CPU
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Data
    train_arrays, train_meta = load_dataset(train_data_path)
    val_arrays, _ = load_dataset(val_data_path)

    u_train_raw = train_arrays["u"]
    y_train_raw = train_arrays.get("y_noisy", train_arrays["y_clean"])
    u_val_raw = val_arrays["u"]
    y_val_raw = val_arrays.get("y_noisy", val_arrays["y_clean"])

    Ts = float(train_meta["Ts"])

    # Normalization statistics computed on train only (Rule 4a.3)
    u_scale = float(np.max(np.abs(u_train_raw)))
    if u_scale < 1e-9:
        u_scale = 1.0
    y_scale = float(np.std(y_train_raw))
    if y_scale < 1e-9:
        y_scale = 1.0

    u_train_n = torch.tensor(u_train_raw / u_scale, dtype=torch.float32, device=device)
    y_train_n = torch.tensor(y_train_raw / y_scale, dtype=torch.float32, device=device)
    u_val_n = torch.tensor(u_val_raw / u_scale, dtype=torch.float32, device=device)
    y_val_n = torch.tensor(y_val_raw / y_scale, dtype=torch.float32, device=device)

    n_train_seqs, n_samples = u_train_n.shape

    # 2. Build Model
    model = NeuralODE(
        n_x=n_x,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        Ts=Ts,
        u_scale=u_scale,
        y_scale=y_scale,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_max,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    # Curriculum definitions: fractions at 0%, 25%, 50%, 75% of epochs
    curriculum_fractions = [0.2, 0.4, 0.7, 1.0]
    epoch_milestones = [
        0,
        int(0.25 * max_epochs),
        int(0.50 * max_epochs),
        int(0.75 * max_epochs),
    ]

    best_val_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_counter = 0

    t_start = time.time()

    # Full time grid for validation
    t_val_full = torch.arange(n_samples, device=device, dtype=torch.float32) * Ts

    for epoch in range(max_epochs):
        model.train()

        # Determine curriculum horizon
        active_frac = 1.0
        for frac, milestone in zip(curriculum_fractions, epoch_milestones, strict=True):
            if epoch >= milestone:
                active_frac = frac
        curr_len = max(10, int(round(active_frac * n_samples)))
        t_train_curr = torch.arange(curr_len, device=device, dtype=torch.float32) * Ts

        # Cosine learning rate decay
        lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1.0 + math.cos(math.pi * epoch / max_epochs)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Shuffle batches
        indices = torch.randperm(n_train_seqs)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, n_train_seqs, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            u_b = u_train_n[b_idx, :curr_len]
            y_b = y_train_n[b_idx, :curr_len]

            optimizer.zero_grad()
            y_pred_b = model(u_b, t_grid=t_train_curr)
            loss = criterion(y_pred_b, y_b)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        # Validation on full horizon (free-run)
        model.eval()
        with torch.no_grad():
            y_val_pred = model(u_val_n, t_grid=t_val_full)
            val_loss = float(criterion(y_val_pred, y_val_n).item())

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    train_time_s = time.time() - t_start

    # Final validation metrics in physical units
    model.eval()
    with torch.no_grad():
        y_val_pred_norm = model(u_val_n, t_grid=t_val_full).cpu().numpy()
        y_val_pred_phys = y_val_pred_norm * y_scale

    val_fit = compute_fit_percent(y_val_raw, y_val_pred_phys)

    train_npz_path = train_data_path.with_suffix(".npz")
    dataset_hash = (
        compute_file_sha256(train_npz_path) if train_npz_path.exists() else "unknown"
    )

    metadata: dict[str, Any] = {
        "plant_id": plant_id,
        "seed": seed,
        "n_x": n_x,
        "hidden_dim": hidden_dim,
        "n_hidden": n_hidden,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "validation_fit_percent": val_fit,
        "training_time_s": train_time_s,
        "git_commit": get_git_commit(),
        "dataset_hash": dataset_hash,
        "library_versions": get_library_versions(),
        "u_scale": u_scale,
        "y_scale": y_scale,
        "Ts": Ts,
    }

    if output_dir is not None:
        export_neural_ode(model, metadata, output_dir)

    return model, metadata
