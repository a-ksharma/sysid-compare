"""Model export utilities: serialization to .npz and JSON model cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sysid.analysis.linearize import linearize_numpy_neural_ode
from sysid.neural.model import NeuralODE
from sysid.neural.numpy_infer import NumpyNeuralODE


def complex_to_dict(c: complex | np.number[Any] | float) -> dict[str, float]:
    """Serialize a complex number to a dictionary."""
    val = complex(c)
    return {"real": float(val.real), "imag": float(val.imag)}


def extract_weights_from_model(model: NeuralODE) -> dict[str, np.ndarray]:
    """Extract weight and bias matrices from a PyTorch NeuralODE model."""
    linear_layers: list[torch.nn.Linear] = [
        m for m in model.vector_field.net.modules() if isinstance(m, torch.nn.Linear)
    ]
    if len(linear_layers) != 3:
        raise ValueError(f"Expected 3 linear layers, got {len(linear_layers)}")

    W1 = linear_layers[0].weight.detach().cpu().numpy()
    b1 = linear_layers[0].bias.detach().cpu().numpy()
    W2 = linear_layers[1].weight.detach().cpu().numpy()
    b2 = linear_layers[1].bias.detach().cpu().numpy()
    W3 = linear_layers[2].weight.detach().cpu().numpy()
    b3 = linear_layers[2].bias.detach().cpu().numpy()
    C = model.readout.weight.detach().cpu().numpy()

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
        "C": C,
        "Ts": np.array(model.Ts, dtype=np.float64),
        "u_scale": np.array(model.u_scale, dtype=np.float64),
        "y_scale": np.array(model.y_scale, dtype=np.float64),
        "n_x": np.array(model.n_x, dtype=np.int64),
    }


def export_neural_ode(
    model: NeuralODE,
    metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Export a trained NeuralODE to neural_ode.npz and neural_ode.json model card.

    Parameters
    ----------
    model : NeuralODE
        Trained model instance.
    metadata : dict[str, Any]
        Model card metadata (seed, epochs, loss, versions, git commit, etc.).
    output_dir : Path
        Target directory (e.g. artifacts/models/msd).

    Returns
    -------
    tuple[Path, Path]
        (npz_path, json_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "neural_ode.npz"
    json_path = output_dir / "neural_ode.json"

    arrays = extract_weights_from_model(model)
    # Save npz for NumPy inference
    np.savez(file=str(npz_path), **arrays)  # type: ignore[arg-type]

    # Build NumPy model to compute exact continuous linearization
    numpy_model = NumpyNeuralODE(
        n_x=model.n_x,
        Ts=model.Ts,
        u_scale=model.u_scale,
        y_scale=model.y_scale,
        W1=arrays["W1"].astype(np.float32),
        b1=arrays["b1"].astype(np.float32),
        W2=arrays["W2"].astype(np.float32),
        b2=arrays["b2"].astype(np.float32),
        W3=arrays["W3"].astype(np.float32),
        b3=arrays["b3"].astype(np.float32),
        C=arrays["C"].astype(np.float32),
    )

    A, B, C, D, poles = linearize_numpy_neural_ode(numpy_model)

    model_card = dict(metadata)
    model_card["linearization"] = {
        "A": A.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
        "D": D.tolist(),
        "poles": [complex_to_dict(p) for p in poles],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, sort_keys=True)

    return npz_path, json_path
