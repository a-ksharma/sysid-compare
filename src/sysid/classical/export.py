"""Export classical identified models (ARX and Prony) to JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sysid.classical.arx import ARXModel, fit_arx
from sysid.classical.prony import PronyModel, fit_prony
from sysid.data_io import load_dataset
from sysid.systems import PlantModel, extract_identifiable_parameters


def complex_to_dict(c: complex | np.number[Any] | float) -> dict[str, float]:
    """Serialize a complex number to a JSON-compatible dictionary."""
    val = complex(c)
    return {"real": float(val.real), "imag": float(val.imag)}


def serialize_arx(model: ARXModel, plant_id: str) -> dict[str, Any]:
    """Serialize an ARXModel to a dictionary."""
    identifiable = extract_identifiable_parameters(plant_id, model.num, model.den)
    return {
        "order": model.order,
        "Ts": model.Ts,
        "a_coeffs": [float(x) for x in model.a_coeffs],
        "b_coeffs": [float(x) for x in model.b_coeffs],
        "A": model.A.tolist(),
        "B": model.B.tolist(),
        "C": model.C.tolist(),
        "D": model.D.tolist(),
        "num": [float(x) for x in model.num],
        "den": [float(x) for x in model.den],
        "poles": [complex_to_dict(p) for p in model.poles],
        "discrete_poles": [complex_to_dict(p) for p in model.discrete_poles],
        "identifiable_parameters": identifiable,
    }


def serialize_prony(model: PronyModel, plant_id: str) -> dict[str, Any]:
    """Serialize a PronyModel to a dictionary."""
    try:
        identifiable = extract_identifiable_parameters(plant_id, model.num, model.den)
    except Exception:
        identifiable = {}

    return {
        "order": model.order,
        "Ts": model.Ts,
        "A": model.A.tolist(),
        "B": model.B.tolist(),
        "C": model.C.tolist(),
        "D": model.D.tolist(),
        "num": [float(x) for x in model.num],
        "den": [float(x) for x in model.den],
        "poles": [complex_to_dict(p) for p in model.continuous_poles],
        "discrete_poles": [complex_to_dict(p) for p in model.discrete_poles],
        "residues": [complex_to_dict(r) for r in model.residues],
        "identifiable_parameters": identifiable,
        "is_stable": model.is_stable,
        "has_nyquist_artifact": model.has_nyquist_artifact,
        "warning": model.warning,
    }


def export_classical_models(
    plant: PlantModel,
    data_dir: Path,
    artifacts_dir: Path,
    snr_tag: str = "40",
) -> Path:
    """Fit ARX and Prony on baseline data and export classical.json artifact.

    Parameters
    ----------
    plant : PlantModel
        Plant model.
    data_dir : Path
        Path to data/<plant_id>/.
    artifacts_dir : Path
        Root artifacts directory (e.g. artifacts/).
    snr_tag : str, optional
        SNR tag to use for training ARX (default '40' for 40 dB baseline).

    Returns
    -------
    Path
        Path to written classical.json.
    """
    plant_data_dir = data_dir / plant.id

    # Load ARX training data
    train_arrays, _ = load_dataset(plant_data_dir / f"train_snr_{snr_tag}")
    u_train = train_arrays["u"]
    y_train = train_arrays["y_noisy"]

    arx_model = fit_arx(
        u_train, y_train, order=plant.order, Ts=plant.derived.get("Ts", 0.02)
    )

    # Load Impulse Experiment data for Prony
    impulse_arrays, _ = load_dataset(plant_data_dir / "impulse_experiment")
    h_scaled = impulse_arrays["y_scaled"]

    prony_model = fit_prony(
        h_scaled, order=plant.order, Ts=plant.derived.get("Ts", 0.02)
    )

    out_dir = artifacts_dir / "models" / plant.id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "classical.json"

    payload = {
        "plant_id": plant.id,
        "order": plant.order,
        "Ts": 0.02,
        "training_snr_db": 40.0 if snr_tag == "40" else None,
        "arx": serialize_arx(arx_model, plant.id),
        "prony": serialize_prony(prony_model, plant.id),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path
