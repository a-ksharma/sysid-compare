"""Dataset file I/O utilities for saving and loading arrays with JSON sidecars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_file_sha256(filepath: Path) -> str:
    """Compute the SHA256 hex digest of a file.

    Parameters
    ----------
    filepath : Path
        Target file path.

    Returns
    -------
    str
        Hexadecimal SHA256 string.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_dataset(
    output_prefix: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save dataset arrays to an .npz file and metadata to a JSON sidecar.

    Computes and injects the SHA256 checksum of the .npz into the JSON sidecar.

    Parameters
    ----------
    output_prefix : Path
        Base path without extension (e.g. data/msd/train_snr_40).
    arrays : dict[str, np.ndarray]
        Mapping from array names to NumPy arrays.
    metadata : dict[str, Any]
        Sidecar metadata fields.

    Returns
    -------
    tuple[Path, Path]
        Paths to the created (npz_path, json_path).
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    # Save npz file (uncompressed for deterministic byte stream)
    np.savez(file=str(npz_path), **arrays)  # type: ignore[arg-type]

    # Calculate content hash of the npz
    npz_sha256 = compute_file_sha256(npz_path)

    sidecar_data = dict(metadata)
    sidecar_data["npz_file"] = npz_path.name
    sidecar_data["sha256"] = npz_sha256

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2, sort_keys=True)

    return npz_path, json_path


def load_dataset(prefix_or_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load dataset arrays from .npz and metadata from the JSON sidecar.

    Verifies the SHA256 integrity against the sidecar record.

    Parameters
    ----------
    prefix_or_path : Path
        Prefix path or direct path to the .npz or .json file.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        (arrays_dict, metadata_dict)
    """
    if prefix_or_path.suffix in (".npz", ".json"):
        base_prefix = prefix_or_path.with_suffix("")
    else:
        base_prefix = prefix_or_path

    npz_path = base_prefix.with_suffix(".npz")
    json_path = base_prefix.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset npz file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset sidecar json file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    # Verify checksum
    actual_hash = compute_file_sha256(npz_path)
    expected_hash = metadata.get("sha256")
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            f"Checksum mismatch for {npz_path.name}: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    with np.load(npz_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    return arrays, metadata
