"""Unit tests for Phase 2 datasets generation and machine learning hygiene."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from sysid.config import (
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_SYSTEMS_PATH,
    load_experiment,
    load_systems,
)
from sysid.data_io import load_dataset, save_dataset
from sysid.systems import get_all_plants


def test_dataset_reproducibility(tmp_path: Path) -> None:
    """Acceptance criterion 1: Generation twice produces identical hashes."""
    from scripts.make_datasets import generate_plant_datasets

    systems_cfg = load_systems(DEFAULT_SYSTEMS_PATH)
    exp_cfg = load_experiment(DEFAULT_EXPERIMENT_PATH)
    plants = get_all_plants(systems_cfg)

    msd_plant = plants["msd"]

    dir_run1 = tmp_path / "run1"
    dir_run2 = tmp_path / "run2"

    hashes1 = generate_plant_datasets(msd_plant, exp_cfg, dir_run1, base_seed=42)
    hashes2 = generate_plant_datasets(msd_plant, exp_cfg, dir_run2, base_seed=42)

    assert set(hashes1.keys()) == set(hashes2.keys())
    for fname in hashes1:
        assert hashes1[fname] == hashes2[fname], (
            f"Hash mismatch for {fname}: run1={hashes1[fname]}, run2={hashes2[fname]}"
        )


def test_no_sequence_appears_in_multiple_splits() -> None:
    """Acceptance criterion 2: No sequence appears in two splits."""
    data_dir = Path("data") / "msd"
    if not (data_dir / "train_snr_inf.npz").exists():
        pytest.skip("Datasets not generated yet")

    train_data, _ = load_dataset(data_dir / "train_snr_inf")
    val_data, _ = load_dataset(data_dir / "val_snr_inf")
    test_data, _ = load_dataset(data_dir / "test_iid_snr_inf")

    u_train = train_data["u"]
    u_val = val_data["u"]
    u_test = test_data["u"]

    # Verify no training sequence equals any validation sequence
    for i, u_tr in enumerate(u_train):
        for j, u_v in enumerate(u_val):
            diff = float(np.linalg.norm(u_tr - u_v))
            assert diff > 0.5, (
                f"Train sequence {i} matches Val sequence {j} (diff={diff})"
            )

    # Verify no training sequence equals any test sequence
    for i, u_tr in enumerate(u_train):
        for j, u_te in enumerate(u_test):
            diff = float(np.linalg.norm(u_tr - u_te))
            assert diff > 0.5, (
                f"Train sequence {i} matches Test sequence {j} (diff={diff})"
            )

    # Verify no validation sequence equals any test sequence
    for i, u_v in enumerate(u_val):
        for j, u_te in enumerate(u_test):
            diff = float(np.linalg.norm(u_v - u_te))
            assert diff > 0.5, (
                f"Val sequence {i} matches Test sequence {j} (diff={diff})"
            )


def test_training_code_never_references_test_or_ood_sets() -> None:
    """Acceptance criterion 3: Test and OOD files not read by training code.

    Enforces strict ML hygiene by scanning all modules under src/sysid/neural/ and
    src/sysid/classical/ to guarantee no code references test_iid or ood splits.
    """
    forbidden_tokens = [
        "test_iid",
        "ood_amplitude",
        "ood_frequency",
        "ood_waveform",
    ]

    source_dirs = [
        Path("src/sysid/neural"),
        Path("src/sysid/classical"),
    ]

    for s_dir in source_dirs:
        for py_file in s_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                pattern = rf"\b{re.escape(token)}\b"
                match = re.search(pattern, content)
                assert match is None, (
                    f"Machine learning hygiene violation: forbidden token '{token}' "
                    f"found in training module {py_file}"
                )


def test_data_io_checksum_verification(tmp_path: Path) -> None:
    """Verify load_dataset raises ValueError when file content is modified."""
    arrays = {"x": np.array([1.0, 2.0, 3.0])}
    metadata = {"info": "test"}

    prefix = tmp_path / "sample"
    npz_path, json_path = save_dataset(prefix, arrays, metadata)

    # Initial load passes
    loaded_arrays, loaded_meta = load_dataset(prefix)
    np.testing.assert_array_equal(loaded_arrays["x"], arrays["x"])
    assert loaded_meta["info"] == "test"

    # Tamper with the npz file
    with open(npz_path, "ab") as f:
        f.write(b"tampered")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_dataset(prefix)
