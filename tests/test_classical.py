"""Unit tests for Phase 3 classical identification: ARX and Prony's method."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.signal

from sysid.classical import (
    discrete_to_continuous_ss,
    export_classical_models,
    fit_arx,
    fit_prony,
)
from sysid.data_io import load_dataset
from sysid.systems import extract_identifiable_parameters, get_plant


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_d2c_roundtrip(plant_id: str) -> None:
    """Acceptance criterion 2: The d2c round trip reproduces the discrete model."""
    plant = get_plant(plant_id)
    Ts = 0.02

    # Step 1: Continuous to Discrete (ZOH)
    sys_d = scipy.signal.cont2discrete(
        (plant.A, plant.B, plant.C, plant.D), Ts, method="zoh"
    )
    Ad, Bd, Cd, Dd = sys_d[0], sys_d[1], sys_d[2], sys_d[3]

    # Step 2: Discrete to Continuous (matrix log)
    A_rec, B_rec, C_rec, D_rec = discrete_to_continuous_ss(Ad, Bd, Cd, Dd, Ts)

    # Step 3: Continuous back to Discrete
    sys_d2 = scipy.signal.cont2discrete((A_rec, B_rec, C_rec, D_rec), Ts, method="zoh")
    Ad2, Bd2, Cd2, Dd2 = sys_d2[0], sys_d2[1], sys_d2[2], sys_d2[3]

    np.testing.assert_allclose(
        Ad,
        Ad2,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Ad mismatch in roundtrip for {plant_id}",
    )
    np.testing.assert_allclose(
        Bd,
        Bd2,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Bd mismatch in roundtrip for {plant_id}",
    )
    np.testing.assert_allclose(
        Cd,
        Cd2,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Cd mismatch in roundtrip for {plant_id}",
    )
    np.testing.assert_allclose(
        Dd,
        Dd2,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Dd mismatch in roundtrip for {plant_id}",
    )


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_arx_noise_free_recovery(plant_id: str) -> None:
    """Acceptance criterion 1a: ARX recovers parameters to rel error < 1e-6."""
    plant = get_plant(plant_id)
    Ts = 0.02

    data_path = Path("data") / plant_id / "train_snr_inf"
    arrays, _ = load_dataset(data_path)
    u = arrays["u"]
    y = arrays["y_clean"]

    arx_model = fit_arx(u, y, order=plant.order, Ts=Ts)

    # Check identifiable parameters
    params = extract_identifiable_parameters(plant_id, arx_model.num, arx_model.den)
    for p_name in plant.identifiable:
        true_val = plant.derived.get(p_name, plant.parameters.get(p_name))
        assert true_val is not None
        rel_err = abs(params[p_name] - true_val) / true_val
        assert rel_err < 1e-6, (
            f"ARX {plant_id} {p_name} error {rel_err:.2e} exceeded 1e-6 "
            f"(got {params[p_name]}, true {true_val})"
        )

    # Check poles
    true_poles = np.linalg.eigvals(plant.A)
    est_poles = arx_model.poles
    order_true = np.lexsort((-true_poles.imag, -true_poles.real))
    order_est = np.lexsort((-est_poles.imag, -est_poles.real))

    np.testing.assert_allclose(
        est_poles[order_est],
        true_poles[order_true],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"ARX {plant_id} poles mismatch",
    )


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_prony_noise_free_recovery(plant_id: str) -> None:
    """Acceptance criterion 1b: Prony recovers parameters to rel error < 1e-6."""
    plant = get_plant(plant_id)
    Ts = 0.02

    data_path = Path("data") / plant_id / "impulse_experiment"
    arrays, _ = load_dataset(data_path)
    h_scaled = arrays["y_scaled"]

    prony_model = fit_prony(h_scaled, order=plant.order, Ts=Ts)

    # Check stability
    assert prony_model.is_stable
    assert prony_model.warning is None

    # Check identifiable parameters
    params = extract_identifiable_parameters(plant_id, prony_model.num, prony_model.den)
    for p_name in plant.identifiable:
        true_val = plant.derived.get(p_name, plant.parameters.get(p_name))
        assert true_val is not None
        rel_err = abs(params[p_name] - true_val) / true_val
        assert rel_err < 1e-6, (
            f"Prony {plant_id} {p_name} error {rel_err:.2e} exceeded 1e-6 "
            f"(got {params[p_name]}, true {true_val})"
        )

    # Check poles
    true_poles = np.linalg.eigvals(plant.A)
    est_poles = prony_model.continuous_poles
    order_true = np.lexsort((-true_poles.imag, -true_poles.real))
    order_est = np.lexsort((-est_poles.imag, -est_poles.real))

    np.testing.assert_allclose(
        est_poles[order_est],
        true_poles[order_true],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"Prony {plant_id} poles mismatch",
    )


def test_prony_failure_mode_caught_and_reported() -> None:
    """Acceptance criterion 4: Prony failure modes caught and reported."""
    Ts = 0.02
    rng = np.random.default_rng(999)

    # Corrupt impulse record with high noise to induce non-physical/unstable poles
    h_noisy = rng.normal(0.0, 1.0, size=50)

    prony_model = fit_prony(h_noisy, order=2, Ts=Ts)

    # Must catch and report via warning if unstable or corrupted
    if not prony_model.is_stable or prony_model.has_nyquist_artifact:
        assert prony_model.warning is not None
        assert len(prony_model.warning) > 0


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_export_classical_models(plant_id: str, tmp_path: Path) -> None:
    """Verify export of artifacts/models/<plant_id>/classical.json."""
    plant = get_plant(plant_id)
    data_dir = Path("data")

    json_path = export_classical_models(plant, data_dir, tmp_path, snr_tag="40")
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["plant_id"] == plant_id
    assert "arx" in data
    assert "prony" in data
    assert "identifiable_parameters" in data["arx"]
    assert "poles" in data["arx"]
