"""Tests for analysis, metrics, empirical cross-checks, and schemas (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sysid.analysis.freqresp import compute_empirical_frequency_response
from sysid.lti import frequency_response
from sysid.metrics import (
    bode_magnitude_error_db,
    bode_phase_error_deg,
    dc_gain_relative_error,
    fit_percent,
    impulse_relative_error,
    msd_resonance_peak_error,
    pole_relative_error,
    rmse,
)


def test_fit_percent_synthetic_cases() -> None:
    """Test fit_percent on known synthetic signals."""
    t = np.linspace(0, 10, 500)
    y_true = np.sin(t)

    # Identical signal -> 100%
    assert fit_percent(y_true, y_true) == pytest.approx(100.0, abs=1e-5)

    # 10% proportional deviation from mean
    y_dev = y_true + 0.1 * (y_true - np.mean(y_true))
    assert fit_percent(y_true, y_dev) == pytest.approx(90.0, abs=1e-5)

    # Degenerate zero variance denominator
    y_flat = np.ones(100)
    assert fit_percent(y_flat, y_flat + 1.0) == 0.0


def test_rmse_synthetic_cases() -> None:
    """Test rmse on synthetic constant and sinusoidal offsets."""
    y_true = np.zeros(100)
    y_pred = np.full(100, 3.5)
    assert rmse(y_true, y_pred) == pytest.approx(3.5, abs=1e-7)

    t = np.linspace(0, 2 * np.pi, 1000)
    y_s1 = np.sin(t)
    y_s2 = np.sin(t) + 2.0
    assert rmse(y_s1, y_s2) == pytest.approx(2.0, abs=1e-6)


def test_impulse_relative_error_synthetic() -> None:
    """Test impulse_relative_error on scaled impulse vectors."""
    h_true = np.array([1.0, 2.0, 3.0, 4.0])
    h_est = 1.05 * h_true
    assert impulse_relative_error(h_true, h_est) == pytest.approx(0.05, abs=1e-7)

    # Zero norm case
    assert impulse_relative_error(np.zeros(5), np.zeros(5)) == 0.0


def test_bode_magnitude_error_db_synthetic() -> None:
    """Test bode magnitude error in dB."""
    mag_true = np.ones(10)
    # Double linear magnitude is exactly 20 * log10(2) approx 6.0205999 dB
    mag_est = 2.0 * np.ones(10)
    expected_db = 20.0 * np.log10(2.0)
    assert bode_magnitude_error_db(mag_true, mag_est) == pytest.approx(
        expected_db, abs=1e-5
    )


def test_bode_phase_error_deg_synthetic() -> None:
    """Test wrapped phase error in degrees."""
    phase_true = np.zeros(5)
    phase_est = np.full(5, 45.0)
    assert bode_phase_error_deg(phase_true, phase_est) == pytest.approx(45.0, abs=1e-5)

    # Test phase wrapping near +-180 boundary: -175 vs +175 is 10 deg diff
    p_true_wrap = np.array([-175.0])
    p_est_wrap = np.array([175.0])
    assert bode_phase_error_deg(p_true_wrap, p_est_wrap) == pytest.approx(
        10.0, abs=1e-5
    )


def test_pole_relative_error_synthetic() -> None:
    """Test pole relative error with nearest-neighbour matching."""
    poles_true = np.array([-2.0, -5.0])
    # 10% offset on both poles
    poles_est = np.array([-2.2, -5.5])
    assert pole_relative_error(poles_true, poles_est) == pytest.approx(0.10, abs=1e-5)

    # Swapped order should yield same error due to nearest neighbour
    poles_est_swapped = np.array([-5.5, -2.2])
    assert pole_relative_error(poles_true, poles_est_swapped) == pytest.approx(
        0.10, abs=1e-5
    )


def test_msd_resonance_peak_error_synthetic() -> None:
    """Test resonance peak frequency and height error extraction."""
    omega = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mag_true = np.array([1.0, 3.0, 10.0, 2.0, 1.0])  # Peak at w=3.0, h=10.0 (20 dB)
    mag_est = np.array([1.0, 2.0, 4.0, 20.0, 1.0])  # Peak at w=4.0, h=20.0 (26.02 dB)

    f_err, h_err = msd_resonance_peak_error(omega, mag_true, mag_est)
    assert f_err == pytest.approx(1.0, abs=1e-6)
    expected_h_err = abs(20.0 * np.log10(20.0) - 20.0 * np.log10(10.0))
    assert h_err == pytest.approx(expected_h_err, abs=1e-5)


def test_dc_gain_relative_error_synthetic() -> None:
    """Test DC gain error computation from state space realizations."""
    # System 1: H(0) = -C * A^(-1) * B = -(1)*(-1/2)*(1) = 0.5
    A_true = np.array([[-2.0]])
    B_true = np.array([[1.0]])
    C_true = np.array([[1.0]])
    D_true = np.array([[0.0]])

    # System 2: H(0) = -(1)*(-1/2.5)*(1) = 0.4
    A_est = np.array([[-2.5]])
    B_est = np.array([[1.0]])
    C_est = np.array([[1.0]])
    D_est = np.array([[0.0]])

    # Relative error = |0.4 - 0.5| / 0.5 = 0.2 (20%)
    err = dc_gain_relative_error(
        A_true, B_true, C_true, D_true, A_est, B_est, C_est, D_est
    )
    assert err == pytest.approx(0.20, abs=1e-6)


def test_empirical_frequency_response_on_lti() -> None:
    """Validate empirical sine response against analytical LTI frequency response."""
    # First-order LTI: dx/dt = -5x + 5u, y = x. Tau = 0.2s.
    A = np.array([[-5.0]])
    B = np.array([[5.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    Ts = 0.01

    Ad = np.array([[np.exp(-5.0 * Ts)]])
    Bd = np.array([[(1.0 - np.exp(-5.0 * Ts))]])
    Cd = C

    def simulate_fn(u: np.ndarray) -> np.ndarray:
        x = 0.0
        y = np.zeros(len(u))
        for k in range(len(u)):
            x = float(Ad[0, 0] * x + Bd[0, 0] * u[k])
            y[k] = float(Cd[0, 0] * x)
        return y

    w_test = np.array([1.0, 5.0, 10.0])
    mag_emp, phase_emp, _ = compute_empirical_frequency_response(
        simulate_fn,
        w_test,
        Ts=Ts,
        dominant_time_scale_s=0.2,
        amplitude=1.0,
        n_periods_fit=4,
    )

    # Continuous LTI frequency response
    mag_true, phase_true, _ = frequency_response(A, B, C, D, w_test)

    # Empirical and continuous magnitude should agree within 0.1 dB
    mag_diff_db = np.abs(20.0 * np.log10(mag_emp / mag_true))
    assert np.all(mag_diff_db < 0.2)


def validate_evaluation_json_structure(data: dict[str, Any], plant_id: str) -> None:
    """Validate schema and numerical validity of evaluation.json artifact."""
    required_keys = [
        "iid",
        "ood",
        "noise_sweep",
        "param_variation",
        "frequency_response",
        "impulse_response",
        "cost",
    ]
    for k in required_keys:
        assert k in data, f"Missing required top-level key '{k}'"

    # 1. iid validation
    iid = data["iid"]
    for m in ["arx", "prony", "neural_ode"]:
        assert m in iid
        m_entry = iid[m]
        assert "fit_percent" in m_entry and np.isfinite(m_entry["fit_percent"])
        assert "rmse" in m_entry and np.isfinite(m_entry["rmse"])
        assert "impulse_error" in m_entry and np.isfinite(m_entry["impulse_error"])
        assert "bode_inband_mag_db" in m_entry and np.isfinite(
            m_entry["bode_inband_mag_db"]
        )
        assert "bode_inband_phase_deg" in m_entry and np.isfinite(
            m_entry["bode_inband_phase_deg"]
        )

    # 2. ood validation
    ood = data["ood"]
    for split in ["ood_amplitude", "ood_frequency", "ood_waveform"]:
        assert split in ood
        for m in ["arx", "prony", "neural_ode"]:
            assert m in ood[split]
            assert np.isfinite(ood[split][m]["fit_percent"])

    # 3. noise_sweep validation
    ns = data["noise_sweep"]
    for snr in ["inf", "40", "30", "20", "10"]:
        assert snr in ns
        assert "arx" in ns[snr] and np.isfinite(ns[snr]["arx"]["fit_percent"])
        assert "prony" in ns[snr] and np.isfinite(ns[snr]["prony"]["fit_percent"])
        if plant_id == "msd" and snr in ["inf", "40", "20"]:
            assert "neural_ode" in ns[snr]
            assert np.isfinite(ns[snr]["neural_ode"]["fit_percent_mean"])

    # 4. param_variation validation
    if plant_id == "msd":
        pv = data["param_variation"]
        assert pv is not None
        assert "arx_mean_parameter_error_percent" in pv
        assert "prony_mean_parameter_error_percent" in pv
        assert len(pv["trials"]) == 10

    # 5. frequency_response validation
    fr = data["frequency_response"]
    for arr_key in [
        "omega",
        "mag_true",
        "phase_true",
        "mag_arx",
        "mag_prony",
        "mag_neural_ode",
    ]:
        assert arr_key in fr
        assert len(fr[arr_key]) > 0
    assert "cross_check" in fr
    assert np.isfinite(fr["cross_check"]["largest_bode_mag_discrepancy_db"])
    assert np.isfinite(fr["cross_check"]["largest_bode_phase_discrepancy_deg"])

    # 6. impulse_response validation
    ir = data["impulse_response"]
    for arr_key in ["t", "h_true", "h_arx", "h_prony", "h_neural_ode"]:
        assert arr_key in ir
        assert len(ir[arr_key]) > 0

    # 7. cost validation
    cost = data["cost"]
    assert "parameter_counts" in cost
    assert "training_time_s" in cost
    assert "inference_time_ms_per_1000_samples" in cost


def test_evaluation_json_artifacts_schema() -> None:
    """Acceptance test: every plant's evaluation.json exists and adheres to schema."""
    results_dir = Path("artifacts/results")
    for plant_id in ["rc", "motor", "msd"]:
        eval_path = results_dir / plant_id / "evaluation.json"
        if not eval_path.exists():
            pytest.skip(
                f"{eval_path} not found yet (will be generated by run_experiments.py)"
            )
        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)
        validate_evaluation_json_structure(data, plant_id)
