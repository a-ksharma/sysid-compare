"""Pytest test suite for FastAPI service endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_endpoint() -> None:
    """GET /api/v1/health returns 200 with loaded plants and models."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert sorted(data["loaded_plants"]) == ["motor", "msd", "rc"]
    assert "msd" in data["loaded_models"]
    assert "neural_ode" in data["loaded_models"]["msd"]


def test_systems_catalog() -> None:
    """GET /api/v1/systems returns catalog of 3 physical plants."""
    resp = client.get("/api/v1/systems")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    plant_ids = [p["id"] for p in data]
    assert "rc" in plant_ids
    assert "motor" in plant_ids
    assert "msd" in plant_ids


def test_system_detail_found_and_not_found() -> None:
    """GET /api/v1/systems/{id} returns plant detail, and 404 for unknown."""
    # Existing plant
    resp = client.get("/api/v1/systems/msd")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "msd"
    assert data["order"] == 2
    assert len(data["true_poles"]) == 2
    assert "arx" in data["identified"]
    assert "neural_ode" in data["identified"]

    # Unknown plant returns 404 with error schema
    err_resp = client.get("/api/v1/systems/unknown_plant")
    assert err_resp.status_code == 404
    err_data = err_resp.json()
    assert "error" in err_data
    assert err_data["error"]["code"] == "PLANT_NOT_FOUND"


def test_models_catalog_and_detail() -> None:
    """GET /api/v1/models returns model cards."""
    # All models
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "cards" in data
    assert "msd" in data["cards"]
    assert "neural_ode" in data["cards"]["msd"]
    assert "classical" in data["cards"]["msd"]

    # Specific model card
    node_resp = client.get("/api/v1/models/msd/neural_ode")
    assert node_resp.status_code == 200
    node_card = node_resp.json()
    assert node_card["plant_id"] == "msd"
    assert node_card["hidden_dim"] == 64

    # Unknown model kind returns 404
    err_resp = client.get("/api/v1/models/msd/unknown_model")
    assert err_resp.status_code == 404
    assert err_resp.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_simulate_endpoint() -> None:
    """POST /api/v1/simulate executes simulation and returns predictions and metrics."""
    req = {
        "system_id": "msd",
        "models": ["arx", "prony", "neural_ode"],
        "input": {
            "kind": "step",
            "amplitude": 5.0,
            "step_time_s": 0.5,
        },
        "duration_s": 2.0,
        "noise_snr_db": 40.0,
        "seed": 42,
    }
    resp = client.post("/api/v1/simulate", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["t"]) == 100
    assert len(data["u"]) == 100
    assert len(data["y_true"]) == 100
    assert data["y_measured"] is not None
    assert "arx" in data["predictions"]
    assert "prony" in data["predictions"]
    assert "neural_ode" in data["predictions"]

    node_pred = data["predictions"]["neural_ode"]
    assert node_pred["fit_percent"] > 90.0
    assert node_pred["rmse"] < 0.1
    assert data["warnings"] == []


def test_simulate_amplitude_warning() -> None:
    """POST /api/v1/simulate returns warning when amplitude exceeds nominal u_max."""
    # Nominal u_max for MSD is 10.0 N; 15.0 N is within 3x limit but exceeds nominal.
    req = {
        "system_id": "msd",
        "models": ["neural_ode"],
        "input": {
            "kind": "step",
            "amplitude": 15.0,
            "step_time_s": 0.2,
        },
        "duration_s": 1.0,
    }
    resp = client.post("/api/v1/simulate", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["warnings"]) >= 1
    assert "exceeds nominal training range" in data["warnings"][0]


def test_simulate_validation_errors() -> None:
    """POST /api/v1/simulate rejects limits (>3000 samples, >3*u_max) with 422."""
    # Exceed sample limit (duration 70s at Ts=0.02s is 3500 samples > 3000)
    req_samples = {
        "system_id": "msd",
        "models": ["neural_ode"],
        "input": {"kind": "step", "amplitude": 1.0, "step_time_s": 0.1},
        "duration_s": 70.0,
    }
    resp_s = client.post("/api/v1/simulate", json=req_samples)
    assert resp_s.status_code == 422
    err_s = resp_s.json()
    assert err_s["error"]["code"] == "SAMPLE_LIMIT_EXCEEDED"

    # Exceed amplitude limit (40 N > 3 * 10 N = 30 N)
    req_amp = {
        "system_id": "msd",
        "models": ["neural_ode"],
        "input": {"kind": "step", "amplitude": 40.0, "step_time_s": 0.1},
        "duration_s": 1.0,
    }
    resp_a = client.post("/api/v1/simulate", json=req_amp)
    assert resp_a.status_code == 422
    err_a = resp_a.json()
    assert err_a["error"]["code"] == "AMPLITUDE_EXCEEDED"


def test_frequency_response_and_neural_ode_agreement() -> None:
    """GET /api/v1/frequency-response tests Bode curves and <1e-9 offline agreement."""
    resp = client.get(
        "/api/v1/frequency-response/msd?omega_min=0.1&omega_max=60.0&points=200"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "true_system" in data
    assert "neural_ode" in data["models"]

    # Verify invalid omega range returns 422
    err_resp = client.get(
        "/api/v1/frequency-response/msd?omega_min=50.0&omega_max=10.0"
    )
    assert err_resp.status_code == 422
    assert err_resp.json()["error"]["code"] == "INVALID_FREQUENCY_RANGE"

    # Compare Neural ODE frequency response with offline evaluation.json
    eval_path = Path("artifacts/results/msd/evaluation.json")
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)
        offline_e4 = eval_data.get("frequency_response", {})
        if "mag_neural_ode" in offline_e4:
            offline_mag_db = 20.0 * np.log10(np.array(offline_e4["mag_neural_ode"]))
            live_mag_db = np.array(data["models"]["neural_ode"]["magnitude_db"])
            diff = np.max(np.abs(live_mag_db - offline_mag_db))
            assert diff < 1e-9, f"Discrepancy {diff} exceeds 1e-9 tolerance"


def test_impulse_response_endpoint() -> None:
    """GET /api/v1/impulse-response returns impulse response trajectories."""
    resp = client.get("/api/v1/impulse-response/msd?points=100")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["t"]) == 100
    assert len(data["true_system"]) == 100
    assert "neural_ode" in data["models"]
    assert len(data["models"]["neural_ode"]) == 100


def test_results_endpoint() -> None:
    """GET /api/v1/results/{id} returns evaluation results."""
    resp = client.get("/api/v1/results/rc")
    assert resp.status_code == 200
    data = resp.json()
    assert "iid" in data
    assert "noise_sweep" in data
    assert "ood" in data
    assert "cost" in data
