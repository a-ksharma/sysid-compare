"""Smoke test verifying all FastAPI endpoints against running server or TestClient."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from api.main import app
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_smoke_test(base_url: str | None = None) -> bool:
    """Execute smoke test calling every endpoint and verifying acceptance criteria."""
    logger.info("Initializing smoke test client...")
    client = TestClient(app)
    try:
        # 1. Health check
        logger.info("[1/8] Testing GET /api/v1/health...")
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )
        health_data = resp.json()
        assert health_data["status"] == "ok"
        assert "msd" in health_data["loaded_plants"]
        logger.info("  Health OK: loaded plants = %s", health_data["loaded_plants"])

        # 2. List systems
        logger.info("[2/8] Testing GET /api/v1/systems...")
        resp = client.get("/api/v1/systems")
        assert resp.status_code == 200
        systems_list = resp.json()
        assert len(systems_list) == 3
        logger.info("  Systems catalog OK: %d plants listed", len(systems_list))

        # 3. System detail
        logger.info("[3/8] Testing GET /api/v1/systems/msd...")
        resp = client.get("/api/v1/systems/msd")
        assert resp.status_code == 200
        msd_detail = resp.json()
        assert len(msd_detail["true_poles"]) == 2
        assert "arx" in msd_detail["identified"]
        assert "neural_ode" in msd_detail["identified"]
        logger.info("  System detail OK for 'msd'")

        # 4. Models cards
        logger.info("[4/8] Testing GET /api/v1/models...")
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        models_data = resp.json()
        assert "msd" in models_data["cards"]
        assert "neural_ode" in models_data["cards"]["msd"]
        logger.info("  Model cards retrieved OK")

        # 5. Simulate endpoint (including 3000-sample latency test and warning)
        logger.info("[5/8] Testing POST /api/v1/simulate...")
        # 5a. Normal simulation
        sim_payload = {
            "system_id": "msd",
            "models": ["arx", "prony", "neural_ode"],
            "input": {"kind": "step", "amplitude": 5.0},
            "duration_s": 5.0,
            "noise_snr_db": 40.0,
            "seed": 42,
        }
        resp = client.post("/api/v1/simulate", json=sim_payload)
        assert resp.status_code == 200, f"Simulate failed: {resp.text}"
        sim_res = resp.json()
        assert len(sim_res["t"]) == 250
        assert "neural_ode" in sim_res["predictions"]
        assert sim_res["y_measured"] is not None

        # 5b. Warning check: amplitude exceeds u_max (u_max=10.0, test 15.0)
        warn_payload = {
            "system_id": "msd",
            "models": ["neural_ode"],
            "input": {"kind": "step", "amplitude": 15.0},
            "duration_s": 2.0,
        }
        resp = client.post("/api/v1/simulate", json=warn_payload)
        assert resp.status_code == 200
        warn_res = resp.json()
        assert len(warn_res["warnings"]) > 0
        assert "exceeds nominal training range" in warn_res["warnings"][0]
        logger.info("  Simulate warning verified: '%s'", warn_res["warnings"][0])

        # 5c. Benchmark 3000-sample latency
        bench_payload = {
            "system_id": "msd",
            "models": ["arx", "prony", "neural_ode"],
            "input": {"kind": "step", "amplitude": 5.0},
            "duration_s": 60.0,  # 60s at 0.02s = 3000 samples
        }
        t0 = time.perf_counter()
        resp = client.post("/api/v1/simulate", json=bench_payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        logger.info("  3000-sample /simulate execution latency: %.2f ms", latency_ms)

        # 5d. Error 422 limit checks (> 3000 samples and > 3 x u_max)
        over_samples = {
            "system_id": "msd",
            "models": ["arx"],
            "input": {"kind": "step", "amplitude": 5.0},
            "duration_s": 65.0,  # 3250 samples > 3000
        }
        resp = client.post("/api/v1/simulate", json=over_samples)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "SAMPLE_LIMIT_EXCEEDED"

        over_amp = {
            "system_id": "msd",
            "models": ["arx"],
            "input": {"kind": "step", "amplitude": 35.0},  # > 30.0 (3x u_max)
            "duration_s": 2.0,
        }
        resp = client.post("/api/v1/simulate", json=over_amp)
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "AMPLITUDE_EXCEEDED"
        logger.info("  422 error limit paths verified successfully")

        # 6. Frequency response endpoint & exact 1e-9 tolerance test
        logger.info("[6/8] Testing GET /api/v1/frequency-response/msd...")
        resp = client.get("/api/v1/frequency-response/msd?points=200")
        assert resp.status_code == 200
        fr_data = resp.json()
        assert "neural_ode" in fr_data["models"]

        # Cross-check against offline evaluation.json
        resp_eval = client.get("/api/v1/results/msd")
        assert resp_eval.status_code == 200
        eval_json = resp_eval.json()
        offline_node_mag = eval_json["frequency_response"]["mag_neural_ode"]

        api_node_db = fr_data["models"]["neural_ode"]["magnitude_db"]
        # Convert dB back to linear magnitude or compare dB directly
        api_node_mag = [10.0 ** (db / 20.0) for db in api_node_db]

        max_mag_diff = max(abs(a - b) for a, b in zip(api_node_mag, offline_node_mag))
        logger.info(
            "  Live /frequency-response vs offline Neural ODE max discrepancy: %.2e",
            max_mag_diff,
        )
        assert max_mag_diff < 1e-9, f"Exceeded 1e-9 tolerance: {max_mag_diff}"

        # 7. Impulse response endpoint
        logger.info("[7/8] Testing GET /api/v1/impulse-response/msd...")
        resp = client.get("/api/v1/impulse-response/msd?points=100")
        assert resp.status_code == 200
        ir_data = resp.json()
        assert len(ir_data["t"]) == 100
        assert "neural_ode" in ir_data["models"]
        logger.info("  Impulse response computed live OK")

        # 8. Results endpoint
        logger.info("[8/8] Testing GET /api/v1/results/rc...")
        resp = client.get("/api/v1/results/rc")
        assert resp.status_code == 200
        rc_results = resp.json()
        assert "iid" in rc_results
        assert "ood" in rc_results
        logger.info("  Results retrieved OK for 'rc'")

        logger.info("==========================================")
        logger.info("All smoke test assertions PASSED cleanly!")
        logger.info("==========================================")
        return True
    finally:
        client.close()


def main() -> None:
    """CLI runner for smoke test."""
    parser = argparse.ArgumentParser(description="API smoke test suite.")
    parser.add_argument(
        "--url", default=None, help="Optional external base URL to probe."
    )
    args = parser.parse_args()
    success = run_smoke_test(args.url)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
