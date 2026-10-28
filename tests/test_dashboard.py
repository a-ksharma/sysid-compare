"""Tests for dashboard components, import constraints, and plotting utilities."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from dashboard.api_client import APIClient
from dashboard.plotting import (
    COLOR_MAP,
    plot_bode,
    plot_impulse_response,
    plot_noise_sweep_curve,
    plot_ood_bar_chart,
    plot_simulation_trajectories,
)


def test_dashboard_never_imports_sysid_or_torch() -> None:
    """Acceptance criterion: dashboard files must NEVER import sysid or torch."""
    dashboard_dir = Path("dashboard")
    py_files = list(dashboard_dir.rglob("*.py"))
    assert len(py_files) >= 5, (
        f"Expected at least 5 dashboard python files, found {len(py_files)}"
    )

    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("sysid"), (
                        f"Prohibited import 'sysid' found in {py_file}"
                    )
                    assert not alias.name.startswith("torch"), (
                        f"Prohibited import 'torch' found in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("sysid"), (
                        f"Prohibited import from 'sysid' found in {py_file}"
                    )
                    assert not node.module.startswith("torch"), (
                        f"Prohibited import from 'torch' found in {py_file}"
                    )


def test_dashboard_contains_no_emojis() -> None:
    """Project rule: zero emoji anywhere in the UI or codebase."""
    dashboard_dir = Path("dashboard")
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    for py_file in dashboard_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        matches = emoji_pattern.findall(content)
        assert not matches, f"Forbidden emoji {matches} found in {py_file}"


def test_okabe_ito_color_palette() -> None:
    """Verify exact Okabe-Ito hex codes match specification."""
    assert COLOR_MAP["truth"] == "#000000"
    assert COLOR_MAP["arx"] == "#0072B2"
    assert COLOR_MAP["prony"] == "#E69F00"
    assert COLOR_MAP["neural_ode"] == "#D55E00"


def test_plotting_simulation_trajectories() -> None:
    """Verify plot_simulation_trajectories builds a 3-panel figure."""
    t = [0.0, 0.02, 0.04, 0.06]
    u = [1.0, 1.0, 1.0, 1.0]
    y_true = [0.0, 0.1, 0.18, 0.25]
    y_meas = [0.01, 0.09, 0.19, 0.26]
    preds = {
        "arx": {"y": [0.0, 0.095, 0.175, 0.245], "fit_percent": 98.5},
        "neural_ode": {"y": [0.0, 0.102, 0.181, 0.252], "fit_percent": 99.1},
    }
    fig = plot_simulation_trajectories(
        t, u, y_true, y_meas, preds, "Force", "N", "Displacement", "m"
    )
    assert len(fig.data) >= 5


def test_plotting_bode_curves() -> None:
    """Verify plot_bode builds a 2-row figure with shaded training band."""
    omega = [0.1, 1.0, 10.0, 50.0]
    curves = {
        "truth": {
            "magnitude_db": [-12.0, -12.1, -15.0, -30.0],
            "phase_deg": [0.0, -5.0, -45.0, -85.0],
        },
        "neural_ode": {
            "magnitude_db": [-12.0, -12.1, -15.1, -30.2],
            "phase_deg": [0.0, -5.1, -45.2, -85.1],
        },
    }
    fig = plot_bode(omega, curves, training_band_rad_s=[0.1, 15.0])
    assert len(fig.data) == 4
    # Check shapes for shaded rect
    assert len(fig.layout.shapes) >= 2


def test_plotting_impulse_response() -> None:
    """Verify plot_impulse_response builds valid figure."""
    t = [0.0, 0.1, 0.2]
    true_h = [5.0, 3.0, 1.8]
    models_h = {"arx": [4.9, 2.9, 1.7]}
    fig = plot_impulse_response(t, true_h, models_h, "V")
    assert len(fig.data) == 2


def test_plotting_ood_and_noise_sweep() -> None:
    """Verify OOD bar chart and noise sweep curve figures."""
    ood_data = {
        "step_half": {
            "arx": {"fit_percent": 99.0},
            "prony": {"fit_percent": 99.0},
            "neural_ode": {"fit_percent": 98.5},
        },
        "chirp": {
            "arx": {"fit_percent": 95.0},
            "prony": {"fit_percent": 94.0},
            "neural_ode": {"fit_percent": 96.0},
        },
    }
    fig_ood = plot_ood_bar_chart(ood_data)
    assert len(fig_ood.data) == 3

    noise_data = {
        "inf": {
            "arx": {"fit_percent": 99.0},
            "prony": {"fit_percent": 99.0},
            "neural_ode": {"fit_percent": 98.0},
        },
        "40": {
            "arx": {"fit_percent": 90.0},
            "prony": {"fit_percent": 85.0},
            "neural_ode": {"fit_percent": 97.0},
        },
        "20": {
            "arx": {"fit_percent": 50.0},
            "prony": {"fit_percent": 30.0},
            "neural_ode": {"fit_percent": 92.0},
        },
        "10": {
            "arx": {"fit_percent": -10.0},
            "prony": {"fit_percent": -50.0},
            "neural_ode": {"fit_percent": 80.0},
        },
    }
    fig_noise = plot_noise_sweep_curve(noise_data)
    assert len(fig_noise.data) == 3


def test_api_client_wake_state_on_unreachable_server() -> None:
    """Verify APIClient gracefully handles unreachable server."""
    client = APIClient(base_url="http://127.0.0.1:59999")
    health = client.check_health(timeout_s=0.2)
    assert health is None

    success, msg = client.wake_service(max_attempts=1, interval_s=0.1)
    assert success is False
    assert "Unable to reach API" in msg
