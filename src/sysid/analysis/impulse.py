"""Impulse response analysis and cross-checks against nonlinear small-pulse tests."""

from __future__ import annotations

import numpy as np

from sysid.lti import impulse_response
from sysid.neural.numpy_infer import NumpyNeuralODE


def compute_linearized_impulse_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Compute continuous impulse response h(t) = C * expm(A * t) * B."""
    return impulse_response(A, B, C, t_grid)


def check_nonlinear_pulse_cross_check(
    model: NumpyNeuralODE,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    horizon_s: float,
    pulse_amplitude_n: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compare linearized and nonlinear small pulse responses (spec 8.1).

    Parameters
    ----------
    model : NumpyNeuralODE
        Trained model instance.
    A : np.ndarray
        Continuous state matrix in physical units.
    B : np.ndarray
        Continuous input matrix in physical units.
    C : np.ndarray
        Continuous output matrix in physical units.
    horizon_s : float
        Evaluation horizon in seconds.
    pulse_amplitude_n : float, optional
        Normalized pulse height, by default 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        (t_grid, h_linear, h_nonlinear, rel_discrepancy)
    """
    Ts = model.Ts
    n_samples = int(round(horizon_s / Ts))
    t_grid = np.arange(n_samples, dtype=np.float64) * Ts

    # 1. Linearized continuous impulse response
    h_linear = compute_linearized_impulse_response(A, B, C, t_grid)

    # 2. Nonlinear small pulse response
    u_norm = np.zeros(n_samples, dtype=np.float32)
    u_norm[0] = pulse_amplitude_n

    y_norm = model.simulate(u_norm)

    # Convert to physical units:
    # Scale by (y_scale / u_scale) and divide by pulse area (pulse_amplitude_n * Ts)
    pulse_area = pulse_amplitude_n * Ts
    h_nonlinear = (y_norm / pulse_area) * (model.y_scale / model.u_scale)

    # 3. Discrepancy metric
    lin_norm = float(np.linalg.norm(h_linear))
    if lin_norm < 1e-12:
        rel_diff = 0.0
    else:
        rel_diff = float(np.linalg.norm(h_nonlinear - h_linear) / lin_norm)

    return t_grid, h_linear, h_nonlinear, rel_diff
