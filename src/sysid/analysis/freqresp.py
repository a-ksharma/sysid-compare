"""Empirical frequency response measurement protocol via sine tests."""

from __future__ import annotations

from typing import Callable

import numpy as np


def compute_empirical_frequency_response(
    simulate_fn: Callable[[np.ndarray], np.ndarray],
    omega_grid: np.ndarray,
    Ts: float,
    dominant_time_scale_s: float,
    amplitude: float = 0.05,
    n_periods_fit: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure empirical frequency response Gd(e^(j*w*Ts)) via sine tests (spec 8.4).

    Drives the model with sampled sines, discards >= 8 dominant time constants,
    and fits a*sin + b*cos by least squares over an integer number of periods.

    Parameters
    ----------
    simulate_fn : Callable[[np.ndarray], np.ndarray]
        Simulation function mapping 1D input array to 1D output array.
    omega_grid : np.ndarray
        Frequencies in rad/s to test.
    Ts : float
        Sampling period in seconds.
    dominant_time_scale_s : float
        Dominant time constant for transient settling.
    amplitude : float, optional
        Input sinusoidal amplitude, by default 0.05.
    n_periods_fit : int, optional
        Number of steady-state periods to fit, by default 4.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (mag, phase_deg, complex_resp)
    """
    n_points = len(omega_grid)
    mag_out = np.zeros(n_points, dtype=np.float64)
    phase_out = np.zeros(n_points, dtype=np.float64)
    complex_out = np.zeros(n_points, dtype=np.complex128)

    for i, w in enumerate(omega_grid):
        period_s = 2.0 * np.pi / w
        # Transient settling: at least 8 time constants and 2 periods
        t_trans = max(8.0 * dominant_time_scale_s, 2.0 * period_s)
        t_fit = n_periods_fit * period_s
        t_total = t_trans + t_fit

        n_samples = int(np.ceil(t_total / Ts))
        t = np.arange(n_samples, dtype=np.float64) * Ts
        u = amplitude * np.sin(w * t)

        y = simulate_fn(u)

        # Discard transient
        ss_mask = t >= t_trans
        t_ss = t[ss_mask]
        y_ss = y[ss_mask]

        # Regressors: y_ss approx a * sin(w*t) + b * cos(w*t)
        sin_basis = np.sin(w * t_ss)
        cos_basis = np.cos(w * t_ss)
        phi_mat = np.column_stack([sin_basis, cos_basis])

        coeffs, _, _, _ = np.linalg.lstsq(phi_mat, y_ss, rcond=None)
        a_coef, b_coef = float(coeffs[0]), float(coeffs[1])

        # Amplitude R and phase phi: a*sin + b*cos = R*sin(w*t + phi)
        # R = sqrt(a^2 + b^2), phi = atan2(b, a)
        r_amp = float(np.hypot(a_coef, b_coef))
        phi_rad = float(np.arctan2(b_coef, a_coef))

        gain = r_amp / amplitude
        mag_out[i] = gain
        phase_deg = float(np.rad2deg(phi_rad))
        phase_out[i] = phase_deg
        complex_out[i] = gain * np.exp(1j * phi_rad)

    # Unwrap phase
    phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phase_out)))
    return mag_out, phase_unwrapped, complex_out
