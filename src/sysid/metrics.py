"""Evaluation metrics for system identification.

All metrics are implemented as pure mathematical functions on NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the normalized goodness-of-fit in percent (spec 8.1).

    Formula:
        fit = 100 * (1 - ||y - y_hat||_2 / ||y - mean(y)||_2)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth output array.
    y_pred : np.ndarray
        Model predicted output array.

    Returns
    -------
    float
        Fit percentage.
    """
    if not np.all(np.isfinite(y_pred)):
        return -100.0
    diff_norm = float(np.linalg.norm(y_true - y_pred))
    denom = float(np.linalg.norm(y_true - np.mean(y_true)))
    if denom < 1e-12:
        return 0.0
    return float(100.0 * (1.0 - diff_norm / denom))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root-mean-squared error in physical units.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth output array.
    y_pred : np.ndarray
        Model predicted output array.

    Returns
    -------
    float
        RMSE in physical units.
    """
    if not np.all(np.isfinite(y_pred)):
        return 1e6
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def impulse_relative_error(h_true: np.ndarray, h_est: np.ndarray) -> float:
    """Calculate relative L2 error of the impulse response (spec 8.1).

    Formula:
        ||h - h_hat||_2 / ||h||_2

    Parameters
    ----------
    h_true : np.ndarray
        True continuous impulse response samples.
    h_est : np.ndarray
        Estimated impulse response samples.

    Returns
    -------
    float
        Relative L2 error.
    """
    h_norm = float(np.linalg.norm(h_true))
    if h_norm < 1e-12:
        return 0.0
    return float(np.linalg.norm(h_true - h_est) / h_norm)


def bode_magnitude_error_db(mag_true: np.ndarray, mag_est: np.ndarray) -> float:
    """Calculate mean absolute magnitude error in decibels (spec 8.1).

    Formula:
        mean(|20 * log10(mag_est / mag_true)|)

    Parameters
    ----------
    mag_true : np.ndarray
        True linear magnitude values.
    mag_est : np.ndarray
        Estimated linear magnitude values.

    Returns
    -------
    float
        Mean absolute error in dB.
    """
    db_true = 20.0 * np.log10(np.maximum(mag_true, 1e-12))
    db_est = 20.0 * np.log10(np.maximum(mag_est, 1e-12))
    return float(np.mean(np.abs(db_est - db_true)))


def bode_phase_error_deg(
    phase_true_deg: np.ndarray, phase_est_deg: np.ndarray
) -> float:
    """Calculate mean absolute wrapped phase error in degrees (spec 8.1).

    Parameters
    ----------
    phase_true_deg : np.ndarray
        True phase in degrees.
    phase_est_deg : np.ndarray
        Estimated phase in degrees.

    Returns
    -------
    float
        Mean absolute wrapped phase difference in degrees.
    """
    diff = (phase_est_deg - phase_true_deg + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(diff)))


def pole_relative_error(poles_true: np.ndarray, poles_est: np.ndarray) -> float:
    """Calculate relative pole distance using nearest-neighbor matching (spec 8.1).

    Parameters
    ----------
    poles_true : np.ndarray
        Array of true continuous poles.
    poles_est : np.ndarray
        Array of estimated continuous poles.

    Returns
    -------
    float
        Mean relative pole error.
    """
    n = len(poles_true)
    if n == 0 or len(poles_est) == 0:
        return 0.0

    # Nearest neighbor matching
    errors: list[float] = []
    available_est = list(poles_est)

    for p_t in poles_true:
        p_t_norm = max(float(abs(p_t)), 1e-6)
        # Find closest pole in available_est
        best_idx = int(np.argmin([abs(p_t - p_e) for p_e in available_est]))
        best_e = available_est.pop(best_idx)
        errors.append(float(abs(p_t - best_e) / p_t_norm))
        if not available_est:
            available_est = list(poles_est)

    return float(np.mean(errors))


def msd_resonance_peak_error(
    omega: np.ndarray,
    mag_true: np.ndarray,
    mag_est: np.ndarray,
) -> tuple[float, float]:
    """Calculate peak frequency and height discrepancy for resonant plants (spec 8.1).

    Parameters
    ----------
    omega : np.ndarray
        Frequency grid in rad/s.
    mag_true : np.ndarray
        True linear magnitude.
    mag_est : np.ndarray
        Estimated linear magnitude.

    Returns
    -------
    tuple[float, float]
        (peak_freq_error_rad_s, peak_height_error_db)
    """
    idx_true = int(np.argmax(mag_true))
    idx_est = int(np.argmax(mag_est))

    w_peak_true = float(omega[idx_true])
    w_peak_est = float(omega[idx_est])
    freq_err = abs(w_peak_est - w_peak_true)

    h_peak_true_db = float(20.0 * np.log10(max(mag_true[idx_true], 1e-12)))
    h_peak_est_db = float(20.0 * np.log10(max(mag_est[idx_est], 1e-12)))
    height_err = abs(h_peak_est_db - h_peak_true_db)

    return freq_err, height_err


def dc_gain_relative_error(
    A_true: np.ndarray,
    B_true: np.ndarray,
    C_true: np.ndarray,
    D_true: np.ndarray,
    A_est: np.ndarray,
    B_est: np.ndarray,
    C_est: np.ndarray,
    D_est: np.ndarray,
) -> float:
    """Calculate DC gain relative error |H_est(0) - H_true(0)| / |H_true(0)|.

    Formula:
        H(0) = -C * A^(-1) * B + D

    Returns
    -------
    float
        Relative DC gain error.
    """
    h0_true = float((-C_true @ np.linalg.solve(A_true, B_true) + D_true)[0, 0])
    h0_est = float((-C_est @ np.linalg.solve(A_est, B_est) + D_est)[0, 0])

    denom = max(abs(h0_true), 1e-12)
    return float(abs(h0_est - h0_true) / denom)
