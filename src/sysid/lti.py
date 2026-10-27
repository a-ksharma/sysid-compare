"""Linear time-invariant (LTI) system analysis utilities.

Pure NumPy and SciPy implementations with zero python-control dependency,
ensuring lightweight and portable execution in the API.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def frequency_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the continuous-time LTI frequency response H(j*omega).

    Computes:
        H(j * omega) = C * (j * omega * I - A)^(-1) * B + D

    Parameters
    ----------
    A : np.ndarray
        State transition matrix of shape (n_states, n_states).
    B : np.ndarray
        Input matrix of shape (n_states, 1).
    C : np.ndarray
        Output matrix of shape (1, n_states).
    D : np.ndarray
        Feedthrough matrix of shape (1, 1).
    omega : np.ndarray
        Angular frequencies in rad/s of shape (n_points,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (mag, phase_deg, complex_resp)
        - mag : Linear magnitude |H(j*omega)| of shape (n_points,).
        - phase_deg : Unwrapped phase angle in degrees of shape (n_points,).
        - complex_resp : Complex transfer function values of shape (n_points,).
    """
    n_states = A.shape[0]
    eye_n = np.eye(n_states, dtype=np.complex128)
    n_points = len(omega)
    complex_resp = np.zeros(n_points, dtype=np.complex128)

    B_c = B.astype(np.complex128)
    C_c = C.astype(np.complex128)
    d_scalar = float(D[0, 0])

    for i, w in enumerate(omega):
        mat = 1j * w * eye_n - A
        # Solve (j*w*I - A) * x = B for x to avoid explicit matrix inversion
        x = np.linalg.solve(mat, B_c)
        val = C_c @ x + d_scalar
        complex_resp[i] = val[0, 0]

    mag = np.abs(complex_resp)
    phase_rad = np.unwrap(np.angle(complex_resp))
    phase_deg = np.rad2deg(phase_rad)

    return mag, phase_deg, complex_resp


def impulse_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Calculate the continuous-time LTI impulse response h(t).

    Computes:
        h(t) = C * expm(A * t) * B

    Parameters
    ----------
    A : np.ndarray
        State transition matrix of shape (n_states, n_states).
    B : np.ndarray
        Input matrix of shape (n_states, 1).
    C : np.ndarray
        Output matrix of shape (1, n_states).
    t : np.ndarray
        Time points in seconds of shape (n_points,), starting at t >= 0.

    Returns
    -------
    np.ndarray
        Impulse response h(t) of shape (n_points,).
    """
    n_points = len(t)
    h = np.zeros(n_points, dtype=np.float64)

    for i, t_val in enumerate(t):
        if t_val == 0.0:
            exp_at = np.eye(A.shape[0], dtype=np.float64)
        else:
            exp_at = scipy.linalg.expm(A * t_val)
        val = C @ exp_at @ B
        h[i] = float(val[0, 0])

    return h


def poles(A: np.ndarray) -> np.ndarray:
    """Calculate the continuous-time system poles (eigenvalues of A).

    Parameters
    ----------
    A : np.ndarray
        State transition matrix of shape (n_states, n_states).

    Returns
    -------
    np.ndarray
        Array of system poles sorted by real part descending.
    """
    eigvals = np.linalg.eigvals(A)
    # Sort for consistent ordering: real descending, then imag descending
    order = np.lexsort((-eigvals.imag, -eigvals.real))
    return eigvals[order]
