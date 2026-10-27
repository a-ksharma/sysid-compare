"""Conversion utilities between discrete-time models and continuous-time models.

Implements controllable canonical form state-space construction from polynomials
and exact discrete-to-continuous (d2c) conversion using matrix logarithms.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.signal


def arx_to_discrete_ss(
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct controllable canonical form state space from ARX polynomials.

    Difference equation:
        y[k] + a_1 y[k-1] + ... + a_n y[k-n] = b_1 u[k-1] + ... + b_n u[k-n]

    Transfer function:
        G(z) = (b_1 z^(n-1) + ... + b_n) / (z^n + a_1 z^(n-1) + ... + a_n)

    Parameters
    ----------
    a_coeffs : np.ndarray
        Denominator AR coefficients [a_1, ..., a_n] of shape (n,).
    b_coeffs : np.ndarray
        Numerator X coefficients [b_1, ..., b_n] of shape (n,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Ad, Bd, Cd, Dd) discrete state-space matrices with Dd = 0.
    """
    n = len(a_coeffs)
    if n != len(b_coeffs):
        msg = f"AR and X orders must match: len(a)={n}, len(b)={len(b_coeffs)}"
        raise ValueError(msg)

    if n == 1:
        Ad = np.array([[-a_coeffs[0]]], dtype=np.float64)
        Bd = np.array([[1.0]], dtype=np.float64)
        Cd = np.array([[b_coeffs[0]]], dtype=np.float64)
        Dd = np.array([[0.0]], dtype=np.float64)
        return Ad, Bd, Cd, Dd

    # Controllable canonical form for order n >= 2:
    Ad = np.zeros((n, n), dtype=np.float64)
    # Upper sub-diagonal is identity
    for i in range(n - 1):
        Ad[i, i + 1] = 1.0
    # Bottom row holds negative AR coefficients in reverse order [-a_n, ..., -a_1]
    Ad[n - 1, :] = -a_coeffs[::-1]

    Bd = np.zeros((n, 1), dtype=np.float64)
    Bd[n - 1, 0] = 1.0

    Cd = np.zeros((1, n), dtype=np.float64)
    Cd[0, :] = b_coeffs[::-1]

    Dd = np.array([[0.0]], dtype=np.float64)

    return Ad, Bd, Cd, Dd


def discrete_to_continuous_ss(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Cd: np.ndarray,
    Dd: np.ndarray,
    Ts: float,
    tol_imag: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a discrete-time ZOH state space (Ad, Bd, Cd, Dd) to continuous time.

    Formulas:
        A = logm(Ad) / Ts
        B = (Ad - I)^(-1) * A * Bd
        C = Cd
        D = Dd

    Parameters
    ----------
    Ad : np.ndarray
        Discrete state matrix of shape (n, n).
    Bd : np.ndarray
        Discrete input matrix of shape (n, 1).
    Cd : np.ndarray
        Discrete output matrix of shape (1, n).
    Dd : np.ndarray
        Discrete feedthrough matrix of shape (1, 1).
    Ts : float
        Sampling interval in seconds.
    tol_imag : float, optional
        Tolerance for discarding machine-precision imaginary residue, by default 1e-10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (A, B, C, D) continuous state-space matrices.

    Raises
    ------
    ValueError
        If logm produces genuine complex matrix entries exceeding tol_imag.
    """
    n = Ad.shape[0]
    log_ad = scipy.linalg.logm(Ad)

    # Check for genuine complex components per Decision K15
    max_imag = float(np.max(np.abs(np.imag(log_ad))))
    if max_imag > tol_imag:
        msg = f"logm(Ad) produced complex matrix: imag {max_imag:.2e} > {tol_imag:.2e}"
        raise ValueError(msg)

    A_mat = np.real(log_ad) / Ts

    # B = (Ad - I)^(-1) * A * Bd
    ad_minus_eye = Ad - np.eye(n, dtype=np.float64)
    b_rhs = A_mat @ Bd
    B_mat = np.linalg.solve(ad_minus_eye, b_rhs)

    C_mat = Cd.copy()
    D_mat = Dd.copy()

    return A_mat, B_mat, C_mat, D_mat


def tf_from_ss(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute continuous transfer function polynomials from state-space matrices.

    Uses scipy.signal.ss2tf and normalizes to a monic denominator.

    Parameters
    ----------
    A : np.ndarray
        Continuous state matrix of shape (n, n).
    B : np.ndarray
        Continuous input matrix of shape (n, 1).
    C : np.ndarray
        Continuous output matrix of shape (1, n).
    D : np.ndarray
        Continuous feedthrough matrix of shape (1, 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (num, den) transfer function polynomial coefficients in descending powers of s.
    """
    num_raw, den_raw = scipy.signal.ss2tf(A, B, C, D)
    num = np.squeeze(num_raw)
    den = np.squeeze(den_raw)

    # Normalize by leading denominator coefficient
    den_lead = float(den[0])
    den_norm = den / den_lead
    num_norm = num / den_lead

    # Remove leading zeros in numerator if any
    first_nonzero = np.argmax(np.abs(num_norm) > 1e-12)
    if np.abs(num_norm[first_nonzero]) > 1e-12:
        num_clean = num_norm[first_nonzero:]
    else:
        num_clean = num_norm[-1:]

    return num_clean, den_norm
