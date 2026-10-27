"""Prony's method for system identification from impulse experiment records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PronyModel:
    """Estimated Prony model and its continuous transfer function/state space.

    Parameters
    ----------
    order : int
        Assumed system order n.
    Ts : float
        Sampling period in seconds.
    discrete_poles : np.ndarray
        Roots of the discrete characteristic polynomial.
    continuous_poles : np.ndarray
        Continuous poles mapped via s_i = ln(z_i) / Ts.
    residues : np.ndarray
        Continuous impulse residues R_i.
    num : np.ndarray
        Continuous transfer function numerator polynomial coefficients.
    den : np.ndarray
        Continuous transfer function denominator polynomial coefficients.
    A : np.ndarray
        Continuous state matrix in companion form.
    B : np.ndarray
        Continuous input matrix.
    C : np.ndarray
        Continuous output matrix.
    D : np.ndarray
        Continuous feedthrough matrix.
    is_stable : bool
        Whether all continuous poles lie in the open left half-plane (Re(s) < 0).
    has_nyquist_artifact : bool
        Whether any pole landed on or near the Nyquist frequency (|Im(s)| approx pi/Ts).
    warning : str | None
        Diagnostic warning describing failure modes under noise, or None.
    """

    order: int
    Ts: float
    discrete_poles: np.ndarray
    continuous_poles: np.ndarray
    residues: np.ndarray
    num: np.ndarray
    den: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    is_stable: bool
    has_nyquist_artifact: bool
    warning: str | None


def fit_prony(
    h_record: np.ndarray,
    order: int,
    Ts: float,
) -> PronyModel:
    """Identify continuous transfer function from impulse record using Prony's method.

    The record h_record contains samples for k >= 0 (sample 0 is pulse interval).
    Prony analysis operates on the free decay samples k >= 1 per spec 5.2.

    Parameters
    ----------
    h_record : np.ndarray
        Scaled impulse experiment output record of shape (n_samples,).
    order : int
        System order n.
    Ts : float
        Sampling interval in seconds.

    Returns
    -------
    PronyModel
        Fitted poles, residues, transfer function, and state-space realization.
    """
    n_samples = len(h_record)
    if n_samples < 2 * order + 2:
        raise ValueError(
            f"Impulse record too short ({n_samples} samples) for order {order}."
        )

    # 1. Linear prediction on k >= 1
    # Equation: h[k] = -sum_{j=1..n} a_j h[k-j] for k >= n + 1
    # Lowest referenced sample is (n+1) - n = 1, so k=0 (pulse interval) is never read.
    k_start = order + 1
    rows = n_samples - k_start
    H_mat = np.zeros((rows, order), dtype=np.float64)
    h_target = np.zeros(rows, dtype=np.float64)

    for row_idx, k in enumerate(range(k_start, n_samples)):
        h_target[row_idx] = h_record[k]
        for j in range(1, order + 1):
            H_mat[row_idx, j - 1] = -h_record[k - j]

    a_coeffs, _, _, _ = np.linalg.lstsq(H_mat, h_target, rcond=None)

    # 2. Discrete poles z_i as roots of z^n + a_1 z^(n-1) + ... + a_n = 0
    poly_coeffs = np.concatenate([[1.0], a_coeffs])
    z_poles = np.roots(poly_coeffs)

    # 3. Residues r_i by least squares on the Vandermonde system h[k] = sum r_i z_i^k
    # Decision K16: index k from 1 to N-1
    k_indices = np.arange(1, n_samples, dtype=np.float64)
    V = np.zeros((len(k_indices), order), dtype=np.complex128)
    for i, z in enumerate(z_poles):
        V[:, i] = np.power(z, k_indices)

    h_decay = h_record[1:].astype(np.complex128)
    r_residues, _, _, _ = np.linalg.lstsq(V, h_decay, rcond=None)

    # 4. Continuous poles: s_i = ln(z_i) / Ts
    # Continuous residues: R_i = r_i * s_i * Ts / (1 - z_i^(-1))
    s_poles = np.log(z_poles) / Ts

    R_residues = np.zeros(order, dtype=np.complex128)
    for i in range(order):
        s_i = s_poles[i]
        z_i = z_poles[i]
        # Avoid division by zero if z_i is close to 1
        denom = 1.0 - (1.0 / z_i)
        if np.abs(denom) < 1e-12:
            R_residues[i] = r_residues[i]
        else:
            R_residues[i] = r_residues[i] * s_i * Ts / denom

    # 5. Continuous transfer function H(s) = sum R_i / (s - s_i)
    # Denominator den(s) = prod(s - s_i)
    den_poly_c = np.poly(s_poles)
    den = np.real(den_poly_c)

    # Numerator num(s) = sum_i R_i * prod_{j != i} (s - s_j)
    num_poly_c = np.zeros(order, dtype=np.complex128)
    for i in range(order):
        other_poles = np.delete(s_poles, i)
        poly_other = np.poly(other_poles)
        num_poly_c += R_residues[i] * poly_other

    num = np.real(num_poly_c)

    # Ensure monic denominator
    den_lead = float(den[0])
    den = den / den_lead
    num = num / den_lead

    # Remove leading zeros in numerator
    first_nonzero = np.argmax(np.abs(num) > 1e-12)
    if np.abs(num[first_nonzero]) > 1e-12:
        num = num[first_nonzero:]
    else:
        num = num[-1:]

    # Construct continuous state space in controllable companion form
    if order == 1:
        A = np.array([[-den[1]]], dtype=np.float64)
        B = np.array([[1.0]], dtype=np.float64)
        C = np.array([[num[0]]], dtype=np.float64)
        D = np.array([[0.0]], dtype=np.float64)
    else:
        A = np.zeros((order, order), dtype=np.float64)
        for i in range(order - 1):
            A[i, i + 1] = 1.0
        A[order - 1, :] = -den[1:][::-1]
        B = np.zeros((order, 1), dtype=np.float64)
        B[order - 1, 0] = 1.0
        C = np.zeros((1, order), dtype=np.float64)
        # Pad numerator to length order if needed
        num_padded = np.zeros(order, dtype=np.float64)
        num_padded[-len(num) :] = num
        C[0, :] = num_padded[::-1]
        D = np.array([[0.0]], dtype=np.float64)

    # Check stability and Nyquist branch cuts
    is_stable = bool(np.all(np.real(s_poles) < 0.0))
    nyquist_freq = np.pi / Ts
    has_nyquist_artifact = bool(
        np.any(np.abs(np.abs(np.imag(s_poles)) - nyquist_freq) < 1.0)
    )

    warnings: list[str] = []
    if not is_stable:
        warnings.append(
            f"Prony fit is unstable: max Re(s) = {float(np.max(np.real(s_poles))):.3f}"
        )
    if has_nyquist_artifact:
        warnings.append(f"Prony fit has Nyquist artifact near {nyquist_freq:.1f} rad/s")
    warning_str = "; ".join(warnings) if warnings else None

    return PronyModel(
        order=order,
        Ts=Ts,
        discrete_poles=z_poles,
        continuous_poles=s_poles,
        residues=R_residues,
        num=num,
        den=den,
        A=A,
        B=B,
        C=C,
        D=D,
        is_stable=is_stable,
        has_nyquist_artifact=has_nyquist_artifact,
        warning=warning_str,
    )
