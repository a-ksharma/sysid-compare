"""ARX (autoregressive with exogenous input) least-squares identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sysid.classical.convert import (
    arx_to_discrete_ss,
    discrete_to_continuous_ss,
    tf_from_ss,
)


@dataclass(frozen=True)
class ARXModel:
    """Estimated ARX model and its continuous-time representation.

    Parameters
    ----------
    order : int
        Polynomial order n.
    Ts : float
        Sampling period in seconds.
    a_coeffs : np.ndarray
        Fitted AR denominator coefficients [a_1, ..., a_n].
    b_coeffs : np.ndarray
        Fitted X numerator coefficients [b_1, ..., b_n].
    Ad : np.ndarray
        Discrete-time state matrix in controllable canonical form.
    Bd : np.ndarray
        Discrete-time input matrix.
    Cd : np.ndarray
        Discrete-time output matrix.
    Dd : np.ndarray
        Discrete-time feedthrough matrix (zero).
    A : np.ndarray
        Continuous-time state matrix.
    B : np.ndarray
        Continuous-time input matrix.
    C : np.ndarray
        Continuous-time output matrix.
    D : np.ndarray
        Continuous-time feedthrough matrix.
    num : np.ndarray
        Continuous transfer function numerator coefficients.
    den : np.ndarray
        Continuous transfer function denominator coefficients.
    poles : np.ndarray
        Continuous system poles (eigenvalues of A).
    discrete_poles : np.ndarray
        Discrete system poles (eigenvalues of Ad).
    """

    order: int
    Ts: float
    a_coeffs: np.ndarray
    b_coeffs: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    Cd: np.ndarray
    Dd: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    num: np.ndarray
    den: np.ndarray
    poles: np.ndarray
    discrete_poles: np.ndarray


def build_arx_regressors(
    u: np.ndarray,
    y: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct regressor matrix Phi and target vector Y for ARX least squares.

    Stack the regressors:
        [-y[k-1, ..., k-n], u[k-1, ..., k-n]]
    Pre-samples are treated as zero because sequences start at rest.

    Parameters
    ----------
    u : np.ndarray
        Input array of shape (n_sequences, n_samples) or (n_samples,).
    y : np.ndarray
        Output array of shape (n_sequences, n_samples) or (n_samples,).
    order : int
        System order n.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (Phi, Y) where Phi is (n_rows, 2*order) and Y is (n_rows,).
    """
    if u.ndim == 1:
        u_2d = u[np.newaxis, :]
        y_2d = y[np.newaxis, :]
    else:
        u_2d = u
        y_2d = y

    n_seqs, n_samples = u_2d.shape
    phi_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for s in range(n_seqs):
        u_s = u_2d[s]
        y_s = y_2d[s]

        # Slices for k = 1 to n_samples - 1 (since k=0 has only zero pre-samples)
        n_rows = n_samples - 1
        phi_seq = np.zeros((n_rows, 2 * order), dtype=np.float64)

        for i in range(1, order + 1):
            # Term -y[k-i]: for k in [1, n_samples-1]
            y_col = np.zeros(n_rows, dtype=np.float64)
            valid_start = max(0, i - 1)
            y_col[valid_start:] = -y_s[: n_rows - valid_start]
            phi_seq[:, i - 1] = y_col

            # Term u[k-i]: for k in [1, n_samples-1]
            u_col = np.zeros(n_rows, dtype=np.float64)
            u_col[valid_start:] = u_s[: n_rows - valid_start]
            phi_seq[:, order + i - 1] = u_col

        phi_list.append(phi_seq)
        y_list.append(y_s[1:])

    Phi = np.vstack(phi_list)
    Y = np.concatenate(y_list)
    return Phi, Y


def fit_arx(
    u: np.ndarray,
    y: np.ndarray,
    order: int,
    Ts: float,
) -> ARXModel:
    """Identify plant using ARX least squares with continuous-time conversion.

    Parameters
    ----------
    u : np.ndarray
        Input sequences array of shape (n_sequences, n_samples) or (n_samples,).
    y : np.ndarray
        Output sequences array of shape (n_sequences, n_samples) or (n_samples,).
    order : int
        Assumed system order (1 or 2).
    Ts : float
        Sampling period in seconds.

    Returns
    -------
    ARXModel
        Fitted model parameters and state-space realization.
    """
    Phi, Y = build_arx_regressors(u, y, order)

    theta, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)

    a_coeffs = theta[:order]
    b_coeffs = theta[order:]

    # Discrete state-space controllable canonical form
    Ad, Bd, Cd, Dd = arx_to_discrete_ss(a_coeffs, b_coeffs)

    # Continuous state-space conversion via matrix logarithm
    try:
        A, B, C, D = discrete_to_continuous_ss(Ad, Bd, Cd, Dd, Ts)
        num, den = tf_from_ss(A, B, C, D)
        c_poles = np.linalg.eigvals(A)
    except ValueError:
        # Decision K15: d2c conversion rejected if logm produces complex matrix
        A = np.full_like(Ad, np.nan)
        B = np.full_like(Bd, np.nan)
        C = np.full_like(Cd, np.nan)
        D = np.full_like(Dd, np.nan)
        num = np.array([np.nan])
        den = np.array([np.nan])
        c_poles = np.full(order, np.nan + 1j * np.nan)
    d_poles = np.linalg.eigvals(Ad)

    return ARXModel(
        order=order,
        Ts=Ts,
        a_coeffs=a_coeffs,
        b_coeffs=b_coeffs,
        Ad=Ad,
        Bd=Bd,
        Cd=Cd,
        Dd=Dd,
        A=A,
        B=B,
        C=C,
        D=D,
        num=num,
        den=den,
        poles=c_poles,
        discrete_poles=d_poles,
    )
