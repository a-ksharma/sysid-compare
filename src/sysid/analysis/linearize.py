"""Linearization of Neural ODE vector fields around the rest state (origin).

Computes exact analytical Jacobians A = df/dx and B = df/du at (x=0, u=0),
rescaling to physical units according to spec section 7.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sysid.neural.numpy_infer import NumpyNeuralODE


def linearize_numpy_neural_ode(
    model: NumpyNeuralODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact analytical continuous state-space linearization at (x=0, u=0).

    Using the chain rule through tanh activations at equilibrium:
        A_n = W3 * diag(1 - a2^2) * W2 * diag(1 - a1^2) * W1_x
        B_n = W3 * diag(1 - a2^2) * W2 * diag(1 - a1^2) * W1_u
    Physical conversion (spec 7.5):
        A = A_n
        B = B_n / u_scale
        C = y_scale * C_n
        D = 0

    Parameters
    ----------
    model : NumpyNeuralODE
        Trained NumPy Neural ODE model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (A, B, C, D, poles) in physical units.
    """
    n_x = model.n_x

    # Equilibrium forward activations at zero input
    zero_in = np.zeros(n_x + 1, dtype=np.float32)
    z1 = model.W1 @ zero_in + model.b1
    a1 = np.tanh(z1)
    z2 = model.W2 @ a1 + model.b2
    a2 = np.tanh(z2)

    # Diagonal derivative matrices of tanh
    D1 = np.diag(1.0 - a1**2)
    D2 = np.diag(1.0 - a2**2)

    # Input weights split into state and input parts
    W1_x = model.W1[:, :n_x]
    W1_u = model.W1[:, n_x : n_x + 1]

    # Normalized Jacobians
    gain_chain = model.W3 @ D2 @ model.W2 @ D1
    A_n = gain_chain @ W1_x
    B_n = gain_chain @ W1_u
    C_n = model.C

    # Physical scaling (spec 7.5)
    A = A_n.astype(np.float64)
    B = (B_n / model.u_scale).astype(np.float64)
    C = (model.y_scale * C_n).astype(np.float64)
    D = np.array([[0.0]], dtype=np.float64)

    # Continuous poles
    c_poles = np.linalg.eigvals(A)
    order = np.lexsort((-c_poles.imag, -c_poles.real))
    c_poles_sorted = c_poles[order]

    return A, B, C, D, c_poles_sorted
