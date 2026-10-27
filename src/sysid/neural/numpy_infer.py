"""NumPy inference for Neural ODE models with zero PyTorch dependencies.

Replays the exact midpoint ODE numerical integration scheme with two sub-steps
per sample, suitable for ultra-lightweight deployment on Render (Decision D11).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumpyNeuralODE:
    """NumPy-only Neural ODE inference model.

    Parameters
    ----------
    n_x : int
        Latent state dimension.
    Ts : float
        Sampling period in seconds.
    u_scale : float
        Input scaling divisor (u_norm = u / u_scale).
    y_scale : float
        Output scaling multiplier (y = y_norm * y_scale).
    W1 : np.ndarray
        Weight matrix of layer 1, shape (hidden_dim, n_x + 1).
    b1 : np.ndarray
        Bias vector of layer 1, shape (hidden_dim,).
    W2 : np.ndarray
        Weight matrix of layer 2, shape (hidden_dim, hidden_dim).
    b2 : np.ndarray
        Bias vector of layer 2, shape (hidden_dim,).
    W3 : np.ndarray
        Weight matrix of layer 3, shape (n_x, hidden_dim).
    b3 : np.ndarray
        Bias vector of layer 3, shape (n_x,).
    C : np.ndarray
        Linear readout row vector of shape (1, n_x).
    """

    n_x: int
    Ts: float
    u_scale: float
    y_scale: float
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    C: np.ndarray

    def vector_field(self, x: np.ndarray, u_val: float) -> np.ndarray:
        """Evaluate continuous vector field f(x, u) = g(x, u) - g(0, 0).

        Parameters
        ----------
        x : np.ndarray
            Latent state vector of shape (n_x,).
        u_val : float
            Normalized scalar input value.

        Returns
        -------
        np.ndarray
            Derivative dx/dt vector of shape (n_x,).
        """
        xu = np.append(x, u_val)
        h1 = np.tanh(self.W1 @ xu + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        g_xu = self.W3 @ h2 + self.b3

        # g(0, 0) equilibrium subtraction
        zero_in = np.zeros(self.n_x + 1, dtype=np.float32)
        h1_0 = np.tanh(self.W1 @ zero_in + self.b1)
        h2_0 = np.tanh(self.W2 @ h1_0 + self.b2)
        g_00 = self.W3 @ h2_0 + self.b3

        res: np.ndarray = (g_xu - g_00).astype(np.float32)
        return res

    def simulate(self, u_norm: np.ndarray) -> np.ndarray:
        """Simulate normalized output sequence for a 1D input array.

        Performs two midpoint sub-steps of size Ts / 2 per sample interval.

        Parameters
        ----------
        u_norm : np.ndarray
            Normalized input sequence array of shape (n_samples,).

        Returns
        -------
        np.ndarray
            Simulated normalized output sequence of shape (n_samples,).
        """
        n_samples = len(u_norm)
        dt = np.float32(self.Ts / 2.0)
        dt_half = np.float32(dt / 2.0)

        x_curr = np.zeros(self.n_x, dtype=np.float32)
        y_norm = np.zeros(n_samples, dtype=np.float32)

        c_vec = self.C.ravel()

        for k in range(n_samples):
            # Readout y_n[k] = C * x[k]
            y_norm[k] = float(np.dot(c_vec, x_curr))
            u_k = float(u_norm[k])

            # Sub-step 1: from t_k to t_k + dt
            f1 = self.vector_field(x_curr, u_k)
            x_mid1 = x_curr + dt_half * f1
            f2 = self.vector_field(x_mid1, u_k)
            x_half = x_curr + dt * f2

            # Sub-step 2: from t_k + dt to t_{k+1}
            f3 = self.vector_field(x_half, u_k)
            x_mid2 = x_half + dt_half * f3
            f4 = self.vector_field(x_mid2, u_k)
            x_curr = (x_half + dt * f4).astype(np.float32)

        return y_norm


def load_numpy_neural_ode(npz_path: Path) -> NumpyNeuralODE:
    """Load exported Neural ODE parameters from an .npz file.

    Parameters
    ----------
    npz_path : Path
        Path to neural_ode.npz.

    Returns
    -------
    NumpyNeuralODE
        Constructed NumPy inference engine.
    """
    with np.load(npz_path) as data:
        arrays: dict[str, Any] = {k: data[k] for k in data.files}

    return NumpyNeuralODE(
        n_x=int(arrays["n_x"]),
        Ts=float(arrays["Ts"]),
        u_scale=float(arrays["u_scale"]),
        y_scale=float(arrays["y_scale"]),
        W1=arrays["W1"].astype(np.float32),
        b1=arrays["b1"].astype(np.float32),
        W2=arrays["W2"].astype(np.float32),
        b2=arrays["b2"].astype(np.float32),
        W3=arrays["W3"].astype(np.float32),
        b3=arrays["b3"].astype(np.float32),
        C=arrays["C"].astype(np.float32),
    )
