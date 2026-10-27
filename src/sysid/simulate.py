"""Plant simulation and discretization utilities.

Implements exact zero-order hold (ZOH) discrete-time simulation matching the physical
ADC/DAC data-acquisition setup, plus additive Gaussian output noise injection.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

from sysid.systems import PlantModel


def discretize_plant_zoh(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Ts: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the exact continuous-to-discrete ZOH state-space conversion.

    Uses scipy.signal.cont2discrete with method='zoh'.

    Parameters
    ----------
    A : np.ndarray
        Continuous state matrix of shape (n_states, n_states).
    B : np.ndarray
        Continuous input matrix of shape (n_states, 1).
    C : np.ndarray
        Continuous output matrix of shape (1, n_states).
    D : np.ndarray
        Continuous feedthrough matrix of shape (1, 1).
    Ts : float
        Sampling period in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Ad, Bd, Cd, Dd) discrete-time state-space matrices.
    """
    sys_d = scipy.signal.cont2discrete((A, B, C, D), Ts, method="zoh")
    Ad: np.ndarray = sys_d[0]
    Bd: np.ndarray = sys_d[1]
    Cd: np.ndarray = sys_d[2]
    Dd: np.ndarray = sys_d[3]
    return Ad, Bd, Cd, Dd


def simulate_discrete(
    Ad: np.ndarray,
    Bd: np.ndarray,
    Cd: np.ndarray,
    Dd: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate discrete-time state-space model for a given input sequence.

    Iterates the linear difference equations:
        x[k + 1] = Ad * x[k] + Bd * u[k]
        y[k] = Cd * x[k] + Dd * u[k]

    Parameters
    ----------
    Ad : np.ndarray
        Discrete state matrix of shape (n_states, n_states).
    Bd : np.ndarray
        Discrete input matrix of shape (n_states, 1).
    Cd : np.ndarray
        Discrete output matrix of shape (1, n_states).
    Dd : np.ndarray
        Discrete feedthrough matrix of shape (1, 1).
    u : np.ndarray
        Input sequence array of shape (n_samples,).
    x0 : np.ndarray | None, optional
        Initial state of shape (n_states,). If None, system starts at rest (x0 = 0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (y, x)
        - y : Simulated output array of shape (n_samples,).
        - x : State trajectory array of shape (n_samples, n_states).
    """
    n_samples = len(u)
    n_states = Ad.shape[0]

    if x0 is None:
        x_curr = np.zeros(n_states, dtype=np.float64)
    else:
        x_curr = np.array(x0, dtype=np.float64).reshape(-1)

    y = np.zeros(n_samples, dtype=np.float64)
    x = np.zeros((n_samples, n_states), dtype=np.float64)

    b_vec = Bd.ravel()
    c_vec = Cd.ravel()
    d_scalar = float(Dd[0, 0])

    for k in range(n_samples):
        x[k] = x_curr
        u_k = float(u[k])
        y[k] = np.dot(c_vec, x_curr) + d_scalar * u_k
        x_curr = Ad @ x_curr + b_vec * u_k

    return y, x


def simulate_plant(
    plant: PlantModel,
    u: np.ndarray,
    Ts: float,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a continuous PlantModel with zero-order hold at period Ts.

    Parameters
    ----------
    plant : PlantModel
        Continuous LTI plant model.
    u : np.ndarray
        Input sequence array of shape (n_samples,).
    Ts : float
        Sampling period in seconds.
    x0 : np.ndarray | None, optional
        Initial state vector, by default None (starts at rest).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (y, x) output and state trajectory.
    """
    Ad, Bd, Cd, Dd = discretize_plant_zoh(plant.A, plant.B, plant.C, plant.D, Ts)
    return simulate_discrete(Ad, Bd, Cd, Dd, u, x0=x0)


def calculate_noise_sigma(y: np.ndarray, snr_db: float | None) -> float:
    """Calculate the noise standard deviation for a requested SNR in dB.

    Formula:
        sigma = std(y) * 10^(-SNR_dB / 20)

    Parameters
    ----------
    y : np.ndarray
        Clean output signal array.
    snr_db : float | None
        Target signal-to-noise ratio in decibels. If None or inf, returns 0.0.

    Returns
    -------
    float
        Noise standard deviation sigma.
    """
    if snr_db is None or np.isinf(snr_db):
        return 0.0
    signal_std = float(np.std(y))
    if signal_std < 1e-12:
        return 0.0
    return float(signal_std * (10.0 ** (-snr_db / 20.0)))


def add_output_noise(
    y: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
    sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """Add additive zero-mean Gaussian measurement noise to the output signal.

    Parameters
    ----------
    y : np.ndarray
        Clean output signal array of shape (n_samples,).
    snr_db : float | None
        Target signal-to-noise ratio in decibels. None indicates noise-free.
    rng : np.random.Generator
        Explicit pseudorandom number generator.
    sigma : float | None, optional
        Pre-calculated standard deviation. If None, computed from y and snr_db.

    Returns
    -------
    tuple[np.ndarray, float]
        (y_noisy, sigma_used)
    """
    if snr_db is None or np.isinf(snr_db):
        return y.copy(), 0.0

    if sigma is None:
        sigma = calculate_noise_sigma(y, snr_db)

    if sigma == 0.0:
        return y.copy(), 0.0

    noise = rng.normal(0.0, sigma, size=len(y))
    return y + noise, sigma
