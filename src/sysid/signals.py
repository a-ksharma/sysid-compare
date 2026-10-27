"""Signal generation utilities for system identification.

Generates excitation signals in normalized units (peak magnitude <= 1.0)
before scaling to physical units (u_max). The initial period (default 0.5 s)
maintains u = 0 to observe the system at rest.
"""

from __future__ import annotations

import numpy as np


def generate_time_vector(duration_s: float, Ts: float) -> np.ndarray:
    """Generate uniform time vector starting at t = 0 with step Ts.

    Parameters
    ----------
    duration_s : float
        Total duration in seconds.
    Ts : float
        Sampling interval in seconds.

    Returns
    -------
    np.ndarray
        Array of time points t = k * Ts.
    """
    n_samples = int(round(duration_s / Ts))
    return np.arange(n_samples, dtype=np.float64) * Ts


def apply_quiet_lead(u: np.ndarray, Ts: float, quiet_lead_s: float = 0.5) -> np.ndarray:
    """Enforce zero input during the initial quiet lead period.

    Parameters
    ----------
    u : np.ndarray
        Input signal array.
    Ts : float
        Sampling period in seconds.
    quiet_lead_s : float, optional
        Duration of quiet lead in seconds, by default 0.5.

    Returns
    -------
    np.ndarray
        Input signal with the first quiet_lead_s set to zero.
    """
    u_out = u.copy()
    n_quiet = int(round(quiet_lead_s / Ts))
    if n_quiet > 0:
        u_out[: min(n_quiet, len(u_out))] = 0.0
    return u_out


def generate_random_steps(
    duration_s: float,
    Ts: float,
    u_max: float,
    rng: np.random.Generator,
    quiet_lead_s: float = 0.5,
    hold_time_range_s: tuple[float, float] = (0.3, 2.0),
) -> np.ndarray:
    """Generate piecewise-constant random step excitation.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    u_max : float
        Peak amplitude scaling factor.
    rng : np.random.Generator
        Explicit pseudorandom number generator.
    quiet_lead_s : float, optional
        Initial rest period in seconds, by default 0.5.
    hold_time_range_s : tuple[float, float], optional
        Min and max hold times in seconds, by default (0.3, 2.0).

    Returns
    -------
    np.ndarray
        Excitation signal array of length n_samples.
    """
    n_samples = int(round(duration_s / Ts))
    u_norm = np.zeros(n_samples, dtype=np.float64)
    n_quiet = int(round(quiet_lead_s / Ts))

    curr_idx = n_quiet
    while curr_idx < n_samples:
        hold_s = rng.uniform(hold_time_range_s[0], hold_time_range_s[1])
        hold_samples = max(1, int(round(hold_s / Ts)))
        level = rng.uniform(-1.0, 1.0)
        end_idx = min(n_samples, curr_idx + hold_samples)
        u_norm[curr_idx:end_idx] = level
        curr_idx = end_idx

    return u_norm * u_max


def generate_multisine(
    duration_s: float,
    Ts: float,
    u_max: float,
    rng: np.random.Generator,
    quiet_lead_s: float = 0.5,
    n_components: int = 12,
    omega_range_rad_s: tuple[float, float] = (0.1, 15.0),
) -> np.ndarray:
    """Generate a multisine excitation with log-spaced frequencies.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    u_max : float
        Peak amplitude scaling factor.
    rng : np.random.Generator
        Explicit pseudorandom number generator.
    quiet_lead_s : float, optional
        Initial rest period in seconds, by default 0.5.
    n_components : int, optional
        Number of sinusoidal frequency components, by default 12.
    omega_range_rad_s : tuple[float, float], optional
        Angular frequency range [min, max] in rad/s, by default (0.1, 15.0).

    Returns
    -------
    np.ndarray
        Excitation signal array of length n_samples.
    """
    t = generate_time_vector(duration_s, Ts)
    omegas = np.geomspace(omega_range_rad_s[0], omega_range_rad_s[1], n_components)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)

    # Active time relative to quiet lead
    t_active = np.maximum(0.0, t - quiet_lead_s)
    # Sum components: sum_i sin(omega_i * t + phase_i)
    signal = np.zeros_like(t)
    for w, phi in zip(omegas, phases, strict=True):
        signal += np.sin(w * t_active + phi)

    # Peak normalization to unit magnitude across active samples
    n_quiet = int(round(quiet_lead_s / Ts))
    active_slice = signal[n_quiet:]
    peak = float(np.max(np.abs(active_slice))) if len(active_slice) > 0 else 1.0
    if peak > 1e-12:
        signal = signal / peak

    signal = apply_quiet_lead(signal, Ts, quiet_lead_s)
    return signal * u_max


def generate_chirp(
    duration_s: float,
    Ts: float,
    u_max: float,
    rng: np.random.Generator,
    quiet_lead_s: float = 0.5,
    omega_range_rad_s: tuple[float, float] = (0.1, 15.0),
) -> np.ndarray:
    """Generate a logarithmic frequency chirp excitation.

    Log sweep: instantaneous frequency omega(t) = w0 * (w1 / w0)^(t / T_active).
    Phase phi(t) = integral_0^t omega(tau) dtau + phi0
                 = w0 * (k^t - 1) / ln(k) + phi0, where k = w1 / w0.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    u_max : float
        Peak amplitude scaling factor.
    rng : np.random.Generator
        Explicit pseudorandom number generator.
    quiet_lead_s : float, optional
        Initial rest period in seconds, by default 0.5.
    omega_range_rad_s : tuple[float, float], optional
        Swept frequency range [w0, w1] in rad/s, by default (0.1, 15.0).

    Returns
    -------
    np.ndarray
        Excitation signal array of length n_samples.
    """
    t = generate_time_vector(duration_s, Ts)
    t_total_active = duration_s - quiet_lead_s
    w0, w1 = omega_range_rad_s
    amplitude = rng.uniform(0.3, 1.0)
    phi0 = rng.uniform(0.0, 2.0 * np.pi)

    t_rel = np.clip(t - quiet_lead_s, 0.0, t_total_active)
    ratio = w1 / w0
    log_ratio = np.log(ratio)

    # Analytical phase integral for logarithmic sweep
    phase = (w0 * t_total_active / log_ratio) * (
        np.power(ratio, t_rel / t_total_active) - 1.0
    ) + phi0
    signal = amplitude * np.sin(phase)

    signal = apply_quiet_lead(signal, Ts, quiet_lead_s)
    return signal * u_max


def generate_step(
    duration_s: float,
    Ts: float,
    amplitude: float,
    quiet_lead_s: float = 0.5,
) -> np.ndarray:
    """Generate a step input signal with quiet lead.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    amplitude : float
        Step height in physical units.
    quiet_lead_s : float, optional
        Initial rest duration in seconds, by default 0.5.

    Returns
    -------
    np.ndarray
        Step input array of length n_samples.
    """
    n_samples = int(round(duration_s / Ts))
    u = np.full(n_samples, amplitude, dtype=np.float64)
    return apply_quiet_lead(u, Ts, quiet_lead_s)


def generate_sine(
    duration_s: float,
    Ts: float,
    amplitude: float,
    frequency_rad_s: float,
    quiet_lead_s: float = 0.5,
) -> np.ndarray:
    """Generate a single-frequency sine wave with quiet lead.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    amplitude : float
        Sine wave amplitude in physical units.
    frequency_rad_s : float
        Angular frequency in rad/s.
    quiet_lead_s : float, optional
        Initial rest duration in seconds, by default 0.5.

    Returns
    -------
    np.ndarray
        Sine wave input array of length n_samples.
    """
    t = generate_time_vector(duration_s, Ts)
    t_active = np.maximum(0.0, t - quiet_lead_s)
    u = amplitude * np.sin(frequency_rad_s * t_active)
    return apply_quiet_lead(u, Ts, quiet_lead_s)


def generate_square(
    duration_s: float,
    Ts: float,
    amplitude: float,
    period_s: float,
    quiet_lead_s: float = 0.5,
) -> np.ndarray:
    """Generate a square wave with 50% duty cycle and quiet lead.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    amplitude : float
        Wave amplitude in physical units.
    period_s : float
        Fundamental period in seconds.
    quiet_lead_s : float, optional
        Initial rest duration in seconds, by default 0.5.

    Returns
    -------
    np.ndarray
        Square wave input array of length n_samples.
    """
    t = generate_time_vector(duration_s, Ts)
    t_active = np.maximum(0.0, t - quiet_lead_s)
    phase = (t_active % period_s) / period_s
    u = np.where(phase < 0.5, amplitude, -amplitude)
    return apply_quiet_lead(u, Ts, quiet_lead_s)


def generate_sawtooth(
    duration_s: float,
    Ts: float,
    amplitude: float,
    period_s: float,
    quiet_lead_s: float = 0.5,
) -> np.ndarray:
    """Generate a sawtooth wave with quiet lead.

    Ramps linearly from -amplitude to +amplitude over period_s.

    Parameters
    ----------
    duration_s : float
        Sequence duration in seconds.
    Ts : float
        Sampling period in seconds.
    amplitude : float
        Wave peak amplitude in physical units.
    period_s : float
        Fundamental period in seconds.
    quiet_lead_s : float, optional
        Initial rest duration in seconds, by default 0.5.

    Returns
    -------
    np.ndarray
        Sawtooth wave input array of length n_samples.
    """
    t = generate_time_vector(duration_s, Ts)
    t_active = np.maximum(0.0, t - quiet_lead_s)
    phase = (t_active % period_s) / period_s
    u = amplitude * (2.0 * phase - 1.0)
    return apply_quiet_lead(u, Ts, quiet_lead_s)


def generate_impulse_pulse(
    duration_s: float,
    Ts: float,
    u_max: float,
    amplitude_fraction: float = 0.1,
) -> np.ndarray:
    """Generate a single-sample zero-order-hold impulse pulse for Prony's method.

    Pulse of height amplitude_fraction * u_max at sample 0, zero afterwards.

    Parameters
    ----------
    duration_s : float
        Record duration in seconds.
    Ts : float
        Sampling period in seconds.
    u_max : float
        Plant maximum input in physical units.
    amplitude_fraction : float, optional
        Fraction of u_max for the pulse height, by default 0.1.

    Returns
    -------
    np.ndarray
        Input pulse array of length n_samples.
    """
    n_samples = int(round(duration_s / Ts))
    u = np.zeros(n_samples, dtype=np.float64)
    if n_samples > 0:
        u[0] = amplitude_fraction * u_max
    return u
