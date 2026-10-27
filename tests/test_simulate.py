"""Unit tests for plant simulation, ZOH discretization, and noise generation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

from sysid.signals import generate_random_steps
from sysid.simulate import (
    add_output_noise,
    discretize_plant_zoh,
    simulate_discrete,
    simulate_plant,
)
from sysid.systems import get_plant


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_zoh_simulation_agrees_with_solve_ivp(plant_id: str) -> None:
    """Verify ZOH discrete simulation matches solve_ivp to relative error < 1e-6.

    Simulates the exact continuous differential equation dx/dt = A*x + B*u(t)
    under a piecewise-constant (ZOH) input using a high-order adaptive integrator
    (DOP853) with tight tolerances.
    """
    plant = get_plant(plant_id)
    Ts = 0.02
    duration_s = plant.sequence_duration_s
    rng = np.random.default_rng(42)

    # Generate a random steps sequence
    u = generate_random_steps(duration_s, Ts, plant.u_max, rng)

    # Discrete ZOH simulation
    y_discrete, _ = simulate_plant(plant, u, Ts)

    # Continuous integration with solve_ivp on piecewise-constant intervals
    A = plant.A
    B_vec = plant.B.ravel()
    C_vec = plant.C.ravel()
    n_samples = len(u)

    y_ivp = np.zeros(n_samples, dtype=np.float64)
    x_curr = np.zeros(plant.order, dtype=np.float64)

    for k in range(n_samples):
        y_ivp[k] = np.dot(C_vec, x_curr)
        u_val = float(u[k])
        # On each interval [0, Ts], the ODE dx/dt = A*x + B*u is smooth with constant u
        sol = scipy.integrate.solve_ivp(
            lambda _t, x_val: A @ x_val + B_vec * u_val,
            (0.0, Ts),
            x_curr,
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
        )
        x_curr = sol.y[:, -1]

    # Relative L2 error
    rel_error = float(np.linalg.norm(y_discrete - y_ivp) / np.linalg.norm(y_ivp))
    assert rel_error < 1e-6, (
        f"Relative error {rel_error:.2e} exceeded 1e-6 for {plant_id}"
    )


def test_noise_generator_snr_accuracy() -> None:
    """Verify noise generator matches requested SNR within 0.1 dB on a long record."""
    rng = np.random.default_rng(2026)
    target_snr_db = 40.0
    n_samples = 100_000

    # Deterministic multi-frequency test signal
    t = np.linspace(0.0, 100.0, n_samples)
    y_clean = np.sin(2.0 * np.pi * 0.5 * t) + 0.5 * np.cos(2.0 * np.pi * 1.5 * t)

    y_noisy, sigma_used = add_output_noise(y_clean, target_snr_db, rng)

    noise = y_noisy - y_clean
    var_signal = float(np.var(y_clean))
    var_noise = float(np.var(noise))

    empirical_snr_db = 10.0 * np.log10(var_signal / var_noise)
    error_db = abs(empirical_snr_db - target_snr_db)

    assert error_db < 0.1, (
        f"Empirical SNR {empirical_snr_db:.3f} dB differs from "
        f"target {target_snr_db} dB by {error_db:.3f} dB (> 0.1 dB)"
    )


def test_noise_generator_edge_cases() -> None:
    """Verify behavior on None / infinite SNR and pre-calculated sigma."""
    rng = np.random.default_rng(123)
    y = np.ones(50, dtype=np.float64)

    # Noise free
    y_out, sigma = add_output_noise(y, None, rng)
    np.testing.assert_array_equal(y_out, y)
    assert sigma == 0.0

    y_out_inf, sigma_inf = add_output_noise(y, float("inf"), rng)
    np.testing.assert_array_equal(y_out_inf, y)
    assert sigma_inf == 0.0

    # Pre-calculated sigma
    custom_sigma = 0.05
    y_noisy, sigma_used = add_output_noise(y, 30.0, rng, sigma=custom_sigma)
    assert sigma_used == custom_sigma
    assert pytest.approx(float(np.std(y_noisy - y)), rel=0.1) == custom_sigma


def test_discretize_plant_zoh_structure() -> None:
    """Verify cont2discrete preserves state dimensions and produces zero Dd."""
    plant = get_plant("msd")
    Ts = 0.02
    Ad, Bd, Cd, Dd = discretize_plant_zoh(plant.A, plant.B, plant.C, plant.D, Ts)

    assert Ad.shape == (2, 2)
    assert Bd.shape == (2, 1)
    assert Cd.shape == (1, 2)
    assert Dd.shape == (1, 1)
    assert Dd[0, 0] == 0.0


def test_simulate_discrete_custom_initial_state() -> None:
    """Verify simulate_discrete respects non-zero initial state."""
    plant = get_plant("rc")
    Ts = 0.02
    Ad, Bd, Cd, Dd = discretize_plant_zoh(plant.A, plant.B, plant.C, plant.D, Ts)

    u_zero = np.zeros(20, dtype=np.float64)
    x0 = np.array([3.0])

    y, x = simulate_discrete(Ad, Bd, Cd, Dd, u_zero, x0=x0)
    assert pytest.approx(y[0], rel=1e-9) == 3.0
    # Decaying towards zero
    assert y[-1] < y[0]
