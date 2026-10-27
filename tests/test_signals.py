"""Unit tests for excitation signal generators."""

from __future__ import annotations

import numpy as np
import pytest

from sysid.signals import (
    generate_chirp,
    generate_impulse_pulse,
    generate_multisine,
    generate_random_steps,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_step,
    generate_time_vector,
)


def test_time_vector() -> None:
    """Verify uniform time vector length and interval."""
    t = generate_time_vector(duration_s=1.0, Ts=0.02)
    assert len(t) == 50
    assert pytest.approx(t[0], abs=1e-9) == 0.0
    assert pytest.approx(t[-1], abs=1e-9) == 0.98


def test_quiet_lead_enforcement() -> None:
    """Verify that the first 0.5 s has u = 0 for all dynamic signal generators."""
    Ts = 0.02
    quiet_samples = int(round(0.5 / Ts))
    rng = np.random.default_rng(42)

    u_steps = generate_random_steps(6.0, Ts, 5.0, rng, quiet_lead_s=0.5)
    np.testing.assert_allclose(u_steps[:quiet_samples], 0.0)

    u_multi = generate_multisine(6.0, Ts, 5.0, rng, quiet_lead_s=0.5)
    np.testing.assert_allclose(u_multi[:quiet_samples], 0.0)

    u_chirp = generate_chirp(6.0, Ts, 5.0, rng, quiet_lead_s=0.5)
    np.testing.assert_allclose(u_chirp[:quiet_samples], 0.0)


def test_amplitude_bounds() -> None:
    """Verify generated signals do not exceed peak physical amplitude u_max."""
    Ts = 0.02
    u_max = 10.0
    rng = np.random.default_rng(123)

    u_steps = generate_random_steps(10.0, Ts, u_max, rng)
    assert np.max(np.abs(u_steps)) <= u_max + 1e-9

    u_multi = generate_multisine(10.0, Ts, u_max, rng)
    assert np.max(np.abs(u_multi)) <= u_max + 1e-9

    u_chirp = generate_chirp(10.0, Ts, u_max, rng)
    assert np.max(np.abs(u_chirp)) <= u_max + 1e-9


def test_seed_reproducibility() -> None:
    """Verify identical seeds produce identical signal arrays."""
    Ts = 0.02
    u_max = 5.0

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    s1 = generate_random_steps(5.0, Ts, u_max, rng1)
    s2 = generate_random_steps(5.0, Ts, u_max, rng2)
    np.testing.assert_array_equal(s1, s2)

    rng1 = np.random.default_rng(888)
    rng2 = np.random.default_rng(888)

    m1 = generate_multisine(5.0, Ts, u_max, rng1)
    m2 = generate_multisine(5.0, Ts, u_max, rng2)
    np.testing.assert_array_equal(m1, m2)


def test_deterministic_signals() -> None:
    """Verify step, sine, square, sawtooth, and impulse signals."""
    Ts = 0.02
    duration_s = 2.0
    amplitude = 4.0

    step = generate_step(duration_s, Ts, amplitude, quiet_lead_s=0.5)
    assert len(step) == 100
    np.testing.assert_allclose(step[:25], 0.0)
    np.testing.assert_allclose(step[25:], amplitude)

    sine = generate_sine(
        duration_s, Ts, amplitude, frequency_rad_s=1.0, quiet_lead_s=0.5
    )
    np.testing.assert_allclose(sine[:25], 0.0)
    assert np.max(np.abs(sine)) <= amplitude + 1e-9

    square = generate_square(duration_s, Ts, amplitude, period_s=1.0, quiet_lead_s=0.5)
    np.testing.assert_allclose(square[:25], 0.0)
    assert set(np.unique(square[25:])) == {-amplitude, amplitude}

    saw = generate_sawtooth(duration_s, Ts, amplitude, period_s=1.0, quiet_lead_s=0.5)
    np.testing.assert_allclose(saw[:25], 0.0)
    assert np.max(np.abs(saw)) <= amplitude + 1e-9

    impulse = generate_impulse_pulse(duration_s, Ts, u_max=10.0, amplitude_fraction=0.1)
    assert impulse[0] == 1.0  # 0.1 * 10
    np.testing.assert_allclose(impulse[1:], 0.0)
