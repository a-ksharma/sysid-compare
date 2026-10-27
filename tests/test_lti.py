"""Unit tests for lti.py against python-control reference implementations."""

from __future__ import annotations

import control
import numpy as np
import pytest

from sysid.lti import frequency_response, impulse_response, poles
from sysid.systems import get_plant


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_lti_frequency_response_agrees_with_control(plant_id: str) -> None:
    """Verify frequency response matches python-control to relative error < 1e-9."""
    plant = get_plant(plant_id)
    sys_ctrl = control.ss(plant.A, plant.B, plant.C, plant.D)

    omega = np.geomspace(0.1, 60.0, 200)

    # sysid pure numpy implementation
    mag_calc, phase_calc, complex_calc = frequency_response(
        plant.A, plant.B, plant.C, plant.D, omega
    )

    # python-control reference
    resp_ctrl = control.frequency_response(sys_ctrl, omega)
    mag_ctrl = np.squeeze(resp_ctrl.magnitude)
    complex_ctrl = np.squeeze(resp_ctrl.frdata[0][0])

    # Magnitude check: relative error < 1e-9
    np.testing.assert_allclose(
        mag_calc,
        mag_ctrl,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"Magnitude mismatch for {plant_id}",
    )

    # Complex response check: relative error < 1e-9
    np.testing.assert_allclose(
        complex_calc,
        complex_ctrl,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"Complex response mismatch for {plant_id}",
    )

    # Phase check: wrapped difference in degrees < 1e-9
    phase_ctrl_deg = np.rad2deg(np.squeeze(resp_ctrl.phase))
    phase_diff = (phase_calc - phase_ctrl_deg + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(
        phase_diff,
        0.0,
        atol=1e-9,
        err_msg=f"Phase mismatch for {plant_id}",
    )


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_lti_impulse_response_agrees_with_control(plant_id: str) -> None:
    """Verify impulse response matches python-control to relative error < 1e-9."""
    plant = get_plant(plant_id)
    sys_ctrl = control.ss(plant.A, plant.B, plant.C, plant.D)

    t = np.linspace(0.0, plant.impulse_horizon_s, 200)

    h_calc = impulse_response(plant.A, plant.B, plant.C, t)

    # python-control reference
    # control.impulse_response returns TimeResponseData with .outputs
    res_ctrl = control.impulse_response(sys_ctrl, T=t)
    h_ctrl = np.squeeze(res_ctrl.outputs)

    # Relative L2 error < 1e-9
    rel_error = float(np.linalg.norm(h_calc - h_ctrl) / np.linalg.norm(h_ctrl))
    assert rel_error < 1e-9, (
        f"Impulse rel error {rel_error:.2e} exceeded 1e-9 for {plant_id}"
    )


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_lti_poles_agrees_with_control(plant_id: str) -> None:
    """Verify poles match python-control eigenvalues to within 1e-9."""
    plant = get_plant(plant_id)
    sys_ctrl = control.ss(plant.A, plant.B, plant.C, plant.D)

    poles_calc = poles(plant.A)
    poles_ctrl = control.poles(sys_ctrl)

    # Sort both for robust element-wise comparison
    order_calc = np.lexsort((-poles_calc.imag, -poles_calc.real))
    order_ctrl = np.lexsort((-poles_ctrl.imag, -poles_ctrl.real))

    np.testing.assert_allclose(
        poles_calc[order_calc],
        poles_ctrl[order_ctrl],
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"Poles mismatch for {plant_id}",
    )
