"""Unit tests for plant models, registry, and identifiable parameter extraction."""

from __future__ import annotations

import control
import numpy as np
import pytest

from sysid.systems import (
    create_msd_plant,
    create_rc_plant,
    extract_identifiable_parameters,
    get_all_plants,
    get_plant,
)


def test_rc_plant_properties() -> None:
    """Verify RC plant parameters, matrices, and state dimensions."""
    plant = get_plant("rc")
    assert plant.id == "rc"
    assert plant.order == 1
    assert plant.u_max == 5.0
    assert plant.A.shape == (1, 1)
    assert plant.B.shape == (1, 1)
    assert plant.C.shape == (1, 1)
    assert plant.D.shape == (1, 1)
    assert plant.derived["tau_s"] == 0.2
    assert plant.identifiable == ("tau_s",)


def test_motor_plant_properties() -> None:
    """Verify DC motor plant parameters, matrices, and state dimensions."""
    plant = get_plant("motor")
    assert plant.id == "motor"
    assert plant.order == 1
    assert plant.u_max == 12.0
    assert plant.derived["K_rad_s_per_V"] == 5.0
    assert plant.derived["tau_s"] == 0.25
    assert set(plant.identifiable) == {"K_rad_s_per_V", "tau_s"}


def test_msd_plant_properties() -> None:
    """Verify mass-spring-damper plant parameters, matrices, and state dimensions."""
    plant = get_plant("msd")
    assert plant.id == "msd"
    assert plant.order == 2
    assert plant.u_max == 10.0
    assert plant.A.shape == (2, 2)
    assert plant.derived["omega_n_rad_s"] == 2.0
    assert pytest.approx(plant.derived["zeta"], rel=1e-6) == 0.2
    assert set(plant.identifiable) == {"m_kg", "c_Ns_per_m", "k_N_per_m"}


def test_all_plants_load() -> None:
    """Verify get_all_plants loads all 3 plants from the YAML configuration."""
    plants = get_all_plants()
    assert set(plants.keys()) == {"rc", "motor", "msd"}
    with pytest.raises(ValueError, match="Unknown plant id"):
        get_plant("nonexistent")


@pytest.mark.parametrize("plant_id", ["rc", "motor", "msd"])
def test_state_space_and_transfer_function_agreement(plant_id: str) -> None:
    """Verify state-space realization matches transfer function via python-control."""
    plant = get_plant(plant_id)

    # Build continuous state-space object with python-control
    ss_ctrl = control.ss(plant.A, plant.B, plant.C, plant.D)
    tf_ctrl = control.ss2tf(ss_ctrl)

    # Normalize numerator and denominator
    den_ctrl = np.squeeze(tf_ctrl.den[0][0])
    num_ctrl = np.squeeze(tf_ctrl.num[0][0])
    den_lead = float(den_ctrl[0])
    den_ctrl_norm = den_ctrl / den_lead
    num_ctrl_norm = num_ctrl / den_lead

    # Check denominator matches
    np.testing.assert_allclose(
        plant.den,
        den_ctrl_norm,
        rtol=1e-7,
        atol=1e-7,
        err_msg=f"Denominator mismatch for {plant_id}",
    )

    # Check non-zero numerator coefficients match (filter numerical fuzz below 1e-12)
    plant_num = plant.num
    num_ctrl_active = num_ctrl_norm[np.abs(num_ctrl_norm) > 1e-12]
    plant_num_active = plant_num[np.abs(plant_num) > 1e-12]
    np.testing.assert_allclose(
        plant_num_active,
        num_ctrl_active,
        rtol=1e-7,
        atol=1e-7,
        err_msg=f"Numerator mismatch for {plant_id}",
    )


def test_extract_identifiable_parameters_roundtrip() -> None:
    """Verify exact parameter recovery from transfer functions for all plants."""
    for plant_id in ("rc", "motor", "msd"):
        plant = get_plant(plant_id)
        params = extract_identifiable_parameters(plant_id, plant.num, plant.den)
        for param_name in plant.identifiable:
            expected = plant.derived.get(param_name, plant.parameters.get(param_name))
            assert expected is not None
            assert pytest.approx(params[param_name], rel=1e-6) == expected


def test_perturbed_plant_creation() -> None:
    """Verify creation of perturbed plant models for parameter variation studies."""
    perturbed_msd = create_msd_plant(m_kg=1.2, c_ns_per_m=0.9, k_n_per_m=4.5)
    params = extract_identifiable_parameters(
        "msd", perturbed_msd.num, perturbed_msd.den
    )
    assert pytest.approx(params["m_kg"], rel=1e-6) == 1.2
    assert pytest.approx(params["c_Ns_per_m"], rel=1e-6) == 0.9
    assert pytest.approx(params["k_N_per_m"], rel=1e-6) == 4.5

    perturbed_rc = create_rc_plant(tau_s=0.3)
    params_rc = extract_identifiable_parameters(
        "rc", perturbed_rc.num, perturbed_rc.den
    )
    assert pytest.approx(params_rc["tau_s"], rel=1e-6) == 0.3
