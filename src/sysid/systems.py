"""Plant definitions: registry, parameters, transfer functions, and state space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from sysid.config import SystemConfig, load_systems


@dataclass(frozen=True)
class PlantModel:
    """Continuous-time linear time-invariant plant model.

    Parameters
    ----------
    id : str
        Unique plant identifier ('rc', 'motor', or 'msd').
    name : str
        Descriptive human-readable plant name.
    order : int
        System order (1 or 2).
    A : np.ndarray
        State transition matrix of shape (n_states, n_states).
    B : np.ndarray
        Input matrix of shape (n_states, 1).
    C : np.ndarray
        Output matrix of shape (1, n_states).
    D : np.ndarray
        Feedthrough matrix of shape (1, 1).
    num : np.ndarray
        Transfer function numerator polynomial coefficients in descending powers of s.
    den : np.ndarray
        Transfer function denominator polynomial coefficients in descending powers of s.
    u_max : float
        Maximum operational input amplitude in physical units.
    u_name : str
        Physical input quantity name.
    u_unit : str
        Physical input unit.
    y_name : str
        Physical output quantity name.
    y_unit : str
        Physical output unit.
    dominant_time_scale_s : float
        Dominant time scale in seconds for sequence horizon calculation.
    sequence_duration_s : float
        Nominal duration of training sequences in seconds.
    impulse_horizon_s : float
        Horizon for impulse response evaluation in seconds.
    parameters : Mapping[str, float]
        Nominal physical parameters.
    derived : Mapping[str, float]
        Derived parameters (e.g. tau, omega_n, zeta).
    identifiable : tuple[str, ...]
        Names of parameters identifiable from input-output data.
    """

    id: str
    name: str
    order: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    num: np.ndarray
    den: np.ndarray
    u_max: float
    u_name: str
    u_unit: str
    y_name: str
    y_unit: str
    dominant_time_scale_s: float
    sequence_duration_s: float
    impulse_horizon_s: float
    parameters: Mapping[str, float]
    derived: Mapping[str, float]
    identifiable: tuple[str, ...]


def create_rc_plant(
    tau_s: float = 0.2,
    u_max: float = 5.0,
    dominant_time_scale_s: float = 0.2,
    sequence_duration_s: float = 6.0,
    impulse_horizon_s: float = 1.5,
    parameters: Mapping[str, float] | None = None,
) -> PlantModel:
    """Build the 1st-order series RC low-pass plant model.

    Governing equation:
        tau * dv/dt = u - v
        H(s) = 1 / (tau * s + 1)
    """
    if parameters is None:
        parameters = {"R_ohm": 20000.0, "C_farad": 0.00001}
    a_val = -1.0 / tau_s
    b_val = 1.0 / tau_s

    A = np.array([[a_val]], dtype=np.float64)
    B = np.array([[b_val]], dtype=np.float64)
    C = np.array([[1.0]], dtype=np.float64)
    D = np.array([[0.0]], dtype=np.float64)

    num = np.array([1.0 / tau_s], dtype=np.float64)
    den = np.array([1.0, 1.0 / tau_s], dtype=np.float64)

    return PlantModel(
        id="rc",
        name="Series RC low-pass",
        order=1,
        A=A,
        B=B,
        C=C,
        D=D,
        num=num,
        den=den,
        u_max=u_max,
        u_name="source voltage",
        u_unit="V",
        y_name="capacitor voltage",
        y_unit="V",
        dominant_time_scale_s=dominant_time_scale_s,
        sequence_duration_s=sequence_duration_s,
        impulse_horizon_s=impulse_horizon_s,
        parameters=parameters,
        derived={"tau_s": tau_s},
        identifiable=("tau_s",),
    )


def create_motor_plant(
    k_gain: float = 5.0,
    tau_s: float = 0.25,
    u_max: float = 12.0,
    dominant_time_scale_s: float = 0.25,
    sequence_duration_s: float = 10.0,
    impulse_horizon_s: float = 2.5,
    parameters: Mapping[str, float] | None = None,
) -> PlantModel:
    """Build the 1st-order armature-controlled DC motor speed plant model.

    Governing equation:
        J * dw/dt = (Kt / R) * u - (b + Kt * Ke / R) * w
        H(s) = K / (tau * s + 1)
    """
    if parameters is None:
        parameters = {
            "R_ohm": 1.0,
            "Kt": 0.1,
            "Ke": 0.1,
            "b": 0.01,
            "J_kg_m2": 0.005,
        }
    a_val = -1.0 / tau_s
    b_val = k_gain / tau_s

    A = np.array([[a_val]], dtype=np.float64)
    B = np.array([[b_val]], dtype=np.float64)
    C = np.array([[1.0]], dtype=np.float64)
    D = np.array([[0.0]], dtype=np.float64)

    num = np.array([k_gain / tau_s], dtype=np.float64)
    den = np.array([1.0, 1.0 / tau_s], dtype=np.float64)

    return PlantModel(
        id="motor",
        name="Armature-controlled DC motor (inductance neglected)",
        order=1,
        A=A,
        B=B,
        C=C,
        D=D,
        num=num,
        den=den,
        u_max=u_max,
        u_name="armature voltage",
        u_unit="V",
        y_name="shaft speed",
        y_unit="rad/s",
        dominant_time_scale_s=dominant_time_scale_s,
        sequence_duration_s=sequence_duration_s,
        impulse_horizon_s=impulse_horizon_s,
        parameters=parameters,
        derived={"K_rad_s_per_V": k_gain, "tau_s": tau_s},
        identifiable=("K_rad_s_per_V", "tau_s"),
    )


def create_msd_plant(
    m_kg: float = 1.0,
    c_ns_per_m: float = 0.8,
    k_n_per_m: float = 4.0,
    u_max: float = 10.0,
    dominant_time_scale_s: float = 2.5,
    sequence_duration_s: float = 15.0,
    impulse_horizon_s: float = 12.0,
) -> PlantModel:
    """Build the 2nd-order underdamped mass-spring-damper plant model.

    Governing equation:
        m * x'' + c * x' + k * x = u
        H(s) = 1 / (m * s^2 + c * s + k)
    States: [x, v] where v = x'
    """
    omega_n = np.sqrt(k_n_per_m / m_kg)
    zeta = c_ns_per_m / (2.0 * np.sqrt(k_n_per_m * m_kg))

    A = np.array(
        [[0.0, 1.0], [-k_n_per_m / m_kg, -c_ns_per_m / m_kg]], dtype=np.float64
    )
    B = np.array([[0.0], [1.0 / m_kg]], dtype=np.float64)
    C = np.array([[1.0, 0.0]], dtype=np.float64)
    D = np.array([[0.0]], dtype=np.float64)

    num = np.array([1.0 / m_kg], dtype=np.float64)
    den = np.array([1.0, c_ns_per_m / m_kg, k_n_per_m / m_kg], dtype=np.float64)

    return PlantModel(
        id="msd",
        name="Mass-spring-damper",
        order=2,
        A=A,
        B=B,
        C=C,
        D=D,
        num=num,
        den=den,
        u_max=u_max,
        u_name="force",
        u_unit="N",
        y_name="displacement",
        y_unit="m",
        dominant_time_scale_s=dominant_time_scale_s,
        sequence_duration_s=sequence_duration_s,
        impulse_horizon_s=impulse_horizon_s,
        parameters={"m_kg": m_kg, "c_Ns_per_m": c_ns_per_m, "k_N_per_m": k_n_per_m},
        derived={"omega_n_rad_s": float(omega_n), "zeta": float(zeta)},
        identifiable=("m_kg", "c_Ns_per_m", "k_N_per_m"),
    )


def get_plant(
    plant_id: str, systems_config: dict[str, SystemConfig] | None = None
) -> PlantModel:
    """Retrieve a plant model initialized from the configuration.

    Parameters
    ----------
    plant_id : str
        Target plant identifier ('rc', 'motor', or 'msd').
    systems_config : dict[str, SystemConfig] | None, optional
        Loaded systems configuration. If None, loaded from disk.

    Returns
    -------
    PlantModel
        Constructed continuous LTI plant model.
    """
    if systems_config is None:
        systems_config = load_systems()

    if plant_id == "rc":
        rc_cfg = systems_config["rc"]
        return create_rc_plant(
            tau_s=rc_cfg.derived["tau_s"],
            u_max=rc_cfg.u_max,
            dominant_time_scale_s=rc_cfg.dominant_time_scale_s,
            sequence_duration_s=rc_cfg.sequence_duration_s,
            impulse_horizon_s=rc_cfg.impulse_horizon_s,
            parameters=rc_cfg.parameters,
        )
    if plant_id == "motor":
        motor_cfg = systems_config["motor"]
        return create_motor_plant(
            k_gain=motor_cfg.derived["K_rad_s_per_V"],
            tau_s=motor_cfg.derived["tau_s"],
            u_max=motor_cfg.u_max,
            dominant_time_scale_s=motor_cfg.dominant_time_scale_s,
            sequence_duration_s=motor_cfg.sequence_duration_s,
            impulse_horizon_s=motor_cfg.impulse_horizon_s,
            parameters=motor_cfg.parameters,
        )
    if plant_id == "msd":
        msd_cfg = systems_config["msd"]
        return create_msd_plant(
            m_kg=msd_cfg.parameters["m_kg"],
            c_ns_per_m=msd_cfg.parameters["c_Ns_per_m"],
            k_n_per_m=msd_cfg.parameters["k_N_per_m"],
            u_max=msd_cfg.u_max,
            dominant_time_scale_s=msd_cfg.dominant_time_scale_s,
            sequence_duration_s=msd_cfg.sequence_duration_s,
            impulse_horizon_s=msd_cfg.impulse_horizon_s,
        )

    raise ValueError(
        f"Unknown plant id '{plant_id}'. Expected 'rc', 'motor', or 'msd'."
    )


def get_all_plants(
    systems_config: dict[str, SystemConfig] | None = None,
) -> dict[str, PlantModel]:
    """Retrieve all plant models defined in the specification.

    Parameters
    ----------
    systems_config : dict[str, SystemConfig] | None, optional
        Loaded systems configuration. If None, loaded from disk.

    Returns
    -------
    dict[str, PlantModel]
        Mapping from plant identifier to PlantModel instance.
    """
    if systems_config is None:
        systems_config = load_systems()
    return {
        plant_id: get_plant(plant_id, systems_config)
        for plant_id in ("rc", "motor", "msd")
    }


def extract_identifiable_parameters(
    plant_id: str, num: np.ndarray, den: np.ndarray
) -> dict[str, float]:
    """Extract physical identifiable parameters from a continuous transfer function.

    Parameters
    ----------
    plant_id : str
        Plant identifier ('rc', 'motor', or 'msd').
    num : np.ndarray
        Transfer function numerator polynomial coefficients.
    den : np.ndarray
        Transfer function denominator polynomial coefficients.

    Returns
    -------
    dict[str, float]
        Dictionary of identifiable parameters matching spec section 3.
    """
    # Normalize denominator so leading coefficient is 1.0
    den_lead = float(den[0])
    den_norm = den / den_lead
    num_norm = num / den_lead

    if plant_id == "rc":
        # H(s) = (1/tau) / (s + 1/tau)
        tau_s = float(1.0 / den_norm[1])
        return {"tau_s": tau_s}

    if plant_id == "motor":
        # H(s) = (K/tau) / (s + 1/tau)
        tau_s = float(1.0 / den_norm[1])
        # num_norm[-1] is K / tau, so K = num_norm[-1] * tau
        k_val = float(num_norm[-1] * tau_s)
        return {"K_rad_s_per_V": k_val, "tau_s": tau_s}

    if plant_id == "msd":
        # H(s) = (1/m) / (s^2 + (c/m)s + k/m)
        m_kg = float(1.0 / num_norm[-1])
        c_val = float(den_norm[1] * m_kg)
        k_val = float(den_norm[2] * m_kg)
        return {"m_kg": m_kg, "c_Ns_per_m": c_val, "k_N_per_m": k_val}

    raise ValueError(f"Unknown plant id '{plant_id}'.")
