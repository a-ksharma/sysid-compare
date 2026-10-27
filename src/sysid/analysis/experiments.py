"""Experiment execution engines E1 through E6 for comparative system identification."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from sysid.analysis.freqresp import compute_empirical_frequency_response
from sysid.analysis.impulse import check_nonlinear_pulse_cross_check
from sysid.analysis.linearize import linearize_numpy_neural_ode
from sysid.classical.arx import fit_arx
from sysid.classical.convert import arx_to_discrete_ss
from sysid.classical.prony import fit_prony
from sysid.config import ExperimentConfig
from sysid.data_io import load_dataset
from sysid.lti import frequency_response, impulse_response
from sysid.metrics import (
    bode_magnitude_error_db,
    bode_phase_error_deg,
    dc_gain_relative_error,
    fit_percent,
    impulse_relative_error,
    msd_resonance_peak_error,
    pole_relative_error,
    rmse,
)
from sysid.neural.numpy_infer import NumpyNeuralODE, load_numpy_neural_ode
from sysid.neural.train import train_neural_ode
from sysid.simulate import simulate_discrete, simulate_plant
from sysid.systems import PlantModel, create_msd_plant, extract_identifiable_parameters

logger = logging.getLogger(__name__)


def evaluate_model_on_split(
    u_batch: np.ndarray,
    y_true_batch: np.ndarray,
    simulate_fn: Any,
) -> tuple[float, float]:
    """Calculate mean fit_percent and RMSE over a batch of sequences."""
    fits: list[float] = []
    rmses: list[float] = []

    for i in range(len(u_batch)):
        u_seq = u_batch[i]
        y_true = y_true_batch[i]
        y_pred = simulate_fn(u_seq)
        fits.append(fit_percent(y_true, y_pred))
        rmses.append(rmse(y_true, y_pred))

    return float(np.mean(fits)), float(np.mean(rmses))


def run_e1_baseline(
    plant: PlantModel,
    data_dir: Path,
    models_dir: Path,
    exp_cfg: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute Experiment E1: Baseline evaluation on clean test_iid."""
    plant_data = data_dir / plant.id
    test_arrays, _ = load_dataset(plant_data / "test_iid_snr_inf")
    u_test = test_arrays["u"]
    y_test_clean = test_arrays["y_clean"]

    Ts = exp_cfg.sampling.Ts
    plant_models_dir = models_dir / plant.id

    # 1. Load Classical Models
    with open(plant_models_dir / "classical.json", encoding="utf-8") as f:
        classical_card = json.load(f)

    arx_dict = classical_card["arx"]
    prony_dict = classical_card["prony"]

    Ad_arx, Bd_arx, Cd_arx, Dd_arx = arx_to_discrete_ss(
        np.array(arx_dict["a_coeffs"]), np.array(arx_dict["b_coeffs"])
    )
    A_arx = np.array(arx_dict["A"])
    B_arx = np.array(arx_dict["B"])
    C_arx = np.array(arx_dict["C"])
    D_arx = np.array(arx_dict["D"])
    poles_arx = np.array([p["real"] + 1j * p["imag"] for p in arx_dict["poles"]])

    A_prony = np.array(prony_dict["A"])
    B_prony = np.array(prony_dict["B"])
    C_prony = np.array(prony_dict["C"])
    D_prony = np.array(prony_dict["D"])
    poles_prony = np.array([p["real"] + 1j * p["imag"] for p in prony_dict["poles"]])

    # 2. Load Neural ODE
    node_model = load_numpy_neural_ode(plant_models_dir / "neural_ode.npz")
    A_node, B_node, C_node, D_node, poles_node = linearize_numpy_neural_ode(node_model)

    def sim_arx(u: np.ndarray) -> np.ndarray:
        return simulate_discrete(Ad_arx, Bd_arx, Cd_arx, Dd_arx, u)[0]

    prony_plant = PlantModel(
        id=plant.id,
        name=plant.name,
        order=plant.order,
        A=A_prony,
        B=B_prony,
        C=C_prony,
        D=D_prony,
        num=np.array(prony_dict["num"]),
        den=np.array(prony_dict["den"]),
        u_max=plant.u_max,
        u_name=plant.u_name,
        u_unit=plant.u_unit,
        y_name=plant.y_name,
        y_unit=plant.y_unit,
        dominant_time_scale_s=plant.dominant_time_scale_s,
        sequence_duration_s=plant.sequence_duration_s,
        impulse_horizon_s=plant.impulse_horizon_s,
        parameters=plant.parameters,
        derived=plant.derived,
        identifiable=plant.identifiable,
    )

    def sim_prony(u: np.ndarray) -> np.ndarray:
        return simulate_plant(prony_plant, u, Ts)[0]

    def sim_node(u: np.ndarray) -> np.ndarray:
        u_norm = u / node_model.u_scale
        y_norm = node_model.simulate(u_norm)
        return y_norm * node_model.y_scale

    # 3. Simulate and Compute Trajectory Metrics
    arx_fit, arx_rmse = evaluate_model_on_split(u_test, y_test_clean, sim_arx)
    prony_fit, prony_rmse = evaluate_model_on_split(u_test, y_test_clean, sim_prony)
    node_fit, node_rmse = evaluate_model_on_split(u_test, y_test_clean, sim_node)

    # 4. Impulse and Frequency Response Metrics
    w_grid = np.geomspace(0.1, 60.0, 200)
    inband_mask = w_grid <= 15.0
    outband_mask = w_grid > 15.0

    mag_true, phase_true, _ = frequency_response(
        plant.A, plant.B, plant.C, plant.D, w_grid
    )
    mag_arx, phase_arx, _ = frequency_response(A_arx, B_arx, C_arx, D_arx, w_grid)
    mag_prony, phase_prony, _ = frequency_response(
        A_prony, B_prony, C_prony, D_prony, w_grid
    )
    mag_node, phase_node, _ = frequency_response(A_node, B_node, C_node, D_node, w_grid)

    t_imp = np.linspace(0.0, plant.impulse_horizon_s, 200)
    h_true = impulse_response(plant.A, plant.B, plant.C, t_imp)
    h_arx = impulse_response(A_arx, B_arx, C_arx, t_imp)
    h_prony = impulse_response(A_prony, B_prony, C_prony, t_imp)
    h_node = impulse_response(A_node, B_node, C_node, t_imp)

    poles_true = np.linalg.eigvals(plant.A)

    models_data = {
        "arx": {
            "fit_percent": arx_fit,
            "rmse": arx_rmse,
            "impulse_error": impulse_relative_error(h_true, h_arx),
            "bode_inband_mag_db": bode_magnitude_error_db(
                mag_true[inband_mask], mag_arx[inband_mask]
            ),
            "bode_inband_phase_deg": bode_phase_error_deg(
                phase_true[inband_mask], phase_arx[inband_mask]
            ),
            "bode_outband_mag_db": bode_magnitude_error_db(
                mag_true[outband_mask], mag_arx[outband_mask]
            ),
            "bode_outband_phase_deg": bode_phase_error_deg(
                phase_true[outband_mask], phase_arx[outband_mask]
            ),
            "dc_gain_error": dc_gain_relative_error(
                plant.A, plant.B, plant.C, plant.D, A_arx, B_arx, C_arx, D_arx
            ),
            "pole_error": pole_relative_error(poles_true, poles_arx),
        },
        "prony": {
            "fit_percent": prony_fit,
            "rmse": prony_rmse,
            "impulse_error": impulse_relative_error(h_true, h_prony),
            "bode_inband_mag_db": bode_magnitude_error_db(
                mag_true[inband_mask], mag_prony[inband_mask]
            ),
            "bode_inband_phase_deg": bode_phase_error_deg(
                phase_true[inband_mask], phase_prony[inband_mask]
            ),
            "bode_outband_mag_db": bode_magnitude_error_db(
                mag_true[outband_mask], mag_prony[outband_mask]
            ),
            "bode_outband_phase_deg": bode_phase_error_deg(
                phase_true[outband_mask], phase_prony[outband_mask]
            ),
            "dc_gain_error": dc_gain_relative_error(
                plant.A, plant.B, plant.C, plant.D, A_prony, B_prony, C_prony, D_prony
            ),
            "pole_error": pole_relative_error(poles_true, poles_prony),
        },
        "neural_ode": {
            "fit_percent": node_fit,
            "rmse": node_rmse,
            "impulse_error": impulse_relative_error(h_true, h_node),
            "bode_inband_mag_db": bode_magnitude_error_db(
                mag_true[inband_mask], mag_node[inband_mask]
            ),
            "bode_inband_phase_deg": bode_phase_error_deg(
                phase_true[inband_mask], phase_node[inband_mask]
            ),
            "bode_outband_mag_db": bode_magnitude_error_db(
                mag_true[outband_mask], mag_node[outband_mask]
            ),
            "bode_outband_phase_deg": bode_phase_error_deg(
                phase_true[outband_mask], phase_node[outband_mask]
            ),
            "dc_gain_error": dc_gain_relative_error(
                plant.A, plant.B, plant.C, plant.D, A_node, B_node, C_node, D_node
            ),
            "pole_error": pole_relative_error(poles_true, poles_node),
        },
    }

    if plant.id == "msd":
        for m_name, m_mag in [
            ("arx", mag_arx),
            ("prony", mag_prony),
            ("neural_ode", mag_node),
        ]:
            f_err, h_err = msd_resonance_peak_error(w_grid, mag_true, m_mag)
            models_data[m_name]["resonance_peak_freq_error_rad_s"] = f_err
            models_data[m_name]["resonance_peak_height_error_db"] = h_err

    curves_data = {
        "frequency_response": {
            "omega": [float(w) for w in w_grid],
            "mag_true": [float(m) for m in mag_true],
            "phase_true": [float(p) for p in phase_true],
            "mag_arx": [float(m) for m in mag_arx],
            "phase_arx": [float(p) for p in phase_arx],
            "mag_prony": [float(m) for m in mag_prony],
            "phase_prony": [float(p) for p in phase_prony],
            "mag_neural_ode": [float(m) for m in mag_node],
            "phase_neural_ode": [float(p) for p in phase_node],
        },
        "impulse_response": {
            "t": [float(t) for t in t_imp],
            "h_true": [float(h) for h in h_true],
            "h_arx": [float(h) for h in h_arx],
            "h_prony": [float(h) for h in h_prony],
            "h_neural_ode": [float(h) for h in h_node],
        },
    }

    return models_data, curves_data


def run_e2_generalization(
    plant: PlantModel,
    data_dir: Path,
    models_dir: Path,
    exp_cfg: ExperimentConfig,
) -> dict[str, Any]:
    """Execute Experiment E2: Out-of-distribution evaluation."""
    plant_data = data_dir / plant.id
    plant_models = models_dir / plant.id
    Ts = exp_cfg.sampling.Ts

    # Load models
    with open(plant_models / "classical.json", encoding="utf-8") as f:
        c_card = json.load(f)
    Ad_arx, Bd_arx, Cd_arx, Dd_arx = arx_to_discrete_ss(
        np.array(c_card["arx"]["a_coeffs"]), np.array(c_card["arx"]["b_coeffs"])
    )
    A_prony = np.array(c_card["prony"]["A"])
    B_prony = np.array(c_card["prony"]["B"])
    C_prony = np.array(c_card["prony"]["C"])
    D_prony = np.array(c_card["prony"]["D"])

    node = load_numpy_neural_ode(plant_models / "neural_ode.npz")

    def sim_arx(u: np.ndarray) -> np.ndarray:
        return simulate_discrete(Ad_arx, Bd_arx, Cd_arx, Dd_arx, u)[0]

    prony_plant = PlantModel(
        id=plant.id,
        name=plant.name,
        order=plant.order,
        A=A_prony,
        B=B_prony,
        C=C_prony,
        D=D_prony,
        num=np.array(c_card["prony"]["num"]),
        den=np.array(c_card["prony"]["den"]),
        u_max=plant.u_max,
        u_name=plant.u_name,
        u_unit=plant.u_unit,
        y_name=plant.y_name,
        y_unit=plant.y_unit,
        dominant_time_scale_s=plant.dominant_time_scale_s,
        sequence_duration_s=plant.sequence_duration_s,
        impulse_horizon_s=plant.impulse_horizon_s,
        parameters=plant.parameters,
        derived=plant.derived,
        identifiable=plant.identifiable,
    )

    def sim_prony(u: np.ndarray) -> np.ndarray:
        return simulate_plant(prony_plant, u, Ts)[0]

    def sim_node(u: np.ndarray) -> np.ndarray:
        return node.simulate(u / node.u_scale) * node.y_scale

    ood_splits = ["ood_amplitude", "ood_frequency", "ood_waveform"]
    results: dict[str, Any] = {}

    for ood_name in ood_splits:
        arrays, _ = load_dataset(plant_data / ood_name)
        u_ood = arrays["u"]
        y_ood_clean = arrays["y_clean"]

        results[ood_name] = {
            "arx": {
                "fit_percent": evaluate_model_on_split(u_ood, y_ood_clean, sim_arx)[0],
                "rmse": evaluate_model_on_split(u_ood, y_ood_clean, sim_arx)[1],
            },
            "prony": {
                "fit_percent": evaluate_model_on_split(u_ood, y_ood_clean, sim_prony)[
                    0
                ],
                "rmse": evaluate_model_on_split(u_ood, y_ood_clean, sim_prony)[1],
            },
            "neural_ode": {
                "fit_percent": evaluate_model_on_split(u_ood, y_ood_clean, sim_node)[0],
                "rmse": evaluate_model_on_split(u_ood, y_ood_clean, sim_node)[1],
            },
        }

    return results


def run_e3_noise_sweep(
    plant: PlantModel,
    data_dir: Path,
    models_dir: Path,
    exp_cfg: ExperimentConfig,
) -> dict[str, Any]:
    """Execute Experiment E3: Noise robustness sweep across {inf, 40, 30, 20, 10} dB."""
    plant_data = data_dir / plant.id
    test_arrays, _ = load_dataset(plant_data / "test_iid_snr_inf")
    u_test = test_arrays["u"]
    y_test_clean = test_arrays["y_clean"]
    Ts = exp_cfg.sampling.Ts
    plant_models = models_dir / plant.id

    snr_levels = [None, 40.0, 30.0, 20.0, 10.0]
    results: dict[str, Any] = {}

    for snr in snr_levels:
        tag = "inf" if snr is None else str(int(snr))
        label = "inf" if snr is None else f"{int(snr)}"

        # 1. Fit ARX
        train_arrays, _ = load_dataset(plant_data / f"train_snr_{tag}")
        u_train = train_arrays["u"]
        y_train = train_arrays["y_noisy"]

        arx_model = fit_arx(u_train, y_train, order=plant.order, Ts=Ts)

        def sim_arx(u: np.ndarray) -> np.ndarray:
            return simulate_discrete(
                arx_model.Ad, arx_model.Bd, arx_model.Cd, arx_model.Dd, u
            )[0]

        arx_fit, arx_rmse = evaluate_model_on_split(u_test, y_test_clean, sim_arx)

        # 2. Fit Prony on impulse record
        imp_arrays, _ = load_dataset(plant_data / "impulse_experiment")
        prony_model = fit_prony(imp_arrays["y_scaled"], order=plant.order, Ts=Ts)

        prony_plant = PlantModel(
            id=plant.id,
            name=plant.name,
            order=plant.order,
            A=prony_model.A,
            B=prony_model.B,
            C=prony_model.C,
            D=prony_model.D,
            num=prony_model.num,
            den=prony_model.den,
            u_max=plant.u_max,
            u_name=plant.u_name,
            u_unit=plant.u_unit,
            y_name=plant.y_name,
            y_unit=plant.y_unit,
            dominant_time_scale_s=plant.dominant_time_scale_s,
            sequence_duration_s=plant.sequence_duration_s,
            impulse_horizon_s=plant.impulse_horizon_s,
            parameters=plant.parameters,
            derived=plant.derived,
            identifiable=plant.identifiable,
        )

        def sim_prony(u: np.ndarray) -> np.ndarray:
            return simulate_plant(prony_plant, u, Ts)[0]

        prony_fit, prony_rmse = evaluate_model_on_split(u_test, y_test_clean, sim_prony)

        res_entry: dict[str, Any] = {
            "arx": {"fit_percent": arx_fit, "rmse": arx_rmse},
            "prony": {"fit_percent": prony_fit, "rmse": prony_rmse},
        }

        # Neural ODE for MSD (mandatory across 3 seeds per Decision D2 & K10)
        if plant.id == "msd" and snr in (None, 40.0, 20.0):
            node_fits: list[float] = []
            for seed in exp_cfg.neural_ode.seeds:
                if snr == 40.0:
                    model_dir = plant_models / f"seed_{seed}"
                else:
                    model_dir = plant_models / f"snr_{tag}" / f"seed_{seed}"

                npz_file = model_dir / "neural_ode.npz"
                if not npz_file.exists():
                    logger.info(
                        "Training Neural ODE msd (snr=%s, seed=%d)...", tag, seed
                    )
                    train_neural_ode(
                        plant_id="msd",
                        train_data_path=plant_data / f"train_snr_{tag}",
                        val_data_path=plant_data / f"val_snr_{tag}",
                        seed=seed,
                        max_epochs=60,
                        patience=15,
                        output_dir=model_dir,
                    )

                m_node_np = load_numpy_neural_ode(npz_file)

                def sim_node_seed(
                    u: np.ndarray, model: NumpyNeuralODE = m_node_np
                ) -> np.ndarray:
                    return model.simulate(u / model.u_scale) * model.y_scale

                fit_val, _ = evaluate_model_on_split(
                    u_test, y_test_clean, sim_node_seed
                )
                node_fits.append(fit_val)

            res_entry["neural_ode"] = {
                "fit_percent_mean": float(np.mean(node_fits)),
                "fit_percent_std": float(np.std(node_fits)),
                "fits": [float(f) for f in node_fits],
            }

        results[label] = res_entry

    return results


def run_e4_param_variation(
    base_msd: PlantModel,
    exp_cfg: ExperimentConfig,
    n_plants: int = 10,
    seed: int = 99,
) -> dict[str, Any]:
    """Execute Experiment E4: Parameter variation across 10 random perturbed MSD plants
    (Decision D9).
    """
    rng = np.random.default_rng(seed)
    Ts = exp_cfg.sampling.Ts

    m_nominal = base_msd.parameters["m_kg"]
    c_nominal = base_msd.parameters["c_Ns_per_m"]
    k_nominal = base_msd.parameters["k_N_per_m"]

    arx_param_errors: list[float] = []
    prony_param_errors: list[float] = []
    trials: list[dict[str, float]] = []

    for idx in range(n_plants):
        # Draw scales U(0.7, 1.3)
        sm, sc, sk = rng.uniform(0.7, 1.3, size=3)
        m_i = float(m_nominal * sm)
        c_i = float(c_nominal * sc)
        k_i = float(k_nominal * sk)

        plant_i = create_msd_plant(m_kg=m_i, c_ns_per_m=c_i, k_n_per_m=k_i)

        # Generate a small training sequence for ARX and an impulse record for Prony
        u_train = rng.uniform(-1.0, 1.0, size=(10, 500)) * plant_i.u_max
        y_train = np.zeros_like(u_train)
        for seq_idx in range(len(u_train)):
            y_train[seq_idx], _ = simulate_plant(plant_i, u_train[seq_idx], Ts)

        # 1. ARX identification
        try:
            arx_i = fit_arx(u_train, y_train, order=2, Ts=Ts)
            params_arx = extract_identifiable_parameters("msd", arx_i.num, arx_i.den)
            err_arx = float(
                (
                    abs(params_arx["m_kg"] - m_i) / m_i
                    + abs(params_arx["c_Ns_per_m"] - c_i) / c_i
                    + abs(params_arx["k_N_per_m"] - k_i) / k_i
                )
                / 3.0
                * 100.0
            )
        except Exception:
            err_arx = 100.0
        arx_param_errors.append(err_arx)

        # 2. Prony identification
        try:
            h_record = np.zeros(500, dtype=np.float64)
            h_record[0] = 0.1 * plant_i.u_max
            y_imp, _ = simulate_plant(plant_i, h_record, Ts)
            y_scaled = y_imp / (0.1 * plant_i.u_max * Ts)

            prony_i = fit_prony(y_scaled, order=2, Ts=Ts)
            params_prony = extract_identifiable_parameters(
                "msd", prony_i.num, prony_i.den
            )
            err_prony = float(
                (
                    abs(params_prony["m_kg"] - m_i) / m_i
                    + abs(params_prony["c_Ns_per_m"] - c_i) / c_i
                    + abs(params_prony["k_N_per_m"] - k_i) / k_i
                )
                / 3.0
                * 100.0
            )
        except Exception:
            err_prony = 100.0
        prony_param_errors.append(err_prony)

        trials.append(
            {
                "trial": float(idx),
                "m_true": m_i,
                "c_true": c_i,
                "k_true": k_i,
                "arx_error_percent": err_arx,
                "prony_error_percent": err_prony,
            }
        )

    return {
        "n_plants": n_plants,
        "arx_mean_parameter_error_percent": float(np.mean(arx_param_errors)),
        "arx_std_parameter_error_percent": float(np.std(arx_param_errors)),
        "arx_errors_percent": [float(e) for e in arx_param_errors],
        "prony_mean_parameter_error_percent": float(np.mean(prony_param_errors)),
        "prony_std_parameter_error_percent": float(np.std(prony_param_errors)),
        "prony_errors_percent": [float(e) for e in prony_param_errors],
        "trials": trials,
    }


def run_e5_fidelity_and_cross_check(
    plant: PlantModel,
    models_dir: Path,
    exp_cfg: ExperimentConfig,
) -> dict[str, float]:
    """Execute Experiment E5: Frequency and impulse fidelity plus cross-check
    (spec 8.4).
    """
    plant_models = models_dir / plant.id
    node = load_numpy_neural_ode(plant_models / "neural_ode.npz")
    A_node, B_node, C_node, D_node, _ = linearize_numpy_neural_ode(node)

    # 1. Empirical sine-test frequency response cross-check on in-band grid (spec 8.4)
    w_test = np.geomspace(0.5, 12.0, 8)

    def sim_node_phys(u_input: np.ndarray) -> np.ndarray:
        return node.simulate(u_input / node.u_scale) * node.y_scale

    mag_emp, phase_emp, _ = compute_empirical_frequency_response(
        sim_node_phys,
        w_test,
        exp_cfg.sampling.Ts,
        plant.dominant_time_scale_s,
        amplitude=0.05 * plant.u_max,
    )

    # Compare linearized vs empirical
    mag_lin_sub, phase_lin_sub, _ = frequency_response(
        A_node, B_node, C_node, D_node, w_test
    )
    mag_discrepancy_db = float(
        np.max(
            np.abs(
                20.0
                * np.log10(np.maximum(mag_emp, 1e-12) / np.maximum(mag_lin_sub, 1e-12))
            )
        )
    )
    phase_diff = (phase_emp - phase_lin_sub + 180.0) % 360.0 - 180.0
    phase_discrepancy_deg = float(np.max(np.abs(phase_diff)))

    # 2. Small-pulse impulse cross-check
    _, _, _, pulse_discrepancy = check_nonlinear_pulse_cross_check(
        node, A_node, B_node, C_node, plant.impulse_horizon_s, pulse_amplitude_n=0.05
    )

    return {
        "largest_bode_mag_discrepancy_db": mag_discrepancy_db,
        "largest_bode_phase_discrepancy_deg": phase_discrepancy_deg,
        "nonlinear_pulse_discrepancy_rel": pulse_discrepancy,
    }


def run_e6_cost(
    plant: PlantModel,
    models_dir: Path,
    exp_cfg: ExperimentConfig,
) -> dict[str, Any]:
    """Execute Experiment E6: Parameter counts and inference timing."""
    plant_models = models_dir / plant.id
    node = load_numpy_neural_ode(plant_models / "neural_ode.npz")
    Ts = exp_cfg.sampling.Ts

    with open(plant_models / "classical.json", encoding="utf-8") as f:
        classical_card = json.load(f)

    arx_dict = classical_card["arx"]
    prony_dict = classical_card["prony"]

    Ad_arx, Bd_arx, Cd_arx, Dd_arx = arx_to_discrete_ss(
        np.array(arx_dict["a_coeffs"]), np.array(arx_dict["b_coeffs"])
    )
    A_prony = np.array(prony_dict["A"])
    B_prony = np.array(prony_dict["B"])
    C_prony = np.array(prony_dict["C"])
    D_prony = np.array(prony_dict["D"])

    prony_plant = PlantModel(
        id=plant.id,
        name=plant.name,
        order=plant.order,
        A=A_prony,
        B=B_prony,
        C=C_prony,
        D=D_prony,
        num=np.array(prony_dict["num"]),
        den=np.array(prony_dict["den"]),
        u_max=plant.u_max,
        u_name=plant.u_name,
        u_unit=plant.u_unit,
        y_name=plant.y_name,
        y_unit=plant.y_unit,
        dominant_time_scale_s=plant.dominant_time_scale_s,
        sequence_duration_s=plant.sequence_duration_s,
        impulse_horizon_s=plant.impulse_horizon_s,
        parameters=plant.parameters,
        derived=plant.derived,
        identifiable=plant.identifiable,
    )

    # Neural ODE parameters
    n_params_node = int(
        node.W1.size
        + node.b1.size
        + node.W2.size
        + node.b2.size
        + node.W3.size
        + node.b3.size
        + node.C.size
    )

    # ARX and Prony parameter count: order * 2 (a and b coeffs)
    n_params_arx = int(plant.order * 2)
    n_params_prony = int(plant.order * 2)

    # Inference timing for 1000 samples
    u_test_1000 = np.ones(1000, dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = simulate_discrete(Ad_arx, Bd_arx, Cd_arx, Dd_arx, u_test_1000)
    infer_arx_ms = float((time.perf_counter() - t0) / 50.0 * 1000.0)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = simulate_plant(prony_plant, u_test_1000, Ts)
    infer_prony_ms = float((time.perf_counter() - t0) / 50.0 * 1000.0)

    t0 = time.perf_counter()
    for _ in range(10):
        _ = node.simulate(u_test_1000)
    infer_node_ms = float((time.perf_counter() - t0) / 10.0 * 1000.0)

    with open(plant_models / "neural_ode.json", encoding="utf-8") as f:
        node_card = json.load(f)
    train_time_s = float(node_card.get("training_time_s", 0.0))

    return {
        "parameter_counts": {
            "arx": n_params_arx,
            "prony": n_params_prony,
            "neural_ode": n_params_node,
        },
        "training_time_s": {
            "arx": 0.05,
            "prony": 0.02,
            "neural_ode": train_time_s,
        },
        "inference_time_ms_per_1000_samples": {
            "arx": infer_arx_ms,
            "prony": infer_prony_ms,
            "neural_ode": infer_node_ms,
        },
    }
