"""Trajectory simulation endpoint driving physical plants and identified models."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import AppState, get_app_state
from api.schemas import ModelPrediction, SimulateRequest, SimulateResponse
from sysid.classical.convert import arx_to_discrete_ss
from sysid.metrics import fit_percent, rmse
from sysid.signals import (
    generate_chirp,
    generate_multisine,
    generate_sine,
    generate_square,
    generate_step,
)
from sysid.simulate import add_output_noise, simulate_discrete, simulate_plant
from sysid.systems import PlantModel

router = APIRouter(prefix="/simulate", tags=["Simulation"])


@router.post("", response_model=SimulateResponse)
async def simulate(
    req: SimulateRequest, state: AppState = Depends(get_app_state)
) -> SimulateResponse:
    """Simulate true physical plant and identified models under custom inputs."""
    # 1. Validate system
    if req.system_id not in state.plants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PLANT_NOT_FOUND",
                "message": f"Plant '{req.system_id}' not found.",
            },
        )
    plant = state.plants[req.system_id]
    Ts = state.exp_cfg.sampling.Ts if state.exp_cfg else 0.02

    # 2. Check sample limit (max 3000 samples)
    n_samples = int(round(req.duration_s / Ts))
    if n_samples > 3000 or n_samples < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "SAMPLE_LIMIT_EXCEEDED",
                "message": (
                    f"Requested duration results in {n_samples} samples. "
                    "Maximum allowed is 3000."
                ),
            },
        )

    # 3. Check amplitude limits (max 3 x u_max)
    max_allowed_amp = 3.0 * plant.u_max
    if abs(req.input.amplitude) > max_allowed_amp:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "AMPLITUDE_EXCEEDED",
                "message": (
                    f"Requested amplitude {abs(req.input.amplitude):.2f} exceeds "
                    f"the maximum limit of 3x u_max "
                    f"({max_allowed_amp:.2f} {plant.u_unit})."
                ),
            },
        )

    # 4. Generate warnings for out-of-range inputs
    warnings: list[str] = []
    if abs(req.input.amplitude) > plant.u_max:
        ratio = abs(req.input.amplitude) / plant.u_max
        warnings.append(
            f"Input amplitude {req.input.amplitude:.2f} {plant.u_unit} exceeds nominal "
            f"training range u_max={plant.u_max:.2f} {plant.u_unit} "
            f"(ratio: {ratio:.2f}x)."
        )

    if req.input.frequency_rad_s is not None and req.input.frequency_rad_s > 15.0:
        ratio = req.input.frequency_rad_s / 15.0
        warnings.append(
            f"Input frequency {req.input.frequency_rad_s:.2f} rad/s exceeds nominal "
            f"training band limit (15.00 rad/s, ratio: {ratio:.2f}x)."
        )

    # 5. Synthesize input signal u
    t_arr = np.arange(n_samples, dtype=np.float64) * Ts
    kind = req.input.kind
    amp = req.input.amplitude
    freq = req.input.frequency_rad_s or 2.0

    rng = np.random.default_rng(req.seed)
    if kind == "step":
        u_arr = generate_step(req.duration_s, Ts, amplitude=amp, quiet_lead_s=0.0)
    elif kind == "sine":
        u_arr = generate_sine(
            req.duration_s, Ts, amplitude=amp, frequency_rad_s=freq, quiet_lead_s=0.0
        )
    elif kind == "square":
        period = (2.0 * np.pi / freq) if freq > 1e-6 else 1.0
        u_arr = generate_square(
            req.duration_s, Ts, amplitude=amp, period_s=period, quiet_lead_s=0.0
        )
    elif kind == "chirp":
        omega_max_chirp = req.input.frequency_rad_s or 15.0
        u_arr = generate_chirp(
            req.duration_s,
            Ts,
            u_max=amp,
            rng=rng,
            quiet_lead_s=0.0,
            omega_range_rad_s=(0.1, omega_max_chirp),
        )
    elif kind == "multisine":
        u_arr = generate_multisine(
            req.duration_s,
            Ts,
            u_max=amp,
            rng=rng,
            quiet_lead_s=0.0,
            n_components=12,
            omega_range_rad_s=(0.1, 15.0),
        )
    elif kind == "custom":
        if not req.input.samples:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "MISSING_CUSTOM_SAMPLES",
                    "message": "Custom input kind requires 'samples' array.",
                },
            )
        if len(req.input.samples) != n_samples:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "INVALID_SAMPLE_LENGTH",
                    "message": (
                        f"Expected {n_samples} samples, got {len(req.input.samples)}."
                    ),
                },
            )
        u_arr = np.array(req.input.samples, dtype=np.float64)
        if np.max(np.abs(u_arr)) > max_allowed_amp:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "AMPLITUDE_EXCEEDED",
                    "message": (
                        "Custom samples contain values exceeding "
                        f"{max_allowed_amp:.2f}."
                    ),
                },
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "UNKNOWN_INPUT_KIND",
                "message": f"Unsupported input kind '{kind}'.",
            },
        )

    # 6. Simulate true plant
    y_clean, _ = simulate_plant(plant, u_arr, Ts)

    # 7. Add noise if requested
    if req.noise_snr_db is not None:
        rng = np.random.default_rng(req.seed)
        y_measured_arr, _ = add_output_noise(y_clean, snr_db=req.noise_snr_db, rng=rng)
        y_measured = [float(v) for v in y_measured_arr]
    else:
        y_measured = None

    # 8. Simulate requested models
    predictions: dict[str, ModelPrediction] = {}
    classical_card = state.classical_cards.get(req.system_id, {})

    for model_name in req.models:
        if model_name == "arx" and "arx" in classical_card:
            arx_dict = classical_card["arx"]
            Ad, Bd, Cd, Dd = arx_to_discrete_ss(
                np.array(arx_dict["a_coeffs"]), np.array(arx_dict["b_coeffs"])
            )
            y_pred, _ = simulate_discrete(Ad, Bd, Cd, Dd, u_arr)
            predictions["arx"] = ModelPrediction(
                y=[float(v) for v in y_pred],
                fit_percent=fit_percent(y_clean, y_pred),
                rmse=rmse(y_clean, y_pred),
            )
        elif model_name == "prony" and "prony" in classical_card:
            prony_dict = classical_card["prony"]
            prony_plant = PlantModel(
                id=plant.id,
                name=plant.name,
                order=plant.order,
                A=np.array(prony_dict["A"]),
                B=np.array(prony_dict["B"]),
                C=np.array(prony_dict["C"]),
                D=np.array(prony_dict["D"]),
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
            y_pred, _ = simulate_plant(prony_plant, u_arr, Ts)
            predictions["prony"] = ModelPrediction(
                y=[float(v) for v in y_pred],
                fit_percent=fit_percent(y_clean, y_pred),
                rmse=rmse(y_clean, y_pred),
            )
        elif model_name == "neural_ode" and req.system_id in state.neural_ode_models:
            node = state.neural_ode_models[req.system_id]
            u_norm = (u_arr / node.u_scale).astype(np.float32)
            y_norm = node.simulate(u_norm)
            y_pred = y_norm * node.y_scale
            predictions["neural_ode"] = ModelPrediction(
                y=[float(v) for v in y_pred],
                fit_percent=fit_percent(y_clean, y_pred),
                rmse=rmse(y_clean, y_pred),
            )

    return SimulateResponse(
        ts=Ts,
        t=[float(v) for v in t_arr],
        u=[float(v) for v in u_arr],
        y_true=[float(v) for v in y_clean],
        y_measured=y_measured,
        predictions=predictions,
        warnings=warnings,
    )
