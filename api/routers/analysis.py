"""Frequency response, impulse response, and evaluation results endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.deps import AppState, get_app_state
from api.schemas import (
    BodeCurve,
    FrequencyResponseData,
    ImpulseResponseData,
)
from sysid.analysis.linearize import linearize_numpy_neural_ode
from sysid.lti import frequency_response, impulse_response

router = APIRouter(tags=["Analysis"])


@router.get(
    "/frequency-response/{system_id}",
    response_model=FrequencyResponseData,
)
async def get_frequency_response(
    system_id: str,
    models: list[str] = Query(default=["arx", "prony", "neural_ode"]),
    omega_min: float = Query(default=0.1, gt=0.0),
    omega_max: float = Query(default=60.0, gt=0.0),
    points: int = Query(default=200, ge=2, le=500),
    state: AppState = Depends(get_app_state),
) -> FrequencyResponseData:
    """Compute continuous frequency response (Bode data) for plant and models."""
    if system_id not in state.plants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PLANT_NOT_FOUND",
                "message": f"Plant '{system_id}' not found.",
            },
        )

    if omega_min >= omega_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "INVALID_FREQUENCY_RANGE",
                "message": (
                    f"omega_min ({omega_min}) must be less than "
                    f"omega_max ({omega_max})."
                ),
            },
        )

    plant = state.plants[system_id]
    omega = np.geomspace(omega_min, omega_max, points)

    # 1. True system
    mag_true, phase_true, _ = frequency_response(
        plant.A, plant.B, plant.C, plant.D, omega
    )
    db_true = 20.0 * np.log10(np.maximum(mag_true, 1e-12))
    true_curve = BodeCurve(
        magnitude_db=[float(v) for v in db_true],
        phase_deg=[float(v) for v in phase_true],
    )

    # 2. Identified models
    c_card = state.classical_cards.get(system_id, {})
    models_curves: dict[str, BodeCurve] = {}

    for m_name in models:
        if m_name == "arx" and "arx" in c_card:
            arx_dict = c_card["arx"]
            A_arx = np.array(arx_dict["A"])
            B_arx = np.array(arx_dict["B"])
            C_arx = np.array(arx_dict["C"])
            D_arx = np.array(arx_dict["D"])
            mag_m, phase_m, _ = frequency_response(A_arx, B_arx, C_arx, D_arx, omega)
            models_curves["arx"] = BodeCurve(
                magnitude_db=[
                    float(v) for v in 20.0 * np.log10(np.maximum(mag_m, 1e-12))
                ],
                phase_deg=[float(v) for v in phase_m],
            )
        elif m_name == "prony" and "prony" in c_card:
            prony_dict = c_card["prony"]
            A_p = np.array(prony_dict["A"])
            B_p = np.array(prony_dict["B"])
            C_p = np.array(prony_dict["C"])
            D_p = np.array(prony_dict["D"])
            mag_m, phase_m, _ = frequency_response(A_p, B_p, C_p, D_p, omega)
            models_curves["prony"] = BodeCurve(
                magnitude_db=[
                    float(v) for v in 20.0 * np.log10(np.maximum(mag_m, 1e-12))
                ],
                phase_deg=[float(v) for v in phase_m],
            )
        elif m_name == "neural_ode" and system_id in state.neural_ode_models:
            node = state.neural_ode_models[system_id]
            A_n, B_n, C_n, D_n, _ = linearize_numpy_neural_ode(node)
            mag_m, phase_m, _ = frequency_response(A_n, B_n, C_n, D_n, omega)
            models_curves["neural_ode"] = BodeCurve(
                magnitude_db=[
                    float(v) for v in 20.0 * np.log10(np.maximum(mag_m, 1e-12))
                ],
                phase_deg=[float(v) for v in phase_m],
            )

    return FrequencyResponseData(
        omega=[float(w) for w in omega],
        true_system=true_curve,
        models=models_curves,
    )


@router.get(
    "/impulse-response/{system_id}",
    response_model=ImpulseResponseData,
)
async def get_impulse_response(
    system_id: str,
    models: list[str] = Query(default=["arx", "prony", "neural_ode"]),
    horizon_s: float | None = Query(default=None, gt=0.0),
    points: int = Query(default=200, ge=10, le=1000),
    state: AppState = Depends(get_app_state),
) -> ImpulseResponseData:
    """Compute continuous impulse response h(t) for plant and models."""
    if system_id not in state.plants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PLANT_NOT_FOUND",
                "message": f"Plant '{system_id}' not found.",
            },
        )

    plant = state.plants[system_id]
    h_max = horizon_s if horizon_s is not None else plant.impulse_horizon_s
    t = np.linspace(0.0, h_max, points)

    h_true = impulse_response(plant.A, plant.B, plant.C, t)
    c_card = state.classical_cards.get(system_id, {})
    models_h: dict[str, list[float]] = {}

    for m_name in models:
        if m_name == "arx" and "arx" in c_card:
            arx_dict = c_card["arx"]
            A_arx = np.array(arx_dict["A"])
            B_arx = np.array(arx_dict["B"])
            C_arx = np.array(arx_dict["C"])
            h_m = impulse_response(A_arx, B_arx, C_arx, t)
            models_h["arx"] = [float(v) for v in h_m]
        elif m_name == "prony" and "prony" in c_card:
            prony_dict = c_card["prony"]
            A_p = np.array(prony_dict["A"])
            B_p = np.array(prony_dict["B"])
            C_p = np.array(prony_dict["C"])
            h_m = impulse_response(A_p, B_p, C_p, t)
            models_h["prony"] = [float(v) for v in h_m]
        elif m_name == "neural_ode" and system_id in state.neural_ode_models:
            node = state.neural_ode_models[system_id]
            A_n, B_n, C_n, _, _ = linearize_numpy_neural_ode(node)
            h_m = impulse_response(A_n, B_n, C_n, t)
            models_h["neural_ode"] = [float(v) for v in h_m]

    return ImpulseResponseData(
        t=[float(v) for v in t],
        true_system=[float(v) for v in h_true],
        models=models_h,
    )


@router.get("/results/{system_id}")
async def get_results(
    system_id: str, state: AppState = Depends(get_app_state)
) -> dict[str, Any]:
    """Retrieve precomputed evaluation.json experiment results."""
    if system_id not in state.evaluation_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "RESULTS_NOT_FOUND",
                "message": f"Precomputed results for plant '{system_id}' not found.",
            },
        )
    return state.evaluation_results[system_id]
