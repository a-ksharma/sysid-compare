"""Plant system catalog and parameter detail endpoints."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import AppState, get_app_state
from api.schemas import PoleComplex, SystemDetail, SystemSummary

router = APIRouter(prefix="/systems", tags=["Systems"])


@router.get("", response_model=list[SystemSummary])
async def list_systems(state: AppState = Depends(get_app_state)) -> list[SystemSummary]:
    """List all available physical plants with nominal specifications."""
    summaries: list[SystemSummary] = []
    for plant_id in sorted(state.plants.keys()):
        plant = state.plants[plant_id]
        summaries.append(
            SystemSummary(
                id=plant.id,
                name=plant.name,
                order=plant.order,
                nominal_parameters=dict(plant.parameters),
                u_name=plant.u_name,
                u_unit=plant.u_unit,
                y_name=plant.y_name,
                y_unit=plant.y_unit,
                u_max=plant.u_max,
                training_band_rad_s=[0.1, 15.0],
            )
        )
    return summaries


@router.get("/{system_id}", response_model=SystemDetail)
async def get_system(
    system_id: str, state: AppState = Depends(get_app_state)
) -> SystemDetail:
    """Retrieve detailed true and identified parameters for a specific plant."""
    if system_id not in state.plants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PLANT_NOT_FOUND",
                "message": f"Plant '{system_id}' not found.",
            },
        )

    plant = state.plants[system_id]
    true_eigs = np.linalg.eigvals(plant.A)
    true_poles = [
        PoleComplex(real=float(p.real), imag=float(p.imag)) for p in true_eigs
    ]

    identified: dict[str, dict] = {}

    # Classical (ARX and Prony)
    c_card = state.classical_cards.get(system_id)
    if c_card:
        arx_data = c_card.get("arx", {})
        prony_data = c_card.get("prony", {})
        identified["arx"] = {
            "parameters": arx_data.get("parameters", {}),
            "poles": arx_data.get("poles", []),
        }
        identified["prony"] = {
            "parameters": prony_data.get("parameters", {}),
            "poles": prony_data.get("poles", []),
        }

    # Neural ODE
    node_card = state.neural_ode_cards.get(system_id)
    if node_card:
        lin_data = node_card.get("linearization", {})
        identified["neural_ode"] = {
            "poles": lin_data.get("poles", []),
        }

    return SystemDetail(
        id=plant.id,
        name=plant.name,
        order=plant.order,
        nominal_parameters=dict(plant.parameters),
        u_name=plant.u_name,
        u_unit=plant.u_unit,
        y_name=plant.y_name,
        y_unit=plant.y_unit,
        u_max=plant.u_max,
        training_band_rad_s=[0.1, 15.0],
        true_poles=true_poles,
        identified=identified,
    )
