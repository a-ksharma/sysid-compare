"""Health check and service status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import AppState, get_app_state
from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    """Return health status, API version, and loaded physical systems."""
    loaded_plants = sorted(list(state.plants.keys()))
    loaded_models: dict[str, list[str]] = {}

    for plant_id in loaded_plants:
        models = []
        if plant_id in state.classical_cards:
            models.extend(["arx", "prony"])
        if plant_id in state.neural_ode_models:
            models.append("neural_ode")
        loaded_models[plant_id] = models

    return HealthResponse(
        status="ok",
        version="0.1.0",
        loaded_plants=loaded_plants,
        loaded_models=loaded_models,
    )
