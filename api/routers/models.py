"""Model metadata and model card retrieval endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import AppState, get_app_state
from api.schemas import ModelCardResponse

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelCardResponse)
async def get_model_cards(
    state: AppState = Depends(get_app_state),
) -> ModelCardResponse:
    """Retrieve training metadata, parameter counts, and provenance for all models."""
    cards: dict[str, Any] = {}
    for plant_id in sorted(state.plants.keys()):
        cards[plant_id] = {
            "classical": state.classical_cards.get(plant_id, {}),
            "neural_ode": state.neural_ode_cards.get(plant_id, {}),
        }

    return ModelCardResponse(cards=cards)


@router.get("/{system_id}/{model_kind}")
async def get_single_model_card(
    system_id: str, model_kind: str, state: AppState = Depends(get_app_state)
) -> dict[str, Any]:
    """Retrieve specific model card for a given system and model kind."""
    if system_id not in state.plants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "PLANT_NOT_FOUND",
                "message": f"Plant '{system_id}' not found.",
            },
        )

    if model_kind in ("arx", "prony", "classical"):
        card = state.classical_cards.get(system_id)
        if card:
            if model_kind in ("arx", "prony"):
                sub_card = dict(card)
                if model_kind in card:
                    sub_card["model"] = card[model_kind]
                return sub_card
            return card
    elif model_kind == "neural_ode":
        card = state.neural_ode_cards.get(system_id)
        if card:
            return card

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "MODEL_NOT_FOUND",
            "message": f"Model '{model_kind}' not found for plant '{system_id}'.",
        },
    )
