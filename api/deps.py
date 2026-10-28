"""Application dependencies and artifact state management for FastAPI service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sysid.config import (
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_SYSTEMS_PATH,
    ExperimentConfig,
    SystemConfig,
    load_experiment,
    load_systems,
)
from sysid.neural.numpy_infer import NumpyNeuralODE, load_numpy_neural_ode
from sysid.systems import PlantModel, get_all_plants

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """In-memory cache of loaded models, systems, and artifacts."""

    systems_cfg: dict[str, SystemConfig] | None = None
    exp_cfg: ExperimentConfig | None = None
    plants: dict[str, PlantModel] = field(default_factory=dict)
    classical_cards: dict[str, dict[str, Any]] = field(default_factory=dict)
    neural_ode_cards: dict[str, dict[str, Any]] = field(default_factory=dict)
    neural_ode_models: dict[str, NumpyNeuralODE] = field(default_factory=dict)
    evaluation_results: dict[str, dict[str, Any]] = field(default_factory=dict)


state = AppState()


def load_application_state(
    artifacts_dir: Path = Path("artifacts"),
    systems_path: Path = DEFAULT_SYSTEMS_PATH,
    experiment_path: Path = DEFAULT_EXPERIMENT_PATH,
) -> None:
    """Load all physical systems, model cards, numpy neural ODEs, and evaluations."""
    logger.info("Initializing API application state...")
    state.systems_cfg = load_systems(systems_path)
    state.exp_cfg = load_experiment(experiment_path)
    state.plants = get_all_plants(state.systems_cfg)

    models_dir = artifacts_dir / "models"
    results_dir = artifacts_dir / "results"

    for plant_id in state.plants:
        p_models = models_dir / plant_id

        # 1. Classical card
        c_path = p_models / "classical.json"
        if c_path.exists():
            with open(c_path, encoding="utf-8") as f:
                state.classical_cards[plant_id] = json.load(f)

        # 2. Neural ODE card and numpy weights
        node_card_path = p_models / "neural_ode.json"
        if node_card_path.exists():
            with open(node_card_path, encoding="utf-8") as f:
                state.neural_ode_cards[plant_id] = json.load(f)

        node_npz_path = p_models / "neural_ode.npz"
        if node_npz_path.exists():
            state.neural_ode_models[plant_id] = load_numpy_neural_ode(node_npz_path)

        # 3. Evaluation result
        eval_path = results_dir / plant_id / "evaluation.json"
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                state.evaluation_results[plant_id] = json.load(f)

    logger.info("Application state initialization complete.")


async def get_app_state() -> AppState:
    """Dependency provider returning loaded application state."""
    if not state.plants:
        load_application_state()
    return state
