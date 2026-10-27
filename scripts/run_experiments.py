"""Run experiments E1-E6 and write evaluation.json files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sysid.analysis.experiments import (
    run_e1_baseline,
    run_e2_generalization,
    run_e3_noise_sweep,
    run_e4_param_variation,
    run_e5_fidelity_and_cross_check,
    run_e6_cost,
)
from sysid.config import (
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_SYSTEMS_PATH,
    load_experiment,
    load_systems,
)
from sysid.systems import get_all_plants

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute evaluation protocol across plants and export evaluation.json."""
    parser = argparse.ArgumentParser(description="Run experiments E1 through E6.")
    parser.add_argument(
        "--plants",
        nargs="+",
        default=["rc", "motor", "msd"],
        help="Plant IDs to evaluate (default: rc motor msd)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset root directory (default: data/)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Models root directory (default: artifacts/models/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/results"),
        help="Results output directory (default: artifacts/results/)",
    )
    args = parser.parse_args()

    systems_cfg = load_systems(DEFAULT_SYSTEMS_PATH)
    exp_cfg = load_experiment(DEFAULT_EXPERIMENT_PATH)
    plants = get_all_plants(systems_cfg)

    for plant_id in args.plants:
        if plant_id not in plants:
            logger.warning("Skipping unknown plant: %s", plant_id)
            continue

        plant = plants[plant_id]
        logger.info("==========================================")
        logger.info("Running evaluation experiments for '%s'...", plant.name)
        logger.info("==========================================")

        # E1: Baseline
        logger.info("[%s] E1: Baseline test_iid evaluation...", plant_id)
        iid_data, curves_data = run_e1_baseline(
            plant, args.data_dir, args.models_dir, exp_cfg
        )

        # E2: Generalization (OOD)
        logger.info("[%s] E2: OOD generalization...", plant_id)
        ood_data = run_e2_generalization(plant, args.data_dir, args.models_dir, exp_cfg)

        # E3: Noise robustness sweep
        logger.info("[%s] E3: Noise sweep across SNR levels...", plant_id)
        noise_data = run_e3_noise_sweep(plant, args.data_dir, args.models_dir, exp_cfg)

        # E4: Parameter variation (MSD only per D2 & D9)
        if plant_id == "msd":
            logger.info("[%s] E4: Parameter variation study (10 plants)...", plant_id)
            param_data = run_e4_param_variation(plant, exp_cfg, n_plants=10, seed=99)
        else:
            param_data = None

        # E5: Frequency and impulse fidelity plus cross-check
        logger.info("[%s] E5: Frequency and impulse cross-check...", plant_id)
        cross_check_data = run_e5_fidelity_and_cross_check(
            plant, args.models_dir, exp_cfg
        )

        # E6: Computational cost
        logger.info("[%s] E6: Computational cost evaluation...", plant_id)
        cost_data = run_e6_cost(plant, args.models_dir, exp_cfg)

        # Assemble evaluation.json
        freq_resp_dict = {
            **curves_data["frequency_response"],
            "cross_check": cross_check_data,
        }

        eval_result = {
            "iid": iid_data,
            "ood": ood_data,
            "noise_sweep": noise_data,
            "param_variation": param_data,
            "frequency_response": freq_resp_dict,
            "impulse_response": curves_data["impulse_response"],
            "cost": cost_data,
        }

        plant_results_dir = args.results_dir / plant_id
        plant_results_dir.mkdir(parents=True, exist_ok=True)
        eval_path = plant_results_dir / "evaluation.json"

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)

        logger.info("[%s] Saved evaluation results to %s", plant_id, eval_path)


if __name__ == "__main__":
    main()
