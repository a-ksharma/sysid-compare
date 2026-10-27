"""Train all models (Neural ODE) for all plants and export artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sysid.config import (
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_SYSTEMS_PATH,
    load_experiment,
    load_systems,
)
from sysid.neural.train import train_neural_ode
from sysid.systems import get_all_plants

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI to train baseline Neural ODE models across plants and seeds."""
    parser = argparse.ArgumentParser(description="Train Neural ODE models.")
    parser.add_argument(
        "--plants",
        nargs="+",
        default=["rc", "motor", "msd"],
        help="Plant IDs to train (default: rc motor msd)",
    )
    parser.add_argument(
        "--snr-tag",
        type=str,
        default="40",
        help="Training dataset SNR tag (default: '40')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Maximum training epochs (default: 500)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds to train (default: [0])",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root artifacts directory (default: artifacts/)",
    )
    args = parser.parse_args()

    systems_cfg = load_systems(DEFAULT_SYSTEMS_PATH)
    exp_cfg = load_experiment(DEFAULT_EXPERIMENT_PATH)
    plants = get_all_plants(systems_cfg)

    data_root = Path("data")

    for plant_id in args.plants:
        if plant_id not in plants:
            logger.warning("Skipping unknown plant: %s", plant_id)
            continue

        train_path = data_root / plant_id / f"train_snr_{args.snr_tag}"
        val_path = data_root / plant_id / f"val_snr_{args.snr_tag}"

        if not train_path.with_suffix(".npz").exists():
            logger.error("Dataset not found: %s", train_path)
            continue

        seeds_to_run = args.seeds
        # Decision K10: primary plant (msd) uses 3 seeds
        if plant_id == "msd" and len(args.seeds) == 1 and args.seeds[0] == 0:
            seeds_to_run = exp_cfg.neural_ode.seeds

        for seed in seeds_to_run:
            logger.info(
                "Training Neural ODE for '%s' (seed=%d, snr=%s)...",
                plant_id,
                seed,
                args.snr_tag,
            )
            out_dir = args.artifacts_dir / "models" / plant_id
            if len(seeds_to_run) > 1:
                out_dir = out_dir / f"seed_{seed}"

            _, meta = train_neural_ode(
                plant_id=plant_id,
                train_data_path=train_path,
                val_data_path=val_path,
                seed=seed,
                n_x=exp_cfg.neural_ode.n_x,
                hidden_dim=exp_cfg.neural_ode.hidden_units,
                n_hidden=exp_cfg.neural_ode.hidden_layers,
                lr_max=exp_cfg.neural_ode.lr_max,
                lr_min=exp_cfg.neural_ode.lr_min,
                weight_decay=exp_cfg.neural_ode.weight_decay,
                max_epochs=args.epochs,
                batch_size=exp_cfg.neural_ode.batch_size,
                patience=exp_cfg.neural_ode.patience,
                output_dir=out_dir,
            )
            logger.info(
                "Completed '%s' seed=%d: best val loss=%.6e, fit=%.2f%% (epoch %d)",
                plant_id,
                seed,
                meta["best_val_loss"],
                meta["validation_fit_percent"],
                meta["best_epoch"],
            )


if __name__ == "__main__":
    main()
