"""Dataset generation script for classical and learned system identification.

Generates all required splits for rc, motor, and msd plants across all
specified SNR levels, with reproducible pseudorandom seeds, exact ZOH
simulation, and JSON sidecars.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from sysid.config import (
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_SYSTEMS_PATH,
    ExperimentConfig,
    load_experiment,
    load_systems,
)
from sysid.data_io import save_dataset
from sysid.signals import (
    generate_chirp,
    generate_impulse_pulse,
    generate_multisine,
    generate_random_steps,
    generate_sawtooth,
    generate_square,
    generate_time_vector,
)
from sysid.simulate import add_output_noise, simulate_plant
from sysid.systems import PlantModel, get_all_plants

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_sequence_split(
    plant: PlantModel,
    n_random_steps: int,
    n_multisine: int,
    n_chirp: int,
    Ts: float,
    rng: np.random.Generator,
    quiet_lead_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a batch of input-output sequences for a given split configuration.

    Parameters
    ----------
    plant : PlantModel
        Target plant model.
    n_random_steps : int
        Number of random steps sequences to generate.
    n_multisine : int
        Number of multisine sequences to generate.
    n_chirp : int
        Number of chirp sequences to generate.
    Ts : float
        Sampling interval in seconds.
    rng : np.random.Generator
        Split-specific pseudorandom generator.
    quiet_lead_s : float, optional
        Initial quiet duration in seconds, by default 0.5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        (u_batch, y_clean_batch, signal_type_list)
        - u_batch of shape (n_sequences, n_samples)
        - y_clean_batch of shape (n_sequences, n_samples)
        - signal_type_list of length n_sequences
    """
    duration_s = plant.sequence_duration_s
    n_total = n_random_steps + n_multisine + n_chirp
    n_samples = int(round(duration_s / Ts))

    u_batch = np.zeros((n_total, n_samples), dtype=np.float64)
    y_clean_batch = np.zeros((n_total, n_samples), dtype=np.float64)
    signal_types: list[str] = []

    idx = 0
    for _ in range(n_random_steps):
        u_seq = generate_random_steps(
            duration_s, Ts, plant.u_max, rng, quiet_lead_s=quiet_lead_s
        )
        y_seq, _ = simulate_plant(plant, u_seq, Ts)
        u_batch[idx] = u_seq
        y_clean_batch[idx] = y_seq
        signal_types.append("random_steps")
        idx += 1

    for _ in range(n_multisine):
        u_seq = generate_multisine(
            duration_s, Ts, plant.u_max, rng, quiet_lead_s=quiet_lead_s
        )
        y_seq, _ = simulate_plant(plant, u_seq, Ts)
        u_batch[idx] = u_seq
        y_clean_batch[idx] = y_seq
        signal_types.append("multisine")
        idx += 1

    for _ in range(n_chirp):
        u_seq = generate_chirp(
            duration_s, Ts, plant.u_max, rng, quiet_lead_s=quiet_lead_s
        )
        y_seq, _ = simulate_plant(plant, u_seq, Ts)
        u_batch[idx] = u_seq
        y_clean_batch[idx] = y_seq
        signal_types.append("chirp")
        idx += 1

    return u_batch, y_clean_batch, signal_types


def generate_plant_datasets(
    plant: PlantModel,
    exp_config: ExperimentConfig,
    output_dir: Path,
    base_seed: int,
) -> dict[str, str]:
    """Generate all dataset splits and noise variants for a single plant.

    Parameters
    ----------
    plant : PlantModel
        Target plant model.
    exp_config : ExperimentConfig
        Experiment configuration settings.
    output_dir : Path
        Directory where plant datasets are stored (e.g. data/msd).
    base_seed : int
        Deterministic seed for this plant.

    Returns
    -------
    dict[str, str]
        Mapping from filename to SHA256 content checksum.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    Ts = exp_config.sampling.Ts
    file_hashes: dict[str, str] = {}

    # Distinct seeds per split to guarantee statistical independence
    train_rng = np.random.default_rng(base_seed + 10)
    val_rng = np.random.default_rng(base_seed + 20)
    test_rng = np.random.default_rng(base_seed + 30)
    ood_rng = np.random.default_rng(base_seed + 40)
    noise_rng = np.random.default_rng(base_seed + 50)

    # 1. Generate Clean Base Sequences
    train_cfg = exp_config.data.splits["train"]
    u_train, y_train_clean, types_train = generate_sequence_split(
        plant,
        train_cfg.random_steps,
        train_cfg.multisine,
        train_cfg.chirp,
        Ts,
        train_rng,
    )

    val_cfg = exp_config.data.splits["val"]
    u_val, y_val_clean, types_val = generate_sequence_split(
        plant, val_cfg.random_steps, val_cfg.multisine, val_cfg.chirp, Ts, val_rng
    )

    test_cfg = exp_config.data.splits["test_iid"]
    u_test, y_test_clean, types_test = generate_sequence_split(
        plant,
        test_cfg.random_steps,
        test_cfg.multisine,
        test_cfg.chirp,
        Ts,
        test_rng,
    )

    t_grid = generate_time_vector(plant.sequence_duration_s, Ts)

    # 2. Compute Noise Sigma from the Entire Clean Training Split
    # Spec 4.4: sigma = std(y_clean over the train split) * 10^(-SNR_dB / 20)
    train_clean_std = float(np.std(y_train_clean))

    # 3. Create datasets for all requested SNR levels
    snr_levels: list[float | None] = [None, 40.0, 30.0, 20.0, 10.0]

    for snr_db in snr_levels:
        if snr_db is None:
            tag = "inf"
            sigma = 0.0
        else:
            tag = f"{int(snr_db)}"
            sigma = float(train_clean_std * (10.0 ** (-snr_db / 20.0)))

        # Train split
        y_train_noisy = np.zeros_like(y_train_clean)
        for i in range(len(y_train_clean)):
            y_train_noisy[i], _ = add_output_noise(
                y_train_clean[i], snr_db, noise_rng, sigma=sigma
            )

        npz_p, json_p = save_dataset(
            output_dir / f"train_snr_{tag}",
            arrays={
                "t": t_grid,
                "u": u_train,
                "y_clean": y_train_clean,
                "y_noisy": y_train_noisy,
                "signal_types": np.array(types_train),
            },
            metadata={
                "plant_id": plant.id,
                "split": "train",
                "snr_db": snr_db,
                "sigma": sigma,
                "n_sequences": len(u_train),
                "Ts": Ts,
            },
        )
        file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

        # Val split
        y_val_noisy = np.zeros_like(y_val_clean)
        for i in range(len(y_val_clean)):
            y_val_noisy[i], _ = add_output_noise(
                y_val_clean[i], snr_db, noise_rng, sigma=sigma
            )

        npz_p, json_p = save_dataset(
            output_dir / f"val_snr_{tag}",
            arrays={
                "t": t_grid,
                "u": u_val,
                "y_clean": y_val_clean,
                "y_noisy": y_val_noisy,
                "signal_types": np.array(types_val),
            },
            metadata={
                "plant_id": plant.id,
                "split": "val",
                "snr_db": snr_db,
                "sigma": sigma,
                "n_sequences": len(u_val),
                "Ts": Ts,
            },
        )
        file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

        # Test IID split (scored against clean output per spec 4.4)
        y_test_noisy = np.zeros_like(y_test_clean)
        for i in range(len(y_test_clean)):
            y_test_noisy[i], _ = add_output_noise(
                y_test_clean[i], snr_db, noise_rng, sigma=sigma
            )

        npz_p, json_p = save_dataset(
            output_dir / f"test_iid_snr_{tag}",
            arrays={
                "t": t_grid,
                "u": u_test,
                "y_clean": y_test_clean,
                "y_noisy": y_test_noisy,
                "signal_types": np.array(types_test),
            },
            metadata={
                "plant_id": plant.id,
                "split": "test_iid",
                "snr_db": snr_db,
                "sigma": sigma,
                "n_sequences": len(u_test),
                "Ts": Ts,
            },
        )
        file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

    # 4. Out-of-Distribution (OOD) Test Sets (Spec 4.3)
    n_ood_seqs = 6
    duration_s = plant.sequence_duration_s
    n_samples = int(round(duration_s / Ts))

    # OOD 1: Amplitude (3 x u_max)
    u_ood_amp = np.zeros((n_ood_seqs, n_samples), dtype=np.float64)
    y_ood_amp = np.zeros((n_ood_seqs, n_samples), dtype=np.float64)
    for i in range(n_ood_seqs):
        u_seq = generate_random_steps(duration_s, Ts, 3.0 * plant.u_max, ood_rng)
        y_seq, _ = simulate_plant(plant, u_seq, Ts)
        u_ood_amp[i] = u_seq
        y_ood_amp[i] = y_seq

    npz_p, json_p = save_dataset(
        output_dir / "ood_amplitude",
        arrays={"t": t_grid, "u": u_ood_amp, "y_clean": y_ood_amp},
        metadata={
            "plant_id": plant.id,
            "split": "ood_amplitude",
            "scale_factor": 3.0,
            "n_sequences": n_ood_seqs,
            "Ts": Ts,
        },
    )
    file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

    # OOD 2: Frequency (chirps sweeping 15 to 40 rad/s)
    u_ood_freq = np.zeros((n_ood_seqs, n_samples), dtype=np.float64)
    y_ood_freq = np.zeros((n_ood_seqs, n_samples), dtype=np.float64)
    for i in range(n_ood_seqs):
        u_seq = generate_chirp(
            duration_s,
            Ts,
            plant.u_max,
            ood_rng,
            omega_range_rad_s=(15.0, 40.0),
        )
        y_seq, _ = simulate_plant(plant, u_seq, Ts)
        u_ood_freq[i] = u_seq
        y_ood_freq[i] = y_seq

    npz_p, json_p = save_dataset(
        output_dir / "ood_frequency",
        arrays={"t": t_grid, "u": u_ood_freq, "y_clean": y_ood_freq},
        metadata={
            "plant_id": plant.id,
            "split": "ood_frequency",
            "frequency_range_rad_s": [15.0, 40.0],
            "n_sequences": n_ood_seqs,
            "Ts": Ts,
        },
    )
    file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

    # OOD 3: Waveform (square and sawtooth waves, periods 1 s to 8 s)
    u_ood_wave = np.zeros((n_ood_seqs, n_samples), dtype=np.float64)
    y_ood_wave = np.zeros((n_ood_seqs, n_samples), dtype=np.float64)
    for i in range(n_ood_seqs):
        period = float(ood_rng.uniform(1.0, 8.0))
        if i % 2 == 0:
            u_seq = generate_square(duration_s, Ts, plant.u_max, period_s=period)
        else:
            u_seq = generate_sawtooth(duration_s, Ts, plant.u_max, period_s=period)
        y_seq, _ = simulate_plant(plant, u_seq, Ts)
        u_ood_wave[i] = u_seq
        y_ood_wave[i] = y_seq

    npz_p, json_p = save_dataset(
        output_dir / "ood_waveform",
        arrays={"t": t_grid, "u": u_ood_wave, "y_clean": y_ood_wave},
        metadata={
            "plant_id": plant.id,
            "split": "ood_waveform",
            "waveforms": ["square", "sawtooth"],
            "period_range_s": [1.0, 8.0],
            "n_sequences": n_ood_seqs,
            "Ts": Ts,
        },
    )
    file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

    # 5. Impulse Experiment for Prony's Method (Spec 4.3 & 5.2, Kickoff K5)
    t_impulse = generate_time_vector(plant.impulse_horizon_s, Ts)
    pulse_fraction = 0.1
    u_impulse = generate_impulse_pulse(
        plant.impulse_horizon_s,
        Ts,
        plant.u_max,
        amplitude_fraction=pulse_fraction,
    )
    y_impulse_clean, _ = simulate_plant(plant, u_impulse, Ts)

    # Scaled response dividing by the pulse area (0.1 * u_max * Ts)
    pulse_area = pulse_fraction * plant.u_max * Ts
    y_impulse_scaled = y_impulse_clean / pulse_area

    npz_p, json_p = save_dataset(
        output_dir / "impulse_experiment",
        arrays={
            "t": t_impulse,
            "u": u_impulse,
            "y_clean": y_impulse_clean,
            "y_scaled": y_impulse_scaled,
        },
        metadata={
            "plant_id": plant.id,
            "split": "impulse_experiment",
            "pulse_amplitude": pulse_fraction * plant.u_max,
            "pulse_area": pulse_area,
            "Ts": Ts,
        },
    )
    file_hashes[npz_p.name] = json.loads(json_p.read_text())["sha256"]

    # Write Manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(file_hashes, f, indent=2, sort_keys=True)

    return file_hashes


def main() -> None:
    """CLI entrypoint to generate all dataset files."""
    parser = argparse.ArgumentParser(description="Generate datasets for all plants.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root output directory (default: data/)",
    )
    args = parser.parse_args()

    systems_cfg = load_systems(DEFAULT_SYSTEMS_PATH)
    exp_cfg = load_experiment(DEFAULT_EXPERIMENT_PATH)
    plants = get_all_plants(systems_cfg)

    base_master_seed = exp_cfg.data.seed

    logger.info("Generating datasets for plants: %s", list(plants.keys()))
    for idx, (plant_id, plant) in enumerate(plants.items()):
        plant_dir = args.output_dir / plant_id
        plant_seed = base_master_seed + idx * 1000
        logger.info(
            "Generating datasets for '%s' in %s (seed=%d)...",
            plant_id,
            plant_dir,
            plant_seed,
        )
        hashes = generate_plant_datasets(plant, exp_cfg, plant_dir, plant_seed)
        logger.info(
            "Completed '%s': %d dataset files generated.",
            plant_id,
            len(hashes),
        )

    logger.info("All datasets successfully generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
