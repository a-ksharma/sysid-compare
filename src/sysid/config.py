"""Configuration loading and validation for systems.yaml and experiment.yaml.

Reads YAML files into typed dataclasses and validates that all required fields
are present and within acceptable ranges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths relative to the repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEMS_PATH = _REPO_ROOT / "configs" / "systems.yaml"
DEFAULT_EXPERIMENT_PATH = _REPO_ROOT / "configs" / "experiment.yaml"


# ---------------------------------------------------------------------------
# System (plant) configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IOConfig:
    """Input or output signal description."""

    name: str
    unit: str


@dataclass(frozen=True)
class SystemConfig:
    """Single plant configuration loaded from systems.yaml."""

    plant_id: str
    name: str
    order: int
    input: IOConfig
    output: IOConfig
    parameters: dict[str, float]
    derived: dict[str, float]
    u_max: float
    identifiable: list[str]
    dominant_time_scale_s: float
    sequence_duration_s: float
    impulse_horizon_s: float


def _parse_io(raw: dict[str, Any], field_name: str, plant_id: str) -> IOConfig:
    """Parse an input or output block from the YAML."""
    if not isinstance(raw, dict):
        msg = f"Plant '{plant_id}': '{field_name}' must be a mapping."
        raise ValueError(msg)
    for key in ("name", "unit"):
        if key not in raw:
            msg = f"Plant '{plant_id}': '{field_name}' is missing '{key}'."
            raise ValueError(msg)
    return IOConfig(name=raw["name"], unit=raw["unit"])


_SYSTEM_REQUIRED_KEYS = {
    "name",
    "order",
    "input",
    "output",
    "parameters",
    "derived",
    "u_max",
    "identifiable",
    "dominant_time_scale_s",
    "sequence_duration_s",
    "impulse_horizon_s",
}


def _parse_system(plant_id: str, raw: dict[str, Any]) -> SystemConfig:
    """Parse and validate a single plant entry."""
    missing = _SYSTEM_REQUIRED_KEYS - set(raw.keys())
    if missing:
        msg = f"Plant '{plant_id}': missing keys {sorted(missing)}."
        raise ValueError(msg)

    order = raw["order"]
    if not isinstance(order, int) or order < 1:
        msg = f"Plant '{plant_id}': 'order' must be a positive integer, got {order!r}."
        raise ValueError(msg)

    u_max = float(raw["u_max"])
    if u_max <= 0:
        msg = f"Plant '{plant_id}': 'u_max' must be positive, got {u_max}."
        raise ValueError(msg)

    return SystemConfig(
        plant_id=plant_id,
        name=raw["name"],
        order=order,
        input=_parse_io(raw["input"], "input", plant_id),
        output=_parse_io(raw["output"], "output", plant_id),
        parameters={k: float(v) for k, v in raw["parameters"].items()},
        derived={k: float(v) for k, v in raw["derived"].items()},
        u_max=u_max,
        identifiable=list(raw["identifiable"]),
        dominant_time_scale_s=float(raw["dominant_time_scale_s"]),
        sequence_duration_s=float(raw["sequence_duration_s"]),
        impulse_horizon_s=float(raw["impulse_horizon_s"]),
    )


def load_systems(path: Path | None = None) -> dict[str, SystemConfig]:
    """Load and validate all plant definitions from systems.yaml.

    Parameters
    ----------
    path : Path or None
        Path to the YAML file. Defaults to ``configs/systems.yaml`` relative
        to the repository root.

    Returns
    -------
    dict[str, SystemConfig]
        Mapping from plant id to its validated configuration.

    Raises
    ------
    ValueError
        If any required key is missing or a value is out of range.
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = path or DEFAULT_SYSTEMS_PATH
    if not path.exists():
        msg = f"Systems config not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "plants" not in raw:
        msg = f"Systems config must have a top-level 'plants' key: {path}"
        raise ValueError(msg)

    plants_raw = raw["plants"]
    if not isinstance(plants_raw, dict) or len(plants_raw) == 0:
        msg = f"Systems config 'plants' must be a non-empty mapping: {path}"
        raise ValueError(msg)

    systems: dict[str, SystemConfig] = {}
    for plant_id, plant_raw in plants_raw.items():
        systems[plant_id] = _parse_system(plant_id, plant_raw)
        logger.debug("Loaded plant '%s': %s", plant_id, systems[plant_id].name)

    return systems


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling configuration."""

    Ts: float  # noqa: N815 -- matches spec notation


@dataclass(frozen=True)
class ExcitationConfig:
    """Excitation signal band and plotting range."""

    omega_min_rad_s: float
    omega_max_rad_s: float
    plot_omega_max_rad_s: float
    bode_grid_points: int


@dataclass(frozen=True)
class SplitConfig:
    """Number of sequences per signal type in a data split."""

    random_steps: int
    multisine: int
    chirp: int


@dataclass(frozen=True)
class DataConfig:
    """Dataset generation configuration."""

    seed: int
    splits: dict[str, SplitConfig]
    quiet_lead_s: float
    hold_time_range_s: tuple[float, float]
    multisine_components: int
    impulse_amplitude_fraction: float


@dataclass(frozen=True)
class NoiseConfig:
    """Noise configuration."""

    snr_levels_db: list[float | None]  # None means infinity (no noise)
    baseline_snr_db: float


@dataclass(frozen=True)
class NeuralODEConfig:
    """Neural ODE training hyperparameters."""

    n_x: int
    hidden_layers: int
    hidden_units: int
    activation: str
    lr_max: float
    lr_min: float
    weight_decay: float
    max_epochs: int
    batch_size: int
    patience: int
    curriculum_fractions: list[float]
    curriculum_epoch_fractions: list[float]
    seeds: list[int]
    sanity_fit_threshold: float


@dataclass(frozen=True)
class OODConfig:
    """Out-of-distribution test configuration."""

    amplitude_scale: float
    frequency_range_rad_s: tuple[float, float]
    waveform_period_range_s: tuple[float, float]


@dataclass(frozen=True)
class ParamVariationConfig:
    """Parameter variation study configuration."""

    n_plants: int
    scale_range: tuple[float, float]
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration loaded from experiment.yaml."""

    sampling: SamplingConfig
    excitation: ExcitationConfig
    data: DataConfig
    noise: NoiseConfig
    neural_ode: NeuralODEConfig
    ood: OODConfig
    param_variation: ParamVariationConfig


def _require_keys(raw: dict[str, Any], keys: set[str], section: str) -> None:
    """Raise ValueError if any key is missing from raw."""
    missing = keys - set(raw.keys())
    if missing:
        msg = f"Experiment config '{section}': missing keys {sorted(missing)}."
        raise ValueError(msg)


def _parse_split(raw: dict[str, Any], split_name: str) -> SplitConfig:
    """Parse a single data split."""
    for key in ("random_steps", "multisine", "chirp"):
        if key not in raw:
            msg = f"Split '{split_name}': missing '{key}'."
            raise ValueError(msg)
    return SplitConfig(
        random_steps=int(raw["random_steps"]),
        multisine=int(raw["multisine"]),
        chirp=int(raw["chirp"]),
    )


def load_experiment(path: Path | None = None) -> ExperimentConfig:
    """Load and validate experiment configuration from experiment.yaml.

    Parameters
    ----------
    path : Path or None
        Path to the YAML file. Defaults to ``configs/experiment.yaml``
        relative to the repository root.

    Returns
    -------
    ExperimentConfig
        Validated experiment configuration.

    Raises
    ------
    ValueError
        If any required key is missing or a value is out of range.
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = path or DEFAULT_EXPERIMENT_PATH
    if not path.exists():
        msg = f"Experiment config not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Experiment config must be a mapping: {path}"
        raise ValueError(msg)

    # -- sampling --
    _require_keys(raw, {"sampling"}, "top-level")
    sampling_raw = raw["sampling"]
    _require_keys(sampling_raw, {"Ts"}, "sampling")
    ts = float(sampling_raw["Ts"])
    if ts <= 0:
        msg = f"sampling.Ts must be positive, got {ts}."
        raise ValueError(msg)
    sampling = SamplingConfig(Ts=ts)

    # -- excitation --
    _require_keys(raw, {"excitation"}, "top-level")
    exc_raw = raw["excitation"]
    _require_keys(
        exc_raw,
        {
            "omega_min_rad_s",
            "omega_max_rad_s",
            "plot_omega_max_rad_s",
            "bode_grid_points",
        },
        "excitation",
    )
    excitation = ExcitationConfig(
        omega_min_rad_s=float(exc_raw["omega_min_rad_s"]),
        omega_max_rad_s=float(exc_raw["omega_max_rad_s"]),
        plot_omega_max_rad_s=float(exc_raw["plot_omega_max_rad_s"]),
        bode_grid_points=int(exc_raw["bode_grid_points"]),
    )

    # -- data --
    _require_keys(raw, {"data"}, "top-level")
    data_raw = raw["data"]
    _require_keys(
        data_raw,
        {
            "seed",
            "splits",
            "quiet_lead_s",
            "hold_time_range_s",
            "multisine_components",
            "impulse_amplitude_fraction",
        },
        "data",
    )
    splits_raw = data_raw["splits"]
    splits: dict[str, SplitConfig] = {}
    for split_name in ("train", "val", "test_iid"):
        if split_name not in splits_raw:
            msg = f"data.splits: missing split '{split_name}'."
            raise ValueError(msg)
        splits[split_name] = _parse_split(splits_raw[split_name], split_name)

    hold_range = data_raw["hold_time_range_s"]
    if not isinstance(hold_range, list) or len(hold_range) != 2:
        msg = "data.hold_time_range_s must be a list of two floats."
        raise ValueError(msg)

    data = DataConfig(
        seed=int(data_raw["seed"]),
        splits=splits,
        quiet_lead_s=float(data_raw["quiet_lead_s"]),
        hold_time_range_s=(float(hold_range[0]), float(hold_range[1])),
        multisine_components=int(data_raw["multisine_components"]),
        impulse_amplitude_fraction=float(data_raw["impulse_amplitude_fraction"]),
    )

    # -- noise --
    _require_keys(raw, {"noise"}, "top-level")
    noise_raw = raw["noise"]
    _require_keys(noise_raw, {"snr_levels_db", "baseline_snr_db"}, "noise")
    snr_levels: list[float | None] = []
    for v in noise_raw["snr_levels_db"]:
        snr_levels.append(None if v is None else float(v))
    baseline_snr = float(noise_raw["baseline_snr_db"])
    if baseline_snr not in [s for s in snr_levels if s is not None]:
        msg = (
            f"noise.baseline_snr_db ({baseline_snr}) "
            f"must appear in noise.snr_levels_db."
        )
        raise ValueError(msg)
    noise = NoiseConfig(snr_levels_db=snr_levels, baseline_snr_db=baseline_snr)

    # -- neural_ode --
    _require_keys(raw, {"neural_ode"}, "top-level")
    node_raw = raw["neural_ode"]
    _require_keys(
        node_raw,
        {
            "n_x",
            "hidden_layers",
            "hidden_units",
            "activation",
            "lr_max",
            "lr_min",
            "weight_decay",
            "max_epochs",
            "batch_size",
            "patience",
            "curriculum_fractions",
            "curriculum_epoch_fractions",
            "seeds",
            "sanity_fit_threshold",
        },
        "neural_ode",
    )
    seeds = node_raw["seeds"]
    if not isinstance(seeds, list) or len(seeds) == 0:
        msg = "neural_ode.seeds must be a non-empty list of integers."
        raise ValueError(msg)
    neural_ode = NeuralODEConfig(
        n_x=int(node_raw["n_x"]),
        hidden_layers=int(node_raw["hidden_layers"]),
        hidden_units=int(node_raw["hidden_units"]),
        activation=str(node_raw["activation"]),
        lr_max=float(node_raw["lr_max"]),
        lr_min=float(node_raw["lr_min"]),
        weight_decay=float(node_raw["weight_decay"]),
        max_epochs=int(node_raw["max_epochs"]),
        batch_size=int(node_raw["batch_size"]),
        patience=int(node_raw["patience"]),
        curriculum_fractions=list(map(float, node_raw["curriculum_fractions"])),
        curriculum_epoch_fractions=list(
            map(float, node_raw["curriculum_epoch_fractions"])
        ),
        seeds=[int(s) for s in seeds],
        sanity_fit_threshold=float(node_raw["sanity_fit_threshold"]),
    )

    # -- ood --
    _require_keys(raw, {"ood"}, "top-level")
    ood_raw = raw["ood"]
    _require_keys(
        ood_raw,
        {"amplitude_scale", "frequency_range_rad_s", "waveform_period_range_s"},
        "ood",
    )
    freq_range = ood_raw["frequency_range_rad_s"]
    period_range = ood_raw["waveform_period_range_s"]
    ood = OODConfig(
        amplitude_scale=float(ood_raw["amplitude_scale"]),
        frequency_range_rad_s=(float(freq_range[0]), float(freq_range[1])),
        waveform_period_range_s=(float(period_range[0]), float(period_range[1])),
    )

    # -- param_variation --
    _require_keys(raw, {"param_variation"}, "top-level")
    pv_raw = raw["param_variation"]
    _require_keys(pv_raw, {"n_plants", "scale_range", "seed"}, "param_variation")
    scale_range = pv_raw["scale_range"]
    param_variation = ParamVariationConfig(
        n_plants=int(pv_raw["n_plants"]),
        scale_range=(float(scale_range[0]), float(scale_range[1])),
        seed=int(pv_raw["seed"]),
    )

    config = ExperimentConfig(
        sampling=sampling,
        excitation=excitation,
        data=data,
        noise=noise,
        neural_ode=neural_ode,
        ood=ood,
        param_variation=param_variation,
    )
    logger.debug(
        "Loaded experiment config: Ts=%s s, %d SNR levels", ts, len(snr_levels)
    )
    return config
