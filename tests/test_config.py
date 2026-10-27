"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from sysid.config import load_experiment, load_systems

# Use the real config files from the repository
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYSTEMS_PATH = _REPO_ROOT / "configs" / "systems.yaml"
_EXPERIMENT_PATH = _REPO_ROOT / "configs" / "experiment.yaml"


class TestLoadSystems:
    """Tests for load_systems."""

    def test_loads_all_three_plants(self) -> None:
        systems = load_systems(_SYSTEMS_PATH)
        assert set(systems.keys()) == {"rc", "motor", "msd"}

    def test_rc_plant_fields(self) -> None:
        systems = load_systems(_SYSTEMS_PATH)
        rc = systems["rc"]
        assert rc.name == "Series RC low-pass"
        assert rc.order == 1
        assert rc.u_max == 5.0
        assert rc.input.unit == "V"
        assert rc.output.unit == "V"
        assert rc.dominant_time_scale_s == 0.2
        assert rc.sequence_duration_s == 6.0
        assert rc.impulse_horizon_s == 1.5

    def test_motor_derived_values(self) -> None:
        systems = load_systems(_SYSTEMS_PATH)
        motor = systems["motor"]
        assert motor.derived["K_rad_s_per_V"] == 5.0
        assert motor.derived["tau_s"] == 0.25

    def test_msd_identifiable_params(self) -> None:
        systems = load_systems(_SYSTEMS_PATH)
        msd = systems["msd"]
        assert set(msd.identifiable) == {"m_kg", "c_Ns_per_m", "k_N_per_m"}
        assert msd.order == 2

    def test_reject_missing_key(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad_systems.yaml"
        # Missing 'u_max' key for the plant
        content = dedent("""\
            plants:
              broken:
                name: "Test"
                order: 1
                input: {name: "v", unit: "V"}
                output: {name: "v", unit: "V"}
                parameters: {R: 1.0}
                derived: {tau: 0.1}
                identifiable: [tau]
                dominant_time_scale_s: 0.1
                sequence_duration_s: 1.0
                impulse_horizon_s: 0.5
        """)
        bad_yaml.write_text(content)
        with pytest.raises(ValueError, match="missing keys.*u_max"):
            load_systems(bad_yaml)

    def test_reject_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_systems(tmp_path / "nonexistent.yaml")

    def test_reject_no_plants_key(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "no_plants.yaml"
        bad_yaml.write_text("something_else: 42\n")
        with pytest.raises(ValueError, match="plants"):
            load_systems(bad_yaml)


class TestLoadExperiment:
    """Tests for load_experiment."""

    def test_loads_sampling(self) -> None:
        config = load_experiment(_EXPERIMENT_PATH)
        assert config.sampling.Ts == 0.02

    def test_loads_all_splits(self) -> None:
        config = load_experiment(_EXPERIMENT_PATH)
        assert set(config.data.splits.keys()) == {"train", "val", "test_iid"}
        assert config.data.splits["train"].random_steps == 10

    def test_loads_noise_levels(self) -> None:
        config = load_experiment(_EXPERIMENT_PATH)
        # First level is None (infinity), rest are numeric
        assert config.noise.snr_levels_db[0] is None
        assert config.noise.baseline_snr_db == 40.0

    def test_loads_neural_ode_seeds(self) -> None:
        config = load_experiment(_EXPERIMENT_PATH)
        assert config.neural_ode.seeds == [0, 1, 2]
        assert config.neural_ode.n_x == 4

    def test_reject_negative_ts(self, tmp_path: Path) -> None:
        """A negative sampling period must be caught."""
        config_path = _EXPERIMENT_PATH
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        raw["sampling"]["Ts"] = -1.0
        bad_yaml = tmp_path / "bad_experiment.yaml"
        with open(bad_yaml, "w") as f:
            yaml.dump(raw, f)
        with pytest.raises(ValueError, match="Ts must be positive"):
            load_experiment(bad_yaml)

    def test_reject_empty_seeds(self, tmp_path: Path) -> None:
        """An empty seed list must be caught."""
        with open(_EXPERIMENT_PATH) as f:
            raw = yaml.safe_load(f)
        raw["neural_ode"]["seeds"] = []
        bad_yaml = tmp_path / "empty_seeds.yaml"
        with open(bad_yaml, "w") as f:
            yaml.dump(raw, f)
        with pytest.raises(ValueError, match="seeds must be a non-empty"):
            load_experiment(bad_yaml)

    def test_reject_baseline_not_in_levels(self, tmp_path: Path) -> None:
        """baseline_snr_db must appear in snr_levels_db."""
        with open(_EXPERIMENT_PATH) as f:
            raw = yaml.safe_load(f)
        raw["noise"]["baseline_snr_db"] = 99.0
        bad_yaml = tmp_path / "bad_baseline.yaml"
        with open(bad_yaml, "w") as f:
            yaml.dump(raw, f)
        with pytest.raises(ValueError, match="must appear in"):
            load_experiment(bad_yaml)
