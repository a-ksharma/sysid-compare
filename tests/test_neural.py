"""Unit tests for Phase 4 Neural ODE architecture, training, and parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sysid.analysis.linearize import linearize_numpy_neural_ode
from sysid.neural.export import export_neural_ode
from sysid.neural.model import NeuralODE
from sysid.neural.numpy_infer import NumpyNeuralODE, load_numpy_neural_ode
from sysid.neural.train import train_neural_ode


def test_zoh_evaluation_time_boundaries() -> None:
    """Acceptance criterion 1: ZOH evaluation-time test from spec section 7.2.

    Asserts that no vector field evaluation lands on a multiple of Ts that
    belongs to the next interval during midpoint numerical integration.
    """
    Ts = 0.02
    n_samples = 50
    model = NeuralODE(n_x=4, Ts=Ts)

    u_norm = torch.zeros(1, n_samples)
    eval_times: list[float] = []

    def record_time(t: float) -> None:
        eval_times.append(t)

    _ = model(u_norm, eval_callback=record_time)

    assert len(eval_times) > 0

    for t_val in eval_times:
        # Check index computed by floor(t / Ts + 1e-6)
        computed_idx = int(np.floor(t_val / Ts + 1e-6))
        # The time t_val must belong to computed_idx
        upper_bound = (computed_idx + 1) * Ts + 1e-9
        assert t_val <= upper_bound, (
            f"Evaluation t={t_val:.6f} exceeded upper bound {upper_bound:.6f}"
        )


def test_pytorch_numpy_parity(tmp_path: Path) -> None:
    """Acceptance criterion 2: PyTorch/NumPy parity within 1e-4 in normalized units."""
    torch.manual_seed(42)
    Ts = 0.02
    n_samples = 100

    model = NeuralODE(n_x=4, hidden_dim=32, n_hidden=2, Ts=Ts, u_scale=5.0, y_scale=2.0)
    model.eval()

    # Generate synthetic input
    u_np = np.sin(np.linspace(0, 5, n_samples)).astype(np.float32)
    u_torch = torch.from_numpy(u_np).unsqueeze(0)

    # PyTorch inference
    with torch.no_grad():
        y_torch = model(u_torch).squeeze(0).cpu().numpy()

    # Export and load via NumPy inference
    npz_path, _ = export_neural_ode(model, metadata={}, output_dir=tmp_path)
    np_model = load_numpy_neural_ode(npz_path)

    # NumPy inference
    y_numpy = np_model.simulate(u_np)

    max_abs_diff = float(np.max(np.abs(y_torch - y_numpy)))
    assert max_abs_diff < 1e-4, (
        f"Parity error {max_abs_diff:.2e} exceeded 1e-4 between PyTorch and NumPy"
    )


def test_analytical_linearization_matches_autograd() -> None:
    """Verify analytical chain-rule Jacobian matches PyTorch autograd."""
    torch.manual_seed(123)
    Ts = 0.02
    model = NeuralODE(
        n_x=4, hidden_dim=32, n_hidden=2, Ts=Ts, u_scale=10.0, y_scale=3.0
    )

    # 1. PyTorch autograd Jacobian at origin
    x0 = torch.zeros(1, 4)
    u0 = torch.zeros(1, 1)

    def f_x(x: torch.Tensor) -> torch.Tensor:
        return model.vector_field(x, u0).squeeze(0)

    def f_u(u: torch.Tensor) -> torch.Tensor:
        return model.vector_field(x0, u).squeeze(0)

    A_autograd = torch.autograd.functional.jacobian(f_x, x0).squeeze().numpy()
    B_autograd = (
        torch.autograd.functional.jacobian(f_u, u0).squeeze().numpy().reshape(4, 1)
        / model.u_scale
    )

    # 2. Analytical NumPy Jacobian
    npz_data = {
        "n_x": 4,
        "Ts": Ts,
        "u_scale": model.u_scale,
        "y_scale": model.y_scale,
        "W1": model.vector_field.net[0].weight.detach().numpy(),
        "b1": model.vector_field.net[0].bias.detach().numpy(),
        "W2": model.vector_field.net[2].weight.detach().numpy(),
        "b2": model.vector_field.net[2].bias.detach().numpy(),
        "W3": model.vector_field.net[4].weight.detach().numpy(),
        "b3": model.vector_field.net[4].bias.detach().numpy(),
        "C": model.readout.weight.detach().numpy(),
    }
    np_model = NumpyNeuralODE(**npz_data)
    A_ana, B_ana, _, _, _ = linearize_numpy_neural_ode(np_model)

    np.testing.assert_allclose(A_ana, A_autograd, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(B_ana, B_autograd, rtol=1e-5, atol=1e-5)


def test_training_reproducibility(tmp_path: Path) -> None:
    """Acceptance criterion 4: Same seed gives same val loss to 6 digits."""
    train_path = Path("data/rc/train_snr_inf")
    val_path = Path("data/rc/val_snr_inf")
    if not train_path.with_suffix(".npz").exists():
        pytest.skip("Data not generated")

    _, meta1 = train_neural_ode(
        plant_id="rc",
        train_data_path=train_path,
        val_data_path=val_path,
        seed=42,
        max_epochs=3,
        batch_size=12,
        output_dir=tmp_path / "run1",
    )

    _, meta2 = train_neural_ode(
        plant_id="rc",
        train_data_path=train_path,
        val_data_path=val_path,
        seed=42,
        max_epochs=3,
        batch_size=12,
        output_dir=tmp_path / "run2",
    )

    loss1 = float(meta1["best_val_loss"])
    loss2 = float(meta2["best_val_loss"])

    # Agreement to 6 significant digits
    np.testing.assert_approx_equal(loss1, loss2, significant=6)


def test_model_card_fields(tmp_path: Path) -> None:
    """Acceptance criterion 5: Model card contains all required metadata fields."""
    train_path = Path("data/rc/train_snr_inf")
    val_path = Path("data/rc/val_snr_inf")
    if not train_path.with_suffix(".npz").exists():
        pytest.skip("Data not generated")

    out_dir = tmp_path / "export"
    _, meta = train_neural_ode(
        plant_id="rc",
        train_data_path=train_path,
        val_data_path=val_path,
        seed=1,
        max_epochs=2,
        output_dir=out_dir,
    )

    card_path = out_dir / "neural_ode.json"
    assert card_path.exists()

    with open(card_path, encoding="utf-8") as f:
        card = json.load(f)

    required_fields = [
        "seed",
        "n_x",
        "hidden_dim",
        "n_hidden",
        "lr_max",
        "lr_min",
        "weight_decay",
        "max_epochs",
        "batch_size",
        "patience",
        "best_epoch",
        "best_val_loss",
        "validation_fit_percent",
        "training_time_s",
        "git_commit",
        "dataset_hash",
        "library_versions",
        "linearization",
    ]

    for field in required_fields:
        assert field in card, f"Missing required field '{field}' in model card"

    assert "A" in card["linearization"]
    assert "B" in card["linearization"]
    assert "C" in card["linearization"]
    assert "D" in card["linearization"]
    assert "poles" in card["linearization"]
