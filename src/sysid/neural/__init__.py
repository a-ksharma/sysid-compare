"""Neural ODE learned model track: PyTorch training and NumPy inference."""

from __future__ import annotations

from sysid.neural.export import export_neural_ode
from sysid.neural.model import MLPVectorField, NeuralODE
from sysid.neural.numpy_infer import NumpyNeuralODE, load_numpy_neural_ode
from sysid.neural.train import compute_fit_percent, train_neural_ode

__all__ = [
    "MLPVectorField",
    "NeuralODE",
    "NumpyNeuralODE",
    "compute_fit_percent",
    "export_neural_ode",
    "load_numpy_neural_ode",
    "train_neural_ode",
]
