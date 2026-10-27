"""Neural ODE model architecture and vector field definition.

Implements the zero-equilibrium prior f(x, u) = g(x, u) - g(0, 0) and
linear readout y_n = C * x without bias, integrated via midpoint torchdiffeq.
"""

from __future__ import annotations

from typing import Callable, cast

import torch
import torch.nn as nn
import torchdiffeq


class MLPVectorField(nn.Module):
    """MLP parameterizing g(x, u) with zero-equilibrium prior.

    Enforces:
        f(x, u) = g(x, u) - g(0, 0)
    so that f(0, 0) = 0 identically (Decision D7).
    """

    def __init__(
        self,
        n_x: int = 4,
        n_u: int = 1,
        hidden_dim: int = 64,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.n_x = n_x
        self.n_u = n_u

        layers: list[nn.Module] = []
        in_dim = n_x + n_u
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_x))
        self.net = nn.Sequential(*layers)

        self.register_buffer("zero_in", torch.zeros(1, n_x + n_u))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Evaluate continuous vector field f(x, u).

        Parameters
        ----------
        x : torch.Tensor
            Latent state tensor of shape (batch_size, n_x).
        u : torch.Tensor
            Normalized input tensor of shape (batch_size, 1).

        Returns
        -------
        torch.Tensor
            State derivative dx/dt of shape (batch_size, n_x).
        """
        xu = torch.cat([x, u], dim=-1)
        g_xu = self.net(xu)
        g_00 = self.net(self.zero_in)
        return cast(torch.Tensor, g_xu - g_00)


class NeuralODE(nn.Module):
    """Neural ODE dynamic model with linear readout.

    Parameters
    ----------
    n_x : int
        Latent state dimension (default 4).
    hidden_dim : int
        Number of hidden units per layer in vector field MLP (default 64).
    n_hidden : int
        Number of hidden layers (default 2).
    Ts : float
        Nominal sampling period in seconds (default 0.02).
    u_scale : float
        Input physical scaling factor u = u_norm * u_scale.
    y_scale : float
        Output physical scaling factor y = y_norm * y_scale.
    """

    def __init__(
        self,
        n_x: int = 4,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        Ts: float = 0.02,
        u_scale: float = 1.0,
        y_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_x = n_x
        self.Ts = Ts
        self.u_scale = u_scale
        self.y_scale = y_scale

        self.vector_field = MLPVectorField(
            n_x=n_x,
            n_u=1,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        )
        # Decision K6 & K17: linear readout without bias (y(0) = C * 0 = 0)
        self.readout = nn.Linear(n_x, 1, bias=False)

    def forward(
        self,
        u_norm: torch.Tensor,
        t_grid: torch.Tensor | None = None,
        eval_callback: Callable[[float], None] | None = None,
    ) -> torch.Tensor:
        """Simulate free-run trajectories from rest (x0 = 0) over the time grid.

        Parameters
        ----------
        u_norm : torch.Tensor
            Normalized input batch of shape (batch_size, n_samples).
        t_grid : torch.Tensor | None, optional
            Time grid points in seconds. If None, constructed as k * Ts.
        eval_callback : Callable[[float], None] | None, optional
            Optional evaluation-time recording callback for ZOH boundary testing.

        Returns
        -------
        torch.Tensor
            Simulated normalized output y_norm of shape (batch_size, n_samples).
        """
        batch_size, n_samples = u_norm.shape
        device = u_norm.device

        if t_grid is None:
            t_grid = (
                torch.arange(n_samples, device=device, dtype=torch.float32) * self.Ts
            )

        x0 = torch.zeros(batch_size, self.n_x, device=device, dtype=torch.float32)
        step_size = self.Ts / 2.0

        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            t_val = float(t.item())
            if eval_callback is not None:
                eval_callback(t_val)

            # Spec 7.2: idx = floor(t/Ts + 1e-6) clipped to last sample
            idx = int(
                min(max(0, int(torch.floor(t / self.Ts + 1e-6).item())), n_samples - 1)
            )
            u_t = u_norm[:, idx : idx + 1]
            return cast(torch.Tensor, self.vector_field(x, u_t))

        # torchdiffeq midpoint integration with step_size = Ts / 2
        # Returns states of shape (n_samples, batch_size, n_x)
        x_traj = torchdiffeq.odeint(
            ode_func,
            x0,
            t_grid,
            method="midpoint",
            options={"step_size": step_size},
        )

        # Permute to (batch_size, n_samples, n_x)
        x_perm = x_traj.permute(1, 0, 2)
        y_norm = self.readout(x_perm).squeeze(-1)
        return cast(torch.Tensor, y_norm)
