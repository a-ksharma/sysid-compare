"""Plotly visualization helpers adhering to Okabe-Ito palette and signals standards."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Strict Okabe-Ito color palette from spec section 11
COLOR_MAP: dict[str, str] = {
    "truth": "#000000",
    "true_system": "#000000",
    "plant": "#000000",
    "arx": "#0072B2",
    "prony": "#E69F00",
    "neural_ode": "#D55E00",
    "lstm": "#009E73",
}

MODEL_NAMES: dict[str, str] = {
    "truth": "Ground Truth",
    "true_system": "Ground Truth",
    "plant": "Ground Truth",
    "arx": "ARX (Least Squares)",
    "prony": "Prony Method",
    "neural_ode": "Neural ODE",
    "lstm": "LSTM Baseline",
}


def get_color(key: str) -> str:
    """Return Okabe-Ito hex color for a given model or plant identifier."""
    return COLOR_MAP.get(key.lower(), "#56B4E9")


def get_label(key: str) -> str:
    """Return plain domain label for model or plant identifier."""
    return MODEL_NAMES.get(key.lower(), key)


def plot_simulation_trajectories(
    t: list[float],
    u: list[float],
    y_true: list[float],
    y_measured: list[float] | None,
    predictions: dict[str, Any],
    u_name: str,
    u_unit: str,
    y_name: str,
    y_unit: str,
) -> go.Figure:
    """Generate time-domain simulation figure with outputs and residual error."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Input Excitation: {u_name} ({u_unit})",
            f"Output Response: {y_name} ({y_unit})",
            f"Residual Tracking Error: y_pred - y_true ({y_unit})",
        ),
    )

    # 1. Input signal
    fig.add_trace(
        go.Scatter(
            x=t,
            y=u,
            mode="lines",
            name="Input u(t)",
            line=dict(color="#444444", width=1.5),
        ),
        row=1,
        col=1,
    )

    # 2. Ground truth output
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_true,
            mode="lines",
            name="Ground Truth Plant",
            line=dict(color=get_color("truth"), width=2.5),
        ),
        row=2,
        col=1,
    )

    # Optional noisy measurement
    if y_measured is not None:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_measured,
                mode="lines",
                name="Measured Output (Noisy)",
                line=dict(color="#999999", width=1, dash="dot"),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

    # Predictions and residuals
    for m_key, pred in predictions.items():
        y_pred = pred.get("y", [])
        m_color = get_color(m_key)
        m_label = get_label(m_key)
        fit_pct = pred.get("fit_percent", 0.0)

        # Output trajectory
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_pred,
                mode="lines",
                name=f"{m_label} (Fit: {fit_pct:.1f}%)",
                line=dict(color=m_color, width=1.8),
            ),
            row=2,
            col=1,
        )

        # Residual
        if len(y_pred) == len(y_true):
            error = [yp - yt for yp, yt in zip(y_pred, y_true)]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=error,
                    mode="lines",
                    name=f"{m_label} Error",
                    line=dict(color=m_color, width=1.2, dash="dash"),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text=f"u ({u_unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"y ({y_unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"Error ({y_unit})", row=3, col=1)

    fig.update_layout(
        height=750,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
    )
    return fig


def plot_bode(
    omega: list[float],
    curves: dict[str, Any],
    training_band_rad_s: list[float] | None = None,
) -> go.Figure:
    """Generate Bode magnitude (dB) and phase (deg) frequency response curves."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Bode Magnitude Response (dB)",
            "Bode Phase Response (degrees)",
        ),
    )

    for m_key, curve in curves.items():
        mag_db = curve.get("magnitude_db", [])
        phase_deg = curve.get("phase_deg", [])
        color = get_color(m_key)
        label = get_label(m_key)
        width = 2.5 if m_key in ("truth", "true_system", "plant") else 1.8

        # Magnitude trace
        fig.add_trace(
            go.Scatter(
                x=omega,
                y=mag_db,
                mode="lines",
                name=label,
                line=dict(color=color, width=width),
            ),
            row=1,
            col=1,
        )

        # Phase trace
        fig.add_trace(
            go.Scatter(
                x=omega,
                y=phase_deg,
                mode="lines",
                name=label,
                line=dict(color=color, width=width),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Shaded excitation training band
    if training_band_rad_s and len(training_band_rad_s) == 2:
        w_min, w_max = training_band_rad_s
        for r in (1, 2):
            fig.add_vrect(
                x0=w_min,
                x1=w_max,
                fillcolor="#CCCCCC",
                opacity=0.25,
                line_width=0,
                annotation_text="Training Band" if r == 1 else None,
                annotation_position="top left",
                row=r,  # type: ignore[arg-type]
                col=1,  # type: ignore[arg-type]
            )

    fig.update_xaxes(type="log", title_text="Frequency omega (rad/s)", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)

    fig.update_layout(
        height=650,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
    )
    return fig


def plot_impulse_response(
    t: list[float],
    true_h: list[float],
    models_h: dict[str, list[float]],
    y_unit: str,
) -> go.Figure:
    """Generate impulse response overlay."""
    fig = go.Figure()

    # Ground truth
    fig.add_trace(
        go.Scatter(
            x=t,
            y=true_h,
            mode="lines",
            name=get_label("truth"),
            line=dict(color=get_color("truth"), width=2.5),
        )
    )

    # Models
    for m_key, h in models_h.items():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=h,
                mode="lines",
                name=get_label(m_key),
                line=dict(color=get_color(m_key), width=1.8),
            )
        )

    fig.update_layout(
        title="Continuous-Time Impulse Response Comparison h(t)",
        xaxis_title="Time (s)",
        yaxis_title=f"Impulse Response ({y_unit}/unit-impulse)",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
    )
    return fig


def plot_ood_bar_chart(ood_results: dict[str, Any]) -> go.Figure:
    """Generate grouped bar chart comparing OOD fit percent across conditions."""
    fig = go.Figure()

    # Extract conditions
    conditions = sorted(list(ood_results.keys()))
    models = ["arx", "prony", "neural_ode"]

    for m_key in models:
        fits = []
        for cond in conditions:
            cond_data = ood_results.get(cond, {}).get(m_key, {})
            fit_val = cond_data.get("fit_percent", 0.0)
            fits.append(fit_val)

        fig.add_trace(
            go.Bar(
                name=get_label(m_key),
                x=conditions,
                y=fits,
                marker_color=get_color(m_key),
            )
        )

    fig.update_layout(
        barmode="group",
        title="Out-of-Distribution (OOD) Validation Fit Percent",
        xaxis_title="Excitation Condition",
        yaxis_title="Goodness-of-Fit (%)",
        yaxis_range=[min(min(fig.data[0].y or [0.0]) - 5.0, 0.0), 105.0],
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
    )
    return fig


def plot_noise_sweep_curve(noise_sweep: dict[str, Any]) -> go.Figure:
    """Generate noise sweep curve showing fit degradation with measurement noise."""
    fig = go.Figure()

    # Levels: clean (inf), 40dB, 20dB, 10dB
    snr_keys = ["inf", "40", "20", "10"]
    snr_labels = ["Clean (inf dB)", "40 dB", "20 dB", "10 dB"]
    models = ["arx", "prony", "neural_ode"]

    for m_key in models:
        fits = []
        for snr_k in snr_keys:
            fit_val = noise_sweep.get(snr_k, {}).get(m_key, {}).get("fit_percent", 0.0)
            fits.append(fit_val)

        fig.add_trace(
            go.Scatter(
                x=snr_labels,
                y=fits,
                mode="lines+markers",
                name=get_label(m_key),
                line=dict(color=get_color(m_key), width=2.0),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        title="Test Fit Percent vs Measurement Output Noise Level",
        xaxis_title="Signal-to-Noise Ratio (SNR)",
        yaxis_title="Goodness-of-Fit (%)",
        yaxis_range=[-105.0, 105.0],
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        template="plotly_white",
    )
    return fig
