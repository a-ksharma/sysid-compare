"""Methodology page documenting identifiability, estimators, and limits."""

from __future__ import annotations

import streamlit as st
from dashboard.api_client import cached_get_models


def render_methodology_page() -> None:
    """Render identifiability, estimator theories, model cards, and limits."""
    st.title("Methodology & Mathematical Provenance")

    # 1. Identifiability
    st.header("1. Input-Output Identifiability")
    st.markdown(
        """
        From input-output observations alone, only the coefficients of the transfer
        function $H(s)$ or similarity-invariant Markov parameters can be recovered.
        Internal state-space realizations are unique only up to coordinate similarity.
        """
    )
    identifiability_rows = [
        {
            "Plant": "RC Low-Pass (Order 1)",
            "Governing ODE": r"\tau \dot{v} + v = u",
            "Identifiable Quantities": r"\tau = R C",
            "Unidentifiable Parameters": "R and C individually",
        },
        {
            "Plant": "DC Motor (Order 1)",
            "Governing ODE": r"\tau \dot{\omega} + \omega = K u",
            "Identifiable Quantities": r"K, \tau",
            "Unidentifiable Parameters": "K_t, K_e, b, J separately",
        },
        {
            "Plant": "Mass-Spring-Damper (Order 2)",
            "Governing ODE": r"m \ddot{x} + c \dot{x} + k x = u",
            "Identifiable Quantities": "m, c, k (all three)",
            "Unidentifiable Parameters": "None (DC and HF limits fix scale)",
        },
    ]
    st.table(identifiability_rows)

    # 2. Estimator Formulation
    st.header("2. Identification Estimators")

    st.subheader("Classical ARX (Equation-Error Least Squares)")
    st.markdown(
        r"""
        The AutoRegressive with eXogenous input model estimates difference equations:
        $$y[k] + \sum_{i=1}^{n_a} a_i y[k-i] = \sum_{j=1}^{n_b} b_j u[k-j]$$
        The parameter vector $\theta$ is solved via ordinary least squares:
        $$\hat{\theta} = (\Phi^T \Phi)^{-1} \Phi^T Y$$
        **Noise Bias Vulnerability**: When measurement noise contaminates the output,
        regressors in $\Phi$ contain lagged noise terms correlated with equation error,
        violating Gauss-Markov orthogonality and inducing severe asymptotic bias.
        """
    )

    st.subheader("Classical Prony Method")
    st.markdown(
        r"""
        Prony's method fits a sum of complex damped exponentials to an impulse response:
        $$h[k] = \sum_{i=1}^n R_i z_i^k$$
        It solves linear prediction equations for poles $z_i$, followed by a
        Vandermonde system for residues $R_i$.
        **Sensitivity**: Exact on noise-free calibration pulses ($< 10^{-12}$ error),
        but ill-conditioned or unstable under noise or perturbation.
        """
    )

    st.subheader("Learned Neural ODE (Runge-Kutta 4th Order)")
    st.markdown(
        r"""
        The Neural ODE models vector field $\dot{x}(t) = f_\theta(x(t), u(t))$
        via a Multi-Layer Perceptron (MLP) with $\tanh$ activations:
        $$\dot{x}(t) = W_3 \tanh(W_2 \tanh(W_1 [x; u] + b_1) + b_2)$$
        **Minimal Zero-Equilibrium Prior**: Omission of bias $b_3$ guarantees
        $f_\theta(0, 0) = 0$, enforcing that the origin is an invariant resting state.
        Simulation uses explicit 4th-order Runge-Kutta (RK4) integration with ZOH.
        **Noise Robustness**: Continuous integration filters high-frequency noise.
        """
    )

    # 3. Model Cards
    st.header("3. Model Cards & Provenance")
    try:
        cards = cached_get_models()
        for p_id, p_cards in cards.items():
            with st.expander(f"Model Cards: {p_id.upper()}"):
                st.json(p_cards)
    except Exception as exc:
        st.error(f"Failed to fetch model cards: {exc}")

    # 4. Limitations
    st.header("4. Project Limitations")
    st.markdown(
        """
        To maintain research rigor, the following constraints must be noted:
        1. **Simulated Data**: Trajectories are generated from known ODE equations.
        2. **Linear Time-Invariant**: Real physical systems exhibit nonlinearities.
        3. **White Output Noise**: The noise model is purely additive output noise.
        4. **Seed Sensitivity**: Neural ODE training depends on initial weight seeds.
        5. **Free-Tier Sleep**: Free Render instances sleep when idle (cold start ~60s).
        """
    )


if __name__ == "__main__":
    render_methodology_page()
