"""Generalization and robustness page displaying benchmark results."""

from __future__ import annotations

import streamlit as st

try:
    from dashboard.api_client import (
        cached_get_results,
        cached_list_systems,
    )
    from dashboard.plotting import (
        plot_noise_sweep_curve,
        plot_ood_bar_chart,
    )
except ModuleNotFoundError:
    from api_client import (
        cached_get_results,
        cached_list_systems,
    )
    from plotting import (
        plot_noise_sweep_curve,
        plot_ood_bar_chart,
    )


def render_robustness_page() -> None:
    """Render OOD generalization, noise sensitivity, and cost benchmarks."""
    st.title("Generalization & Robustness")
    st.markdown(
        "Quantitative benchmarks evaluating model generalization under "
        "out-of-distribution (OOD) inputs, measurement noise, and plant variations."
    )

    try:
        systems = cached_list_systems()
    except Exception as exc:
        st.error(f"Failed to fetch system catalog: {exc}")
        return

    sys_options = {s["id"]: f"{s['name']} ({s['id']})" for s in systems}
    system_id = st.sidebar.selectbox(
        "Physical Plant",
        options=list(sys_options.keys()),
        format_func=lambda x: sys_options[x],
    )

    try:
        results = cached_get_results(system_id)
    except Exception as exc:
        st.error(f"Failed to fetch evaluation results for {system_id}: {exc}")
        return

    # 1. OOD Generalization
    st.header("1. Out-of-Distribution Generalization")
    st.markdown(
        "Goodness-of-fit evaluated on 8 distinct test waveforms: in-band "
        "and out-of-band sine sweeps, amplitude scaling, and novel shapes."
    )
    ood_data = results.get("ood", {})
    if ood_data:
        ood_fig = plot_ood_bar_chart(ood_data)
        st.plotly_chart(ood_fig, use_container_width=True)

    # 2. Noise Robustness Sweep
    st.header("2. Measurement Noise Sensitivity")
    st.markdown(
        "Degradation of test goodness-of-fit across additive Gaussian "
        "measurement noise levels: clean (inf dB), 40 dB, 20 dB, and 10 dB SNR."
    )
    noise_data = results.get("noise_sweep", {})
    if noise_data:
        noise_fig = plot_noise_sweep_curve(noise_data)
        st.plotly_chart(noise_fig, use_container_width=True)

    # 3. Parameter Variation
    st.header("3. Parameter Perturbation Sensitivity")
    st.markdown(
        "Prony failure rate and recovery performance across 10 perturbed plants "
        "drawn uniformly within +/- 10% of nominal parameter values."
    )
    pv_data = results.get("param_variation", {})
    if pv_data:
        prony_fail_count = pv_data.get("prony_failure_count", 0)
        st.markdown(
            f"**Prony Perturbation Failures:** `{prony_fail_count} / 10 plants`"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "ARX Mean Pole Error",
                f"{pv_data.get('arx_mean_pole_error', 0.0) * 100:.2f}%",
            )
        with col2:
            st.metric(
                "Prony Mean Pole Error (Non-failing)",
                f"{pv_data.get('prony_mean_pole_error', 0.0) * 100:.2f}%",
            )

    # 4. Computational Cost
    st.header("4. Computational Efficiency & Complexity")
    cost_data = results.get("cost", {})
    if cost_data:
        inf_times = cost_data.get("inference_time_ms_per_1000_samples", {})
        param_counts = cost_data.get("parameter_counts", {})
        train_times = cost_data.get("training_time_s", {})

        cost_rows = []
        for m_key in ["arx", "prony", "neural_ode"]:
            m_label = m_key.upper() if m_key != "neural_ode" else "Neural ODE"
            cost_rows.append(
                {
                    "Model": m_label,
                    "Parameters": param_counts.get(m_key, "N/A"),
                    "Training Duration": f"{train_times.get(m_key, 0.0):.2f} s",
                    "Latency (per 1k samples)": f"{inf_times.get(m_key, 0.0):.2f} ms",
                }
            )
        st.table(cost_rows)


if __name__ == "__main__":
    render_robustness_page()
