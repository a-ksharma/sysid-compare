"""Simulator sandbox page allowing live evaluation of plants and models."""

from __future__ import annotations

import streamlit as st
from dashboard.api_client import cached_get_system, cached_list_systems, get_api_client
from dashboard.plotting import plot_simulation_trajectories


def render_simulator_page() -> None:
    """Render interactive simulation control panel and trajectory visualizer."""
    st.title("Dynamic Response Simulator")
    st.markdown(
        "Drive physical plants and identified models under custom input waveforms, "
        "inspect tracking errors, and observe out-of-distribution amplitude warnings."
    )

    client = get_api_client()
    try:
        systems = cached_list_systems()
    except Exception as exc:
        st.error(f"Failed to reach API service: {exc}")
        return

    sys_options = {s["id"]: f"{s['name']} ({s['id']})" for s in systems}

    # Sidebar / controls
    st.sidebar.header("Simulation Settings")
    system_id = st.sidebar.selectbox(
        "Physical Plant",
        options=list(sys_options.keys()),
        format_func=lambda x: sys_options[x],
    )
    plant_detail = cached_get_system(system_id)
    u_max = float(plant_detail["u_max"])
    u_unit = plant_detail["u_unit"]
    u_name = plant_detail["u_name"]
    y_unit = plant_detail["y_unit"]
    y_name = plant_detail["y_name"]

    st.sidebar.markdown(f"**Nominal $u_{{max}}$:** {u_max:.1f} {u_unit}")

    # Input waveform selection
    input_kind = st.sidebar.selectbox(
        "Input Waveform Family",
        options=["step", "sine", "square", "chirp", "multisine"],
    )

    amp = st.sidebar.slider(
        f"Input Amplitude ({u_unit})",
        min_value=0.1,
        max_value=float(round(2.5 * u_max, 1)),
        value=float(round(u_max, 1)),
        step=0.1,
        help="Values > u_max will trigger API out-of-distribution warnings.",
    )

    freq_val = 2.0
    if input_kind in ("sine", "square", "chirp"):
        freq_val = st.sidebar.slider(
            "Waveform Frequency (rad/s)",
            min_value=0.1,
            max_value=30.0,
            value=2.0,
            step=0.1,
            help="Excitation training band was [0.1, 15.0] rad/s.",
        )

    default_dur = 6.0 if system_id == "rc" else (10.0 if system_id == "motor" else 15.0)
    duration_s = st.sidebar.slider(
        "Duration (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=default_dur,
        step=0.5,
    )

    noise_choice = st.sidebar.selectbox(
        "Output Measurement Noise SNR",
        options=[
            "Clean (No Noise)",
            "40 dB (Mild Noise)",
            "20 dB (Moderate Noise)",
            "10 dB (Severe Noise)",
        ],
    )
    snr_mapping = {
        "Clean (No Noise)": None,
        "40 dB (Mild Noise)": 40.0,
        "20 dB (Moderate Noise)": 20.0,
        "10 dB (Severe Noise)": 10.0,
    }
    noise_snr_db = snr_mapping[noise_choice]

    models_selected = st.sidebar.multiselect(
        "Models to Simulate",
        options=["arx", "prony", "neural_ode"],
        default=["arx", "prony", "neural_ode"],
    )

    if (
        st.sidebar.button("Run Simulation", type="primary")
        or "last_sim" not in st.session_state
    ):
        input_spec: dict[str, object] = {
            "kind": input_kind,
            "amplitude": amp,
        }
        if input_kind in ("sine", "square", "chirp"):
            input_spec["frequency_rad_s"] = freq_val

        with st.spinner("Simulating dynamics on FastAPI server..."):
            try:
                res = client.simulate(
                    system_id=system_id,
                    models=models_selected,
                    input_spec=input_spec,
                    duration_s=duration_s,
                    noise_snr_db=noise_snr_db,
                    seed=42,
                )
                st.session_state["last_sim"] = res
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

    sim_data = st.session_state.get("last_sim")
    if not sim_data:
        return

    # Warnings from API
    warnings = sim_data.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(f"API Range Notice: {w}")

    # Plot
    fig = plot_simulation_trajectories(
        t=sim_data["t"],
        u=sim_data["u"],
        y_true=sim_data["y_true"],
        y_measured=sim_data.get("y_measured"),
        predictions=sim_data["predictions"],
        u_name=u_name,
        u_unit=u_unit,
        y_name=y_name,
        y_unit=y_unit,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Performance Metrics")
    pred_data = sim_data["predictions"]
    metric_rows = []
    for m_id, m_pred in pred_data.items():
        metric_rows.append(
            {
                "Model": m_id.upper() if m_id != "neural_ode" else "Neural ODE",
                "Fit Percent (%)": f"{m_pred.get('fit_percent', 0.0):.2f}%",
                "Root-Mean-Squared Error": f"{m_pred.get('rmse', 0.0):.5f} {y_unit}",
            }
        )

    st.table(metric_rows)


if __name__ == "__main__":
    render_simulator_page()
