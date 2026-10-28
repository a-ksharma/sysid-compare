"""Frequency and impulse response page comparing Bode curves and poles."""

from __future__ import annotations

import streamlit as st

try:
    from dashboard.api_client import (
        cached_get_system,
        cached_list_systems,
        get_api_client,
    )
    from dashboard.plotting import plot_bode, plot_impulse_response
except ModuleNotFoundError:
    from api_client import (
        cached_get_system,
        cached_list_systems,
        get_api_client,
    )
    from plotting import plot_bode, plot_impulse_response


def render_frequency_and_impulse_page() -> None:
    """Render continuous Bode response, impulse response, and pole tables."""
    st.title("Frequency & Impulse Response")
    st.markdown(
        "Evaluate linear frequency response characteristics (Bode magnitude "
        "and phase), continuous impulse responses, and continuous poles."
    )

    client = get_api_client()
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
    plant = cached_get_system(system_id)
    training_band = plant.get("training_band_rad_s", [0.1, 15.0])
    y_unit = plant.get("y_unit", "")

    # 1. Frequency response
    st.header("Bode Diagram")
    st.caption("Training excitation band [0.1, 15.0] rad/s is shaded in grey.")

    col1, col2 = st.columns(2)
    with col1:
        omega_min = st.number_input(
            "Min Frequency (rad/s)", value=0.1, min_value=0.01, step=0.1
        )
    with col2:
        omega_max = st.number_input(
            "Max Frequency (rad/s)", value=60.0, min_value=1.0, step=5.0
        )

    try:
        bode_data = client.get_frequency_response(
            system_id=system_id,
            omega_min=float(omega_min),
            omega_max=float(omega_max),
            points=250,
        )
        all_curves = {"truth": bode_data["true_system"]}
        all_curves.update(bode_data["models"])
        bode_fig = plot_bode(
            omega=bode_data["omega"],
            curves=all_curves,
            training_band_rad_s=training_band,
        )
        st.plotly_chart(bode_fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to retrieve frequency response: {exc}")

    # 2. Impulse response
    st.header("Impulse Response")
    try:
        imp_data = client.get_impulse_response(system_id=system_id, points=200)
        imp_fig = plot_impulse_response(
            t=imp_data["t"],
            true_h=imp_data["true_system"],
            models_h=imp_data["models"],
            y_unit=y_unit,
        )
        st.plotly_chart(imp_fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Failed to retrieve impulse response: {exc}")

    # 3. Pole table
    st.header("Continuous-Time System Poles")
    st.caption("Extracted continuous poles s = sigma + j*omega.")

    true_poles = plant.get("true_poles", [])
    identified = plant.get("identified", {})

    pole_rows = []
    # Ground truth poles
    for p in true_poles:
        pole_rows.append(
            {
                "Model": "Ground Truth",
                "Real Part (sigma)": f"{p['real']:.4f}",
                "Imag Part (j*omega)": f"{p['imag']:.4f}",
                "Damping / Frequency": (
                    f"|p| = {(p['real'] ** 2 + p['imag'] ** 2) ** 0.5:.4f} rad/s"
                ),
            }
        )

    # Identified models
    for m_key, m_info in identified.items():
        m_label = m_key.upper() if m_key != "neural_ode" else "Neural ODE"
        for p in m_info.get("poles", []):
            mag_p = (p.get("real", 0.0) ** 2 + p.get("imag", 0.0) ** 2) ** 0.5
            pole_rows.append(
                {
                    "Model": m_label,
                    "Real Part (sigma)": f"{p.get('real', 0.0):.4f}",
                    "Imag Part (j*omega)": f"{p.get('imag', 0.0):.4f}",
                    "Damping / Frequency": f"|p| = {mag_p:.4f} rad/s",
                }
            )

    st.table(pole_rows)


if __name__ == "__main__":
    render_frequency_and_impulse_page()
