"""Overview page describing the comparative study and physical plants."""

from __future__ import annotations

import streamlit as st

try:
    from dashboard.api_client import cached_list_systems, get_api_client
except ModuleNotFoundError:
    from api_client import cached_list_systems, get_api_client


def render_overview_page() -> None:
    """Render project overview, plant specifications, and live API connectivity."""
    st.title("Comparative System Identification Study")
    st.markdown(
        """
        ### Classical vs. Learned Dynamic System Identification
        This project benchmarks two contrasting paradigms for identifying dynamical
        systems from observed input-output data:
        1. **Classical Linear Identification**: Discrete-time autoregressive with
           exogenous input (**ARX**) via equation-error least squares, and continuous
           **Prony's Method** from calibration impulses.
        2. **Learned Continuous-Time Identification**: Continuous-depth **Neural
           Ordinary Differential Equations (Neural ODE)** trained under a minimal
           zero-equilibrium inductive bias and integrated with 4th-order Runge-Kutta.
        """
    )

    st.markdown("---")
    st.header("Physical Plants")

    client = get_api_client()
    health = client.check_health()

    if not health:
        st.warning(
            "API backend is unavailable. Please verify the FastAPI service is running."
        )
        return

    try:
        systems = cached_list_systems()
    except Exception as exc:
        st.error(f"Failed to fetch system catalog from API: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    equations = {
        "rc": {
            "ode": r"\tau \frac{dv}{dt} = u(t) - v(t)",
            "tf": r"H(s) = \frac{1}{\tau s + 1}",
            "desc": (
                "First-order electrical charging dynamics. "
                "Identifiable parameter: time constant tau."
            ),
        },
        "motor": {
            "ode": (
                r"J \frac{d\omega}{dt} = \frac{K_t}{R} u(t) - "
                r"\left(b + \frac{K_t K_e}{R}\right) \omega(t)"
            ),
            "tf": r"H(s) = \frac{K}{\tau s + 1}",
            "desc": (
                "Electromechanical angular velocity dynamics. "
                "Identifiable parameters: gain K and time constant tau."
            ),
        },
        "msd": {
            "ode": r"m \frac{d^2 x}{dt^2} + c \frac{dx}{dt} + k x = u(t)",
            "tf": r"H(s) = \frac{1}{m s^2 + c s + k}",
            "desc": (
                "Second-order resonant mechanical oscillator. "
                "Identifiable parameters: mass m, damping c, stiffness k."
            ),
        },
    }

    for idx, plant in enumerate(systems):
        plant_id = plant["id"]
        col = cols[idx % 3]
        eq_info = equations.get(plant_id, {"ode": "", "tf": "", "desc": ""})

        with col:
            st.subheader(plant["name"])
            st.markdown(f"**Plant Order:** {plant['order']}")
            st.markdown(
                f"**Input ($u$):** {plant['u_name']} ({plant['u_unit']}), "
                f"$u_{{max}} = {plant['u_max']}$"
            )
            st.markdown(f"**Output ($y$):** {plant['y_name']} ({plant['y_unit']})")
            w_band = plant["training_band_rad_s"]
            st.markdown(f"**Training Band:** `[{w_band[0]}, {w_band[1]}]` rad/s")

            st.markdown("**Governing ODE:**")
            st.latex(eq_info["ode"])
            st.markdown("**Transfer Function:**")
            st.latex(eq_info["tf"])
            st.caption(eq_info["desc"])

    st.markdown("---")
    st.header("Service Status & Provenance")
    st.markdown(f"**API Base URL:** `{client.base_url}`")
    st.markdown(
        f"**Backend Status:** `{health.get('status')}` (API v{health.get('version')})"
    )
    loaded_p = ", ".join(health.get("loaded_plants", []))
    st.markdown(f"**Loaded Physical Plants:** `{loaded_p}`")


if __name__ == "__main__":
    render_overview_page()
