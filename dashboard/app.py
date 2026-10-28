"""Main Streamlit application entry point and navigation coordinator."""

from __future__ import annotations

import streamlit as st
from dashboard.api_client import get_api_client

st.set_page_config(
    page_title="System Identification: Classical vs Learned",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Coordinate application navigation and backend service wake state."""
    client = get_api_client()

    # Sidebar service connection indicator
    st.sidebar.title("Navigation")
    health = client.check_health(timeout_s=2.0)

    if not health:
        st.sidebar.error("API: Offline / Sleeping")
        st.warning(
            "The backend API service is offline or waking up from idle on "
            "Render free tier. Please wake the service to enable live "
            "simulations and dynamic plots."
        )
        if st.button("Wake API Service", type="primary"):
            with st.spinner("Pinging API service (this can take up to 60 seconds)..."):
                success, msg = client.wake_service(max_attempts=12, interval_s=5.0)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.sidebar.success(f"API: Connected (v{health.get('version', '0.1.0')})")

    # Multi-page navigation
    pages = [
        st.Page("pages/1_Overview.py", title="1. Overview"),
        st.Page("pages/2_Simulator.py", title="2. Simulator Sandbox"),
        st.Page("pages/3_Frequency_and_Impulse.py", title="3. Frequency & Impulse"),
        st.Page(
            "pages/4_Generalization_and_Robustness.py",
            title="4. Generalization & Robustness",
        ),
        st.Page("pages/5_Methodology.py", title="5. Methodology & Cards"),
    ]

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
