"""Main Streamlit application entry point and navigation coordinator."""

from __future__ import annotations

import sys
from pathlib import Path

from pathlib import Path
import sys

import streamlit as st

# Ensure repository root and dashboard directory are present in sys.path
_DASHBOARD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DASHBOARD_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

try:
    from dashboard.api_client import get_api_client
except ModuleNotFoundError:
    from api_client import get_api_client  # noqa: E402

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

    # Multi-page navigation using explicit paths
    pages_dir = _DASHBOARD_DIR / "pages"
    pages = [
        st.Page(str(pages_dir / "1_Overview.py"), title="1. Overview"),
        st.Page(str(pages_dir / "2_Simulator.py"), title="2. Simulator Sandbox"),
        st.Page(
            str(pages_dir / "3_Frequency_and_Impulse.py"),
            title="3. Frequency & Impulse",
        ),
        st.Page(
            str(pages_dir / "4_Generalization_and_Robustness.py"),
            title="4. Generalization & Robustness",
        ),
        st.Page(
            str(pages_dir / "5_Methodology.py"),
            title="5. Methodology & Cards",
        ),
    ]

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()

