"""HTTP client communicating with the System Identification FastAPI backend."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def get_base_url() -> str:
    """Resolve API base URL from Streamlit secrets or environment variable."""
    # 1. Streamlit secrets
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass

    # 2. Environment variable
    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    # 3. Default local development server
    return DEFAULT_API_BASE_URL


class APIClient:
    """Client for querying the FastAPI system identification service."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or get_base_url()).rstrip("/")
        self.api_v1 = f"{self.base_url}/api/v1"

    def check_health(self, timeout_s: float = 3.0) -> dict[str, Any] | None:
        """Probe the API health endpoint."""
        url = f"{self.api_v1}/health"
        try:
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass
        return None

    def wake_service(
        self, max_attempts: int = 12, interval_s: float = 5.0
    ) -> tuple[bool, str]:
        """Poll the API until it wakes up (Render free tier cold start)."""
        url = f"{self.api_v1}/health"
        start_time = time.time()
        for attempt in range(1, max_attempts + 1):
            elapsed = int(time.time() - start_time)
            try:
                resp = requests.get(url, timeout=interval_s)
                if resp.status_code == 200:
                    return True, f"API is online (connected after {elapsed}s)."
            except requests.RequestException:
                pass
            time.sleep(interval_s)

        return (
            False,
            (
                f"Unable to reach API at {self.base_url} after "
                f"{max_attempts * interval_s}s. "
                "Please verify the backend service is running."
            ),
        )

    def list_systems(self) -> list[dict[str, Any]]:
        """Fetch catalog of all physical plants."""
        resp = requests.get(f"{self.api_v1}/systems", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def get_system(self, system_id: str) -> dict[str, Any]:
        """Fetch details, true parameters, and poles for a specific plant."""
        resp = requests.get(f"{self.api_v1}/systems/{system_id}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def get_models(self) -> dict[str, Any]:
        """Fetch all model cards and training metadata."""
        resp = requests.get(f"{self.api_v1}/models", timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("cards", {})

    def simulate(
        self,
        system_id: str,
        models: list[str],
        input_spec: dict[str, Any],
        duration_s: float,
        noise_snr_db: float | None = None,
        seed: int = 1,
    ) -> dict[str, Any]:
        """Run trajectory simulation for selected plant and models."""
        payload: dict[str, Any] = {
            "system_id": system_id,
            "models": models,
            "input": input_spec,
            "duration_s": duration_s,
            "noise_snr_db": noise_snr_db,
            "seed": seed,
        }
        resp = requests.post(f"{self.api_v1}/simulate", json=payload, timeout=20.0)
        if resp.status_code == 422:
            data = resp.json()
            err_msg = data.get("error", {}).get("message", "Validation error.")
            raise ValueError(err_msg)
        resp.raise_for_status()
        return resp.json()

    def get_frequency_response(
        self,
        system_id: str,
        models: list[str] | None = None,
        omega_min: float = 0.1,
        omega_max: float = 60.0,
        points: int = 200,
    ) -> dict[str, Any]:
        """Fetch continuous frequency response (Bode data)."""
        params: dict[str, Any] = {
            "omega_min": omega_min,
            "omega_max": omega_max,
            "points": points,
        }
        if models:
            params["models"] = models
        resp = requests.get(
            f"{self.api_v1}/frequency-response/{system_id}",
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_impulse_response(
        self,
        system_id: str,
        models: list[str] | None = None,
        points: int = 200,
    ) -> dict[str, Any]:
        """Fetch continuous impulse response trajectories."""
        params: dict[str, Any] = {"points": points}
        if models:
            params["models"] = models
        resp = requests.get(
            f"{self.api_v1}/impulse-response/{system_id}",
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_results(self, system_id: str) -> dict[str, Any]:
        """Fetch precomputed evaluation benchmark data for plant."""
        resp = requests.get(f"{self.api_v1}/results/{system_id}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()


# Cached singleton instances for Streamlit pages
@st.cache_resource
def get_api_client() -> APIClient:
    """Return cached API client singleton."""
    return APIClient()


@st.cache_data(ttl=600)
def cached_list_systems() -> list[dict[str, Any]]:
    """Cached wrapper for list_systems."""
    return get_api_client().list_systems()


@st.cache_data(ttl=600)
def cached_get_system(system_id: str) -> dict[str, Any]:
    """Cached wrapper for get_system."""
    return get_api_client().get_system(system_id)


@st.cache_data(ttl=600)
def cached_get_models() -> dict[str, Any]:
    """Cached wrapper for get_models."""
    return get_api_client().get_models()


@st.cache_data(ttl=600)
def cached_get_results(system_id: str) -> dict[str, Any]:
    """Cached wrapper for get_results."""
    return get_api_client().get_results(system_id)
