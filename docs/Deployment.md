# Deployment

Targets: the FastAPI service on Render (free web service), the Streamlit dashboard on Streamlit Community Cloud. Both deploy from the same GitHub repository.

Items marked **[VERIFY]** were not confirmed against the vendor's current documentation when this file was written. Check them on the vendor's site at deploy time and update this file if they differ.

## 1. Platform constraints (as of October 2026, from public pricing pages and docs)

| Constraint | Render free web service | Effect on this project |
|------------|------------------------|------------------------|
| Memory and CPU | 512 MB RAM, 0.1 CPU | NumPy-only inference (Decision D11); no PyTorch in `requirements-api.txt`; measure peak memory in Phase 7. |
| Idle behaviour | Spins down after 15 minutes without inbound traffic; about one minute to wake | The dashboard handles a slow first call (Decision D14). |
| Local disk | Lost on every spin-down | Nothing is written at runtime; artifacts ship in the repository. |
| Instance hours | A monthly allowance shared by all free services in the workspace | Do not run a keep-alive pinger. One always-awake service consumes the whole allowance. |

Streamlit Community Cloud: apps are deployed from a GitHub repository, installed from the requirements file in the repository, limited in resources, and hibernated after inactivity. Keep the dashboard light: no PyTorch, no model files. **[VERIFY]** the current resource limits and the hibernation period.

## 2. Repository preparation (before any deployment)

1. `make test` and `make lint` pass.
2. `artifacts/models/*` and `artifacts/results/*` are committed. `data/` is not.
3. `requirements-api.txt` contains only: fastapi, uvicorn, pydantic, numpy, scipy (compatible-release pins, for example `numpy~=2.1`; choose the versions that the local API environment resolved in Phase 7).
4. `dashboard/requirements.txt` contains only: streamlit, plotly, requests (pin streamlit; the Streamlit docs recommend pinning its version).
5. No secrets in the repository. `dashboard/.streamlit/secrets.toml` is git-ignored.
6. The API reads its configuration from environment variables: `ALLOWED_ORIGINS`, `LOG_LEVEL`, `PYTHONPATH`.

## 3. API on Render

### 3.1 `render.yaml` (Blueprint at the repository root)
```yaml
services:
  - type: web
    name: sysid-api
    runtime: python
    plan: free
    region: singapore            # gate G5; pick the nearest region Render offers [VERIFY]
    buildCommand: pip install -r requirements-api.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /api/v1/health
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9            # full version string [VERIFY]
      - key: PYTHONPATH
        value: src
      - key: LOG_LEVEL
        value: INFO
      - key: ALLOWED_ORIGINS
        value: https://<your-dashboard>.streamlit.app
```
Render's web services must listen on `0.0.0.0` on the port given by the `PORT` environment variable. **[VERIFY]** the `PYTHON_VERSION` mechanism and the available regions.

### 3.2 Steps
1. Push the repository to GitHub.
2. In the Render dashboard create a new Blueprint, connect the repository, and select `render.yaml`. (Alternative: create a Web Service by hand with the same build and start commands and the same environment variables.)
3. Wait for the build. Open `https://<service>.onrender.com/api/v1/health` and `/docs`.
4. Run `python scripts/smoke_test.py --base-url https://<service>.onrender.com`. Record the response time of the first (cold) call and of a warm call in the README.
5. Leave `ALLOWED_ORIGINS` as a placeholder until the dashboard URL exists, then set it and redeploy. (The Streamlit server calls the API from its own backend, so CORS only matters for browser-side callers, but keep it correct.)

### 3.3 If the build or the instance fails
- Out-of-memory on start: check `requirements-api.txt` for an accidental PyTorch or Matplotlib dependency; measure memory locally with the same requirements.
- Import errors: confirm `PYTHONPATH=src` is set.
- 502 on first request: the instance is still waking; wait one minute and retry.

## 4. Dashboard on Streamlit Community Cloud

### 4.1 Steps
1. Make sure the repository is on GitHub and the Streamlit Community Cloud account can read it (public, or private with access granted).
2. In Streamlit Community Cloud choose "Create app", select the repository, the branch, and the main file path `dashboard/app.py`. The requirements file is `dashboard/requirements.txt`. **[VERIFY]** where the platform looks for the requirements file (entrypoint directory versus repository root); if it only reads the root, move the file to the root and keep it dashboard-only.
3. In the app's advanced settings choose the Python version (3.11) and paste the secrets in TOML form:
```toml
API_BASE_URL = "https://<service>.onrender.com"
```
4. Deploy. When the app is up, copy its URL into `ALLOWED_ORIGINS` on Render.
5. Dependency changes are made by editing the requirements file and pushing; the platform reinstalls automatically.

### 4.2 Local development
Create `dashboard/.streamlit/secrets.toml` (git-ignored) with `API_BASE_URL = "http://localhost:8000"`, start the API with `make api`, then `make dashboard`.

## 5. Post-deployment checks

1. Open the dashboard in a private window. If the API was asleep, the "waking the API" message appears and the page loads within about a minute.
2. On each of the five pages, confirm that the charts render and the numbers match `artifacts/results/` (spot-check three values).
3. In the Simulator, request an amplitude above the training range and confirm the warning is shown.
4. Check the Render logs for errors and the Streamlit app logs for exceptions.
5. Put both live URLs and the measured cold and warm response times in the README.

## 6. Updating a deployed system
- Retrained or re-exported models: commit the new `artifacts/` files, push; Render redeploys automatically. Redeploys wipe nothing that matters because nothing is written at runtime.
- API schema change: bump the version prefix only for breaking changes; update the dashboard client in the same commit.
- Rollback: redeploy the previous commit from the Render dashboard; revert the dashboard commit on GitHub.

## 7. What not to do
- Do not add a cron job or external pinger to keep the free API awake.
- Do not commit `.streamlit/secrets.toml` or any API key.
- Do not install PyTorch, Matplotlib, or `python-control` in the API or dashboard environments.
- Do not change the API response shape without updating the dashboard client and the tests.
