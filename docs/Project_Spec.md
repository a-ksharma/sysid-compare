# Project Spec: Classical vs. Learned System Identification

Read `Decisions.md` first. Where this file and `Decisions.md` disagree, `Decisions.md` wins. If anything here is unclear or contradictory, stop and ask the user. Do not assume.

## 1. Purpose and scope

Build a comparative study of two ways to identify a physical dynamical system from input-output data:

1. Classical (grey-box): the model structure comes from physics (transfer function of known order); parameters are estimated by least squares (ARX) and by Prony's method.
2. Learned (black-box): a Neural ODE learns the dynamics from data with no assumed equations.

Both are compared on impulse response accuracy, Bode plot fidelity, generalization outside the training distribution, and robustness to noise and parameter variation. The result is a research package, a FastAPI service that serves the models, and a Streamlit dashboard that uses the API.

Non-goals: nonlinear plants, MIMO systems, closed-loop identification, hardware data, real-time inference.

## 2. Conventions

- Language: Python 3.11. Source layout: `src/sysid/`.
- Time: seconds. Frequency: rad/s. Gains in dB are 20*log10(magnitude). Phase in degrees.
- Sampling period `Ts = 0.02 s`. Sample k is at time `t = k*Ts`.
- Inputs are zero-order hold (ZOH): `u(t) = u[k]` for `k*Ts <= t < (k+1)*Ts`.
- Discrete model: `x[k+1] = Ad x[k] + Bd u[k]`, `y[k] = C x[k]`, `D = 0`. All sequences start at rest: `x[0] = 0`, so `y[0] = 0`.
- Notation follows the course: A, B, C, D state-space matrices; H(s) transfer function; h(t) impulse response; tau time constant; zeta damping ratio; omega_n natural frequency.
- Normalized units: the Neural ODE trains on `u_n = u/u_scale`, `y_n = y/y_scale`. Exported linearizations are rescaled to physical units (section 7.5).

## 3. Plants

Configuration lives in `configs/systems.yaml`. One registry entry per plant. No plant constants appear anywhere else in the code.

| id | Plant | Order | Input | Output | Nominal parameters | u_max |
|----|-------|-------|-------|--------|--------------------|-------|
| `rc` | Series RC low-pass | 1 | source voltage (V) | capacitor voltage (V) | R = 20 kOhm, C = 10 uF, tau = RC = 0.2 s | 5 V |
| `motor` | Armature-controlled DC motor, inductance neglected | 1 | armature voltage (V) | shaft speed (rad/s) | R = 1 Ohm, Kt = Ke = 0.1, b = 0.01, J = 0.005; gives K = 5 rad/s/V, tau = 0.25 s | 12 V |
| `msd` | Mass-spring-damper | 2 | force (N) | displacement (m) | m = 1 kg, c = 0.8 N s/m, k = 4 N/m; gives omega_n = 2 rad/s, zeta = 0.2 | 10 N |

Governing equations and state-space realizations (states in brackets):

- `rc`: `tau dv/dt = u - v`. H(s) = 1/(tau s + 1). A = [-1/tau], B = [1/tau], C = [1].
- `motor`: `J dw/dt = (Kt/R) u - (b + Kt Ke / R) w`. H(s) = K/(tau s + 1) with K = Kt/(R b + Kt Ke), tau = J R/(R b + Kt Ke). A = [-1/tau], B = [K/tau], C = [1].
- `msd`: `m x'' + c x' + k x = u`, states [x, v]. H(s) = 1/(m s^2 + c s + k). A = [[0, 1], [-k/m, -c/m]], B = [[0], [1/m]], C = [1, 0].

Identifiability (state this in the report and in the dashboard methodology page): from input-output data only the transfer function can be recovered. Therefore the identifiable quantities are: `rc`: tau (R and C separately are not identifiable); `motor`: K and tau (not Kt, Ke, b, J separately); `msd`: m, c, k (all three, because the DC gain 1/k and the numerator 1/m fix the scaling).

Dominant time scales used for horizons: `rc` 0.2 s, `motor` 0.25 s, `msd` 1/(zeta omega_n) = 2.5 s.

Per-plant simulation settings (in config): sequence duration `rc` 6 s, `motor` 10 s, `msd` 15 s; impulse-response horizon `rc` 1.5 s, `motor` 2.5 s, `msd` 12 s.

## 4. Data generation

### 4.1 Ground truth simulation
Exact ZOH discretization with `scipy.signal.cont2discrete(..., method='zoh')`, then iterate the difference equation. Unit test: agreement with `scipy.integrate.solve_ivp` (tight tolerances, piecewise-constant input) to relative error below 1e-6.

### 4.2 Excitation signals
All are generated in normalized units (peak magnitude at most 1), then multiplied by `u_max`. The first 0.5 s of every sequence has `u = 0`, so the rest state is observed.

- `random_steps`: levels drawn from U(-1, 1), hold times drawn from U(0.3, 2.0) s.
- `multisine`: 12 components with log-spaced frequencies in [0.1, 15] rad/s, random phases, normalized to unit peak.
- `chirp`: log sweep from 0.1 to 15 rad/s over the sequence, amplitude from U(0.3, 1), random initial phase.

### 4.3 Splits (each split uses its own seed; split by sequence, never by sample)
- `train`: 24 sequences (10 random_steps, 10 multisine, 4 chirp).
- `val`: 6 sequences (2 of each type). Used only for early stopping and hyperparameter choices.
- `test_iid`: 6 sequences (2 of each type). Not touched until Phase 5.
- Out-of-distribution test sets (Phase 5 only):
  - `ood_amplitude`: random_steps scaled to 3 x u_max.
  - `ood_frequency`: chirps sweeping 15 to 40 rad/s at unit amplitude.
  - `ood_waveform`: square and sawtooth waves, periods between 1 s and 8 s, unit amplitude.
- `impulse_experiment`: a single ZOH pulse of one sample (sample 0) with height `0.1 * u_max`, zero afterwards. The recorded output is divided by the pulse area `0.1 * u_max * Ts`, which gives the response to a ZOH pulse of height `1/Ts`. That scaled record is what Prony's method receives (section 5.2). Used only by Prony.

### 4.4 Noise
Additive white Gaussian noise on the output only. `sigma = std(y_clean over the train split) * 10^(-SNR_dB/20)`. The same sigma is used for every sequence of a plant and SNR level. Test sets are scored against the clean output. SNR levels: infinity, 40 (baseline), 30, 20, 10 dB.

### 4.5 Storage
`data/` is generated by `scripts/make_datasets.py`, is git-ignored, and each file is an `.npz` with the arrays plus a JSON sidecar (plant id, parameters, seed, Ts, SNR, signal types). Generation is fully deterministic given the seeds in `configs/experiment.yaml`.

## 5. Classical track (`src/sysid/classical/`)

### 5.1 ARX least squares (primary)
Model for known order n: `y[k] = -sum_{i=1..n} a_i y[k-i] + sum_{i=1..n} b_i u[k-i] + e[k]`. Stack the regressors `[-y[k-1..k-n], u[k-1..k-n]]` over all training sequences (pre-samples are zero because every sequence starts at rest) and solve `theta = argmin ||Y - Phi theta||_2` with `numpy.linalg.lstsq`.

Conversion to a continuous-time model:
1. Build a discrete state-space realization (controllable canonical form) from the ARX polynomials.
2. `A = logm(Ad)/Ts` using `scipy.linalg.logm`; `B = (Ad - I)^-1 A Bd` (valid because A is invertible for these plants). Reject and report the fit if `logm` returns a complex A.
3. Continuous transfer function by `scipy.signal.ss2tf`. Physical parameters from the identifiable combinations in section 3.

Unit tests: noise-free recovery of poles and identifiable parameters to relative error below 1e-6; `d2c` round trip against `cont2discrete`.

### 5.2 Prony's method (secondary)
Applied to the `impulse_experiment` record for samples `k >= 1` (the pulse occupies sample 0). Steps:
1. Linear prediction: `h[k] = -sum_{j=1..n} a_j h[k-j]`, solved by least squares over `k >= n+1`.
2. Discrete poles `z_i` are the roots of `z^n + a_1 z^(n-1) + ... + a_n`.
3. Residues `r_i` by least squares on the Vandermonde system `h[k] = sum r_i z_i^k`.
4. Continuous poles `s_i = ln(z_i)/Ts`. Because the data is the response to a one-sample ZOH pulse of height 1/Ts, the fitted residues are `r_i = R_i (1 - z_i^-1)/(s_i Ts)`; recover `R_i = r_i s_i Ts/(1 - z_i^-1)`.
5. `H(s) = sum R_i/(s - s_i)`; convert to numerator and denominator polynomials.

Unit test: noise-free poles recovered to relative error below 1e-6. Expected and to be reported honestly: Prony degrades quickly with noise.

### 5.3 Validation outputs
Impulse response and Bode plot (magnitude and phase) of each identified model against the true plant, using `python-control` as an independent reference for the plots and for cross-checking the NumPy LTI code in `src/sysid/lti.py`.

## 6. Shared LTI utilities (`src/sysid/lti.py`)
Pure NumPy and SciPy, no `python-control`, so the API can reuse them:
- `frequency_response(A, B, C, D, omega)`: `C (jwI - A)^-1 B + D`.
- `impulse_response(A, B, C, t)`: `C expm(A t) B`.
- `poles(A)`.
Tests compare each function against `python-control` to within 1e-9 relative error.

## 7. Learned track (`src/sysid/neural/`)

### 7.1 Architecture
- Latent state dimension `n_x = 4` for every plant (deliberately not equal to the plant order, so the order is not assumed). Hyperparameter in config.
- Vector field: `g(x, u)`: MLP with input `[x, u_n]`, two hidden layers of 64 units, tanh, output `n_x`. Then `f(x, u) = g(x, u) - g(0, 0)` (Decision D7).
- Output: linear readout `y_n = C x` (with bias).
- Initial state: `x[0] = 0`.

### 7.2 Integration
`torchdiffeq.odeint(func, x0, t_grid, method='midpoint', options={'step_size': Ts/2})`. `func(t, x)` returns `f(x, U[:, idx])` with `idx = floor(t/Ts + 1e-6)` clipped to the last sample. Required unit test: record every time at which `func` is called and assert that no evaluation lands on a multiple of Ts that belongs to the next interval (the test proves the ZOH lookup is never corrupted). Batch dimension is the sequence index.

### 7.3 Training (`train.py`)
- Loss: MSE between simulated and measured normalized output over the full horizon (free-run simulation, not one-step prediction).
- Optimizer: Adam, learning rate 1e-2 decaying to 1e-4 (cosine), weight decay 1e-5, 1500 epochs maximum, batch of 12 sequences. These are starting values: change them only after reading validation loss, log every trial in `artifacts/logs/`.
- Horizon curriculum: train on the first 20%, 40%, 70%, then 100% of each sequence at 0, 25, 50, 75% of the epochs. Valid because all sequences start at rest.
- Early stopping on full-horizon validation loss with patience 100 epochs. Save the best checkpoint.
- Seeds: `torch.manual_seed`, NumPy seed, recorded in the model card.
- Sanity gate: on noise-free data the validation fit should exceed 95%. If it does not, stop and report to the user before continuing.

### 7.4 Export and NumPy inference
`export.py` writes `neural_ode.npz` (weights, readout, `Ts`, `u_scale`, `y_scale`, `n_x`) and `neural_ode.json` (model card: seed, hyperparameters, best validation loss, epochs, training time, library versions, git commit, dataset hash, linearization matrices). `numpy_infer.py` replays exactly the same midpoint scheme (two sub-steps per sample, both using `u[k]`). Parity test: maximum absolute difference between PyTorch and NumPy outputs below 1e-4 in normalized units.

### 7.5 Linearization
At the origin: `A_n = df/dx`, `B_n = df/du` from the Jacobian, `C_n = C`. Physical units: `A = A_n`, `B = B_n / u_scale`, `C = y_scale * C_n`, `D = 0`. Stored in the model card. Poles are the eigenvalues of A.

### 7.6 LSTM baseline (optional, Phase 6)
Single-layer LSTM, hidden size 32, input `u_n[k]`, output `y_n[k]` via linear head, hidden state zero at k = 0, trained in free-run mode with the same splits and early stopping. It is not continuous-time, so its Bode and impulse responses come from the empirical protocol in section 8.4, not from a Jacobian.

## 8. Evaluation protocol (`src/sysid/analysis/`)

### 8.1 Metrics
- `fit_percent = 100 * (1 - ||y - y_hat||_2 / ||y - mean(y)||_2)` against the clean output; also RMSE in physical units.
- Impulse error: `||h - h_hat||_2 / ||h||_2` on `t in [0, horizon]` at the sample times. True and classical `h` are analytic. Neural ODE `h` comes from the linearization. Cross-check: the response of the nonlinear model to a small pulse (normalized amplitude 0.05) must match the linearized response; report the discrepancy.
- Bode error on a log grid of 200 points from 0.1 to 60 rad/s: mean absolute magnitude error in dB and mean absolute phase error in degrees (wrapped difference), reported separately in-band (0.1 to 15 rad/s) and out-of-band (15 to 60 rad/s). For `msd` also resonance peak frequency and height error. DC gain error for all plants.
- Pole error: relative distance to the true poles after nearest-neighbour matching.
- Cost: parameter count, training time, inference time per 1000 samples.

### 8.2 Experiments
- E1 Baseline: all plants, all methods, `test_iid`, trained on SNR = 40 dB.
- E2 Generalization: same models on the three OOD sets.
- E3 Noise robustness: re-identify at each SNR level in {inf, 40, 30, 20, 10}; evaluate on the clean `test_iid`. Mandatory for `msd` (Neural ODE with the number of seeds from gate G4); classical methods for all plants.
- E4 Parameter variation: Decision D9. 10 random `msd` plants.
- E5 Frequency and impulse fidelity: all plants, baseline models.
- E6 Cost comparison.

Hypotheses to test (not conclusions; never write results before they are measured): the classical LTI model extrapolates exactly in amplitude and frequency while the Neural ODE degrades outside the training range; the Neural ODE may be more tolerant of noise than ARX (equation-error bias) and Prony (pole sensitivity).

### 8.3 Result files
`scripts/run_experiments.py` writes `artifacts/results/<plant_id>/evaluation.json` with top-level keys `iid`, `ood`, `noise_sweep`, `param_variation`, `frequency_response`, `impulse_response`, `cost`, each holding numbers only. Figures are produced from these files by `scripts/make_figures.py` into `docs/figures/` and are never hand-edited.

### 8.4 Empirical frequency response (cross-check, and LSTM)
Drive the model with sampled sines `A sin(w k Ts)` at log-spaced frequencies; discard the transient (at least 8 dominant time constants); fit `a sin + b cos` by least squares over an integer number of periods. The result is the pulse transfer function `Gd(e^{jwTs})`. Compare it with the exact `Gd` of the true plant (ZOH discretization), not with the continuous `G(jw)`, so the hold effect cancels.

## 9. Artifacts committed to git (kept small)
```
artifacts/
  models/<plant_id>/classical.json       # ARX and Prony: coefficients, poles, TF, state space (A,B,C,D), identifiable parameters
  models/<plant_id>/neural_ode.npz
  models/<plant_id>/neural_ode.json      # model card
  results/<plant_id>/evaluation.json
  logs/                                  # hyperparameter trial logs
```
These files are the single source of truth for the API and for every number in the README.

## 10. API (`api/`, FastAPI, version prefix `/api/v1`)

Pydantic v2 request and response models. Business logic lives in `src/sysid`, routers only validate and delegate. Artifacts are loaded once at startup (lifespan handler). NumPy only, no PyTorch.

| Method and path | Purpose |
|-----------------|---------|
| `GET /health` | Status, API version, loaded plants and models. |
| `GET /systems` | List of plants: id, name, order, nominal parameters, units, `u_max`, training band. |
| `GET /systems/{id}` | One plant with true and identified parameters (ARX, Prony, Neural ODE poles). |
| `GET /models` | Model cards: training seed, hyperparameters, data hash, git commit, parameter counts. |
| `POST /simulate` | Simulate the plant and the selected models for a chosen input; returns trajectories and metrics. |
| `GET /frequency-response/{id}` | Bode data per model, query: `models`, `omega_min`, `omega_max`, `points` (max 500). |
| `GET /impulse-response/{id}` | Impulse response per model, query: `models`, `horizon_s`. |
| `GET /results/{id}` | The precomputed `evaluation.json` for the plant. |

`POST /simulate` request:
```json
{
  "system_id": "msd",
  "models": ["arx", "prony", "neural_ode"],
  "input": {"kind": "step", "amplitude": 5.0, "frequency_rad_s": null, "samples": null},
  "duration_s": 15.0,
  "noise_snr_db": 40.0,
  "seed": 1
}
```
`kind` is one of `step`, `sine`, `square`, `chirp`, `multisine`, `custom` (custom supplies `samples` in physical units at Ts). Response: `ts`, `t`, `u`, `y_true`, `y_measured` (when noise is requested), `predictions` (per model: `y`, `fit_percent`, `rmse`), `warnings`. Add a warning string whenever the amplitude exceeds the training range (`> u_max`) or the frequency exceeds the training band, with the ratio. Limits: at most 3000 samples per request, amplitude at most 3 x `u_max`; violations return HTTP 422.

Error body: `{"error": {"code": "<machine_code>", "message": "<human text>"}}`. CORS origins come from the `ALLOWED_ORIGINS` environment variable. Logging through the `logging` module, level from `LOG_LEVEL`. The OpenAPI docs at `/docs` must describe every field.

## 11. Dashboard (`dashboard/`, Streamlit)

The dashboard calls the API over HTTP (`requests` with timeouts) and never imports `sysid`. Base URL from `st.secrets["API_BASE_URL"]`, falling back to the `API_BASE_URL` environment variable for local development.

Pages:
1. Overview: the question the project asks, the three plants with equations and transfer functions, API status.
2. Simulator: choose plant, input kind and amplitude, noise level, models; overlay of true response and predictions, error trace, metrics table, and the warnings returned by the API when the input leaves the training range.
3. Frequency and impulse response: Bode magnitude and phase for all models with the training band shaded; impulse response overlay; table of poles.
4. Generalization and robustness: bar chart for OOD fit, line chart of fit versus SNR, box plot of the parameter-variation study, all from `/results/{id}`.
5. Methodology: identifiability table, estimator descriptions, model cards, known limitations.

Rules: no mock or hard-coded results anywhere; fixed colour per model across all pages (Okabe-Ito palette: truth black `#000000`, ARX `#0072B2`, Prony `#E69F00`, Neural ODE `#D55E00`, LSTM `#009E73`); axis labels with units; `st.cache_data` with a TTL for GET calls; a "waking the API" state with retries when the first request to `/health` is slow or fails (a free Render instance needs about a minute after sleeping); no emoji.

## 12. Repository layout
```
sysid-compare/
  AGENTS.md
  .agent/rules/rules.md          # copy of docs/Rules.md
  README.md
  pyproject.toml                 # ruff, mypy, pytest configuration; package metadata
  Makefile                       # make data | train | evaluate | figures | test | lint | api | dashboard
  render.yaml
  requirements-api.txt
  requirements-train.txt
  configs/systems.yaml
  configs/experiment.yaml
  docs/                          # Decisions.md, Project_Spec.md, Build_Steps.md, Deployment.md, Rules.md, figures/
  src/sysid/
    systems.py  signals.py  simulate.py  lti.py  metrics.py  config.py  io.py
    classical/  arx.py  prony.py  convert.py
    neural/     model.py  train.py  export.py  numpy_infer.py  lstm.py
    analysis/   linearize.py  freqresp.py  impulse.py  experiments.py
  api/          main.py  schemas.py  deps.py  routers/
  dashboard/    app.py  pages/  api_client.py  plotting.py  requirements.txt  .streamlit/config.toml
  scripts/      make_datasets.py  train_all.py  run_experiments.py  make_figures.py  smoke_test.py
  tests/
  artifacts/
  data/                          # git-ignored
```

## 13. Tech stack
Python 3.11. Research: NumPy, SciPy, `python-control` (validation only), PyTorch, `torchdiffeq`, Matplotlib, PyYAML, pytest, ruff, mypy. API: FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy. Dashboard: Streamlit, Plotly, requests. Pin versions in each requirements file; record the resolved versions of the training environment in the model cards. Check that the installed `torchdiffeq` and PyTorch versions are compatible before Phase 4.

## 14. Definition of done for the project
- All phases in `Build_Steps.md` complete and reviewed by the user.
- `make test` and `make lint` pass from a clean clone.
- Every number and figure in the README is regenerated from `artifacts/results/` by scripts.
- The API and the dashboard run locally against each other and are deployed per `Deployment.md`.
- The README states limitations honestly (section 15).

## 15. Limitations to state in the README
Simulated data only; linear time-invariant plants only; the Neural ODE uses a minimal zero-equilibrium prior; noise model is white output noise only; the Neural ODE results depend on the seed and are reported as mean and spread over seeds; the free-tier API sleeps when idle and a first request can take about a minute.
