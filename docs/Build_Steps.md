# Build Steps

Work one phase at a time. For every phase:

1. Switch to planning mode and produce an implementation plan. Wait for the user's approval.
2. Implement only what the phase lists. No extra features.
3. Run `ruff check`, `ruff format --check`, `mypy src`, and `pytest`. Paste the real output in the walkthrough.
4. Stop. Summarize what was built, what was measured, and what surprised you. Wait for the user's go-ahead before the next phase.

If an acceptance criterion cannot be met, stop and report. Do not weaken the criterion, change the spec, or work around it silently.

Dependencies between phases: 0 -> 1 -> 2 -> (3, 4) -> 5 -> (6 optional) -> 7 -> 8 -> 9 -> 10.

---

## Phase 0: Repository and tooling
Gate G1 must be answered first.

Tasks: create the layout in `Project_Spec.md` section 12 (empty packages are fine, no placeholder code); `pyproject.toml` with ruff, mypy, pytest settings; `Makefile` targets; `.gitignore` (data, caches, virtual environments, `.streamlit/secrets.toml`); copy `Rules.md` to `.agent/rules/rules.md`; `configs/systems.yaml` and `configs/experiment.yaml` with the values from the spec; `src/sysid/config.py` that loads and validates them with dataclasses.

Acceptance: fresh virtual environment installs from `requirements-train.txt`; `make lint` and `make test` run (tests may be few); config loading test passes and rejects a malformed file.

## Phase 1: Plants and simulation
Tasks: `systems.py` (registry, parameters to transfer function and state space, identifiable-parameter maps), `signals.py` (random steps, multisine, chirp, square, sawtooth, sine, step, pulse), `simulate.py` (ZOH discretization, free simulation, noise at a given SNR), `lti.py` (frequency response, impulse response, poles).

Acceptance:
- ZOH simulation agrees with `solve_ivp` to relative error below 1e-6 for each plant.
- State-space and transfer-function forms agree for each plant (compare with `python-control`).
- `lti.py` functions agree with `python-control` to relative error below 1e-9.
- Noise generator achieves the requested SNR within 0.1 dB on a long record.
- Signal generators are deterministic given a seed and respect amplitude limits.

## Phase 2: Datasets
Tasks: `scripts/make_datasets.py` producing all splits in section 4.3 for all plants and all SNR levels, with JSON sidecars and a content hash.

Acceptance: running it twice produces identical hashes; no sequence appears in two splits; test and OOD files are not read by any training or tuning code (add a test that greps the training modules for their filenames).

## Phase 3: Classical identification
Tasks: `classical/arx.py`, `classical/convert.py`, `classical/prony.py`, `artifacts/models/<plant>/classical.json` writer, validation plots (impulse and Bode versus truth, using `python-control`).

Acceptance:
- Noise-free: ARX and Prony recover the true poles and identifiable parameters to relative error below 1e-6 for all plants.
- The `d2c` round trip reproduces the discrete model.
- At 40 dB: report the parameter errors and plots to the user (no numerical threshold, this is a measurement, not a target).
- Prony failure modes (complex or unstable poles) are caught and reported, not hidden.

## Phase 4: Neural ODE
Tasks: `neural/model.py` (vector field with zero-equilibrium, readout), `neural/train.py` (curriculum, early stopping, logging, seeds), `neural/export.py`, `neural/numpy_infer.py`, `analysis/linearize.py`.

Acceptance:
- The ZOH evaluation-time test from spec section 7.2 passes.
- PyTorch and NumPy outputs agree to within 1e-4 (normalized units) on the validation sequences.
- Noise-free training reaches more than 95% validation fit on `msd`. If not, stop and report; do not tune on the test set.
- Training is reproducible: the same seed gives the same final validation loss to six significant digits on CPU.
- Model card is written with every field listed in spec section 7.4.

## Phase 5: Analysis and experiments
Tasks: `analysis/` modules, `scripts/run_experiments.py` for E1 to E6, `scripts/make_figures.py`. Resolve gate G4 first.

Acceptance:
- Every metric in spec section 8.1 is computed by one function in `metrics.py` or `analysis/`, with unit tests on synthetic cases with a known answer.
- The linearization-based Bode response of the Neural ODE is compared with the empirical sine-test response (spec 8.4); report the largest discrepancy in dB and degrees. If it exceeds 1 dB or 5 degrees in-band, investigate and tell the user.
- `evaluation.json` is written for each plant and validates against a schema in `tests/`.
- The test split and OOD sets are used only here.
- Figures: Bode (training band shaded), impulse overlay, OOD bar chart, fit versus SNR, parameter-variation box plot. All with units and legends.

## Phase 6: LSTM baseline (optional, gate G2)
Tasks: `neural/lstm.py`, training under the same protocol, empirical frequency response (spec 8.4), added to E1, E2, E3 for `msd`.

Acceptance: same data splits; same metrics; frequency response from sine tests only; documented in the model card. If the user declines at the gate, skip this phase and remove LSTM mentions from the dashboard configuration.

## Phase 7: FastAPI service
Tasks: everything in spec section 10; `requirements-api.txt` containing no PyTorch; `scripts/smoke_test.py` that calls every endpoint.

Acceptance:
- Tests with FastAPI's `TestClient` for every endpoint, including 422 paths and the amplitude warning.
- `/frequency-response` for the Neural ODE equals the offline linearization result to 1e-9.
- A fresh virtual environment with only `requirements-api.txt` starts the server and passes the smoke test.
- Peak resident memory of the server after loading all artifacts is measured and reported (the free Render instance has 512 MB).
- Response time for a 3000-sample `/simulate` call is measured and reported.

## Phase 8: Streamlit dashboard
Tasks: everything in spec section 11.

Acceptance:
- Runs locally against the local API; no import of `sysid` or `torch` (add a test that scans the dashboard folder).
- The waking state appears when the API is stopped and the page recovers when it starts.
- Every chart shows data returned by the API; no hard-coded numbers.
- Colours, axis labels, and units are consistent across pages.

## Phase 9: Deployment and documentation
Gates G3 and G5 first. Follow `Deployment.md` exactly.

Tasks: `render.yaml`, deploy the API, deploy the dashboard, `scripts/smoke_test.py --base-url`, README with results regenerated from artifacts, figures in `docs/figures/`.

Acceptance: the public dashboard loads, wakes a sleeping API, and renders all five pages; the README contains the live links, the commands to reproduce, and the limitations from spec section 15.

## Phase 10: Final review
Tasks: clean-clone test (`git clone`, create environment, `make test`, `make lint`); read every file against `Rules.md`; remove anything unused; confirm that every README number matches `artifacts/results/`.

Acceptance: a checklist in the final walkthrough, one line per rule in `Rules.md` that applies, each marked pass with the evidence.

---

## Optional extensions (only after Phase 10 and only if the user asks)
- S1: noise and parameter sweeps for the Neural ODE on `rc` and `motor`.
- S2: output-error or instrumental-variable estimator to remove the ARX noise bias.
- S3: latent-dimension and width ablation for the Neural ODE (n_x from 1 to 6).
- S4: a short LaTeX or PDF report generated from the same artifacts.
