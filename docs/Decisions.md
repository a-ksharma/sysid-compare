# Decisions Log

This file is the single record of every choice that was not fixed by the original project brief. The agent reads it first and updates it whenever a decision changes or a gate is resolved.

Status values: **LOCKED** (settled, do not revisit without an explicit instruction from the user, recorded here) and **GATE** (needs an answer from the user before the named phase starts).

Origin: the user asked for 2 to 3 simple first- or second-order systems and delegated the choice of learned model and classical estimator. Those choices were made on their behalf with the reasoning below. The user may overrule any of them.

## Locked decisions

| ID | Decision | Reason | Alternatives rejected |
|----|----------|--------|-----------------------|
| D1 | Three plants: RC low-pass (1st order), DC motor speed (1st order, armature inductance neglected), mass-spring-damper (2nd order, underdamped, zeta = 0.2). | Two time-constant plants and one resonant plant. Resonance is the only place where peak frequency and phase fidelity can be tested. | Series RLC: its transfer function has the same form as the mass-spring-damper, so it adds a plant but no new information. Full 2nd-order DC motor: adds electrical dynamics that are 100x faster than the mechanical ones and make training stiff. |
| D2 | The mass-spring-damper is the primary study plant. Noise sweep and parameter-variation studies for the Neural ODE are mandatory only for it. The other two plants run the baseline and out-of-distribution experiments. | Keeps total training time realistic on a laptop CPU. | Full sweeps on all plants (about 15 extra Neural ODE trainings). Left as an optional extension. |
| D3 | Main learned model: Neural ODE (`torchdiffeq`). LSTM baseline is optional and decided at the Phase 6 gate. | The plant is continuous-time and so is a Neural ODE. This lets the Bode plot and impulse response be obtained exactly by linearizing the learned vector field, with no heuristics. An LSTM is a discrete-time sequence model and needs empirical sine tests for the same quantities. | LSTM only: weaker link to the course (continuous-time systems). |
| D4 | Classical estimators: (1) ARX least squares on input-output data, primary. (2) Prony's method on an impulse-experiment record, secondary. Both use the known physical order (1 or 2). | Both are named in the project brief. ARX is exact for noise-free sampled-data from a zero-order-hold input. Prony is known to be noise-sensitive, which makes it a legitimate comparison point and not a strawman. | Output-error or instrumental-variable refinement: listed as optional extension S2 in the build steps. |
| D5 | All inputs are piecewise constant (zero-order hold, ZOH) at sampling period Ts = 0.02 s. Ground truth uses exact ZOH discretization. | A DAC/ADC data-acquisition setup is ZOH by nature, and ARX is exact under ZOH. | First-order hold: introduces a non-causal feedthrough term in the discrete model and complicates ARX. |
| D6 | The Neural ODE is integrated with `torchdiffeq` using `method='midpoint'` and `step_size = Ts/2`. The input is looked up as `u[floor(t/Ts + 1e-6)]`. | The midpoint rule only evaluates the vector field at the start and middle of a step, never at the step end, so it never reads the next interval's input at an interval boundary. RK4 and the 3/8 rule do, which silently corrupts a ZOH input. | RK4 with ZOH lookup (boundary error); linear input interpolation (mismatches the ZOH ground truth and the ARX assumptions). |
| D7 | Zero-equilibrium parameterization: `f(x,u) = g(x,u) - g(0,0)`. Initial state is x0 = 0. Output is a linear readout y = C x. | Every plant rests at the origin. This is a minimal prior, not a physics model, and it makes the linearization at the origin well defined. It is stated openly in the report. | Numerical equilibrium search (more code, same result when training succeeds). |
| D8 | Neural ODE frequency response and impulse response come from the Jacobian linearization at the origin: A = df/dx, B = df/du (physical units after rescaling). An empirical sine-test frequency response is computed as an independent cross-check. | Exact for a continuous-time model and cheap enough to compute in the API. | Empirical-only: needs a ZOH correction and long simulations. |
| D9 | Parameter variation means re-identification across perturbed plants: for the mass-spring-damper, draw 10 plants with m, c, k each scaled by U(0.7, 1.3); identify each with every method; report the distribution of errors. | Tests whether each identification method works across the plant family. This is the only reading under which both methods are tested fairly. | Fixed model on a drifted plant: neither model can adapt, so it only measures model mismatch. **This is the one genuinely interpretive call. The user may switch to it.** |
| D10 | Baseline training noise: white Gaussian measurement noise on the output at SNR = 40 dB. Sweep: infinity, 40, 30, 20, 10 dB. Input is measured exactly. | Realistic and sweepable. The ARX equation-error bias under output noise is a known textbook effect and is reported, not hidden. | Noise on the input as well (errors-in-variables): out of scope. |
| D11 | The API runs inference with NumPy only. PyTorch is a training dependency and is not installed on Render. Weights are exported to `.npz`; a unit test proves the NumPy forward pass matches PyTorch. | Render free instances have 512 MB RAM and 0.1 CPU. PyTorch is too heavy for that. | Installing CPU PyTorch on Render: likely to exceed memory or build time on the free instance. |
| D12 | Frequency response and impulse response are computed live in the API from state-space matrices (NumPy and `scipy.linalg.expm`). Aggregate study results (generalization, noise, parameter variation) are precomputed JSON served by the API. | Live where it is cheap and interactive, precomputed where it needs hours of training. | Everything precomputed: the dashboard would have nothing live to show. |
| D13 | Dashboard: Streamlit with Plotly for interactive charts. Research figures: Matplotlib. The dashboard talks only to the FastAPI service and never imports the research package. | Keeps the Streamlit Community Cloud install small and keeps one source of truth. | Dashboard loading models directly. |
| D14 | Deployment: API on Render (free web service), dashboard on Streamlit Community Cloud. No keep-alive pinger. The dashboard shows a "waking the API" state and retries. | A free Render service sleeps after 15 minutes of inactivity and takes about a minute to wake. A pinger would consume the monthly free instance hours. | Paid always-on instance. |
| D15 | Sampling: Ts = 0.02 s. Excitation band: 0.1 to 15 rad/s. Plot and metric range: 0.1 to 60 rad/s. Nyquist is about 157 rad/s. | The three plants have corner or natural frequencies at 2 to 5 rad/s, so the band covers them with a decade of margin on each side. | - |

## Gates (answers needed from the user)

| ID | Question | Needed before | Default if the user says "you decide" |
|----|----------|---------------|----------------------------------------|
| G1 | GitHub repository name and visibility; Python version installed locally. | Phase 0 | Repo `sysid-compare`, public, Python 3.11 |
| G2 | Include the LSTM baseline? | Phase 6 | Yes, only if Phases 0-5 are finished and reviewed |
| G3 | Does the professor require a specific submission format (PDF report, slides, live demo)? | Phase 9 | The README is the write-up; figures exported to `docs/figures/` |
| G4 | Number of Neural ODE training seeds for the primary plant. | Phase 5 | 3 |
| G5 | Render region. | Phase 9 | The region nearest to the user that Render offers |

## Resolved at kickoff

Date: 2026-10-20

| # | Item | Resolution |
|---|------|------------|
| K1 | G1 -- Repo name, visibility, Python version | Repo name deferred (not created yet; project folder used as-is). Public. Python 3.11 (confirmed: full PyTorch support from PyTorch 2.11 onward). |
| K2 | Workspace root = repo root | Yes. Build the layout (`src/`, `api/`, `configs/`, etc.) directly in the current workspace. No nested subdirectory. |
| K3 | `io.py` naming conflict | Rename to `data_io.py` to avoid shadowing the built-in `io` module. |
| K4 | Test and validation dependencies | Keep a single `requirements-train.txt` for everything needed in the research/development environment (PyTorch, scipy, python-control, pytest, mypy, ruff, etc.). Separate `requirements-api.txt` and `dashboard/requirements.txt` remain as the spec requires for deployment. |
| K5 | Impulse experiment data file | Store the full record including sample k=0. Prony slices `k >= 1` at call time. |
| K6 | Neural ODE readout bias | No bias. The zero-equilibrium parameterization requires `y(0) = C * 0 = 0`. The "(with bias)" note in spec section 7.1 is overridden by this decision. |
| K7 | Makefile `api` and `dashboard` targets | `make api`: `PYTHONPATH=src uvicorn api.main:app --reload --port 8000`. `make dashboard`: `streamlit run dashboard/app.py`. |
| K8 | Motor parameter arithmetic | Verified: K = 5 rad/s/V, tau = 0.25 s. No action needed. |
| K9 | `experiment.yaml` seeds and structure | Agent chooses sensible defaults (data_seed: 42, train_seed: 0, etc.) and structures the YAML to cover every configurable value from the spec. |
| K10 | G4 preview -- Neural ODE training seeds | Accept 3 seeds as the default now. Phase 4 builds the multi-seed training loop accordingly. |
| K11 | G1 resolution & Git repo initialization | Initialize local git repository on branch `main` with project name `sysid-compare`, public visibility when pushed. |
| K12 | Windows tooling and execution | Maintain `Makefile` for Linux/CI/Render; use equivalent native virtualenv/PowerShell commands on Windows for local checks. |
| K13 | mypy type stubs for PyYAML | Add `types-PyYAML` to `requirements-train.txt` under development dependencies to satisfy strict mypy checks. |
| K14 | Pytest configuration warning | Set `addopts = ["--strict-markers"]` in `pyproject.toml` to eliminate the `strict_markers` warning. |
| K15 | ARX `logm` conversion tolerance | Strip imaginary residue (`A.real`) if max absolute imaginary part is below 1e-10; reject the fit if genuinely complex (> 1e-10). |
| K16 | Prony sample indexing | Index the Vandermonde system $h[k] = \sum r_i z_i^k$ using physical sample times $k = 1, \dots, N$ ($t = k T_s$). |
| K17 | Neural ODE readout matrix $C$ | Parameterize $C$ as a learnable linear layer without bias (`nn.Linear(n_x, 1, bias=False)`) trained end-to-end. |
| K18 | Neural ODE midpoint ZOH boundary | Index input via `idx = int(np.clip(np.floor(t / Ts + 1e-6), 0, len(u) - 1))` and verify with test suite callback logging. |
| K19 | Phase 0 baseline completion | Audit and complete Phase 0 files, install missing stubs, eliminate warnings, and verify clean test/lint passes. |
