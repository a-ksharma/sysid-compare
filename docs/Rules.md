# Rules

These rules apply to every task in this repository. Project facts are in `docs/Decisions.md` and `docs/Project_Spec.md`; the order of work is in `docs/Build_Steps.md`.

## 1. Working agreement
1. Do not assume. If a requirement is unclear, contradictory, or missing, stop and ask the user. State the question, two or three options, and your recommended default. Wait for the answer, then record it in `docs/Decisions.md`.
2. Work one phase at a time. Plan first, get approval, implement, verify, summarize, wait.
3. Do only what the phase lists. No extra features, endpoints, pages, or abstractions. Suggest them in the summary instead.
4. If the spec seems wrong or infeasible, say so and stop. Never substitute a different design silently.
5. Never edit `docs/Project_Spec.md` or `docs/Decisions.md` content except to log a resolved gate or a user-approved change.

## 2. Honesty about results
1. Every number in the README, the dashboard, and the summaries must come from a script that ran and from a file in `artifacts/`. Never type a result by hand, never estimate one.
2. Do not write conclusions before measurements exist. If the learned model loses, or the classical method fails, report it plainly.
3. No tuning on test or out-of-distribution data. Hyperparameters are chosen on `val` only; log every trial.
4. Report Neural ODE results as mean and spread over seeds where seeds were run.
5. If a check fails, show the failing output. Do not loosen a tolerance to make a test pass; explain and ask.

## 3. Code quality
1. Python 3.11, type hints on every function signature, `from __future__ import annotations` where needed.
2. Format and lint with `ruff format` and `ruff check`; type-check `src/` with `mypy`. All must pass before a phase is reported complete.
3. Docstrings (NumPy style) on public functions: one-line summary, parameters with units and shapes, returns, and the equation implemented when there is one.
4. Functions do one thing and stay under about 50 lines. Modules stay cohesive; follow the layout in the spec.
5. No magic numbers: constants come from `configs/*.yaml` or named module-level constants with a unit in the name or comment.
6. Use `pathlib`, `logging` (never `print` outside scripts' final summaries), dataclasses or Pydantic for structured data, and explicit exceptions with useful messages. No bare `except`, no silent fallbacks.
7. No global mutable state. Randomness only through an explicit `numpy.random.Generator` or a passed seed.
8. No dead code, no commented-out code, no `TODO`, no placeholder functions, no `pass` bodies, no unused imports or parameters.
9. Do not add dependencies that the spec does not list. Ask first.

## 4. It must read like careful human work
1. Names come from the domain and match the spec notation: `A, B, C, D`, `Ts`, `tau`, `zeta`, `omega_n`, `fit_percent`. No `data2`, `tmp`, `helper`, `utils.py`.
2. Comments explain why, not what. No banner comments, no section dividers, no narration of obvious code, no emoji anywhere (code, logs, docs, UI).
3. Documentation and UI text are plain and specific. Avoid filler and marketing words such as robust, seamless, powerful, comprehensive, cutting-edge, leverage, delve. Prefer a concrete statement with a number or a unit.
4. The README states the question, the method in a few paragraphs, the results (generated), how to reproduce, the live links, and the limitations. No badges for their own sake, no generated feature lists.
5. Commits: conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`), one logical change each, written in the imperative. Do not commit generated data or caches.
6. Figures: axis labels with units, legend, consistent per-model colours from the spec palette, titles that say what is plotted, saved at 200 dpi or as SVG by a script. No decorative styling.
7. Prefer the simplest correct implementation. No factories, plugin systems, dependency-injection frameworks, or base classes with a single subclass.

## 4a. Machine learning hygiene
1. Split by sequence. The test and OOD sets are read only by the Phase 5 experiment code; a test enforces this.
2. Fix seeds and record them with library versions, dataset hash, and git commit in each model card.
3. Normalize with statistics computed on `train` only; store them in the exported model.
4. Training uses free-run simulation loss, not one-step prediction.
5. PyTorch and NumPy inference must agree within the tolerance in the spec; keep the parity test in the suite.

## 5. Testing
1. Every module in `src/sysid`, every API endpoint, and the dashboard's API client has tests.
2. Numerical tests compare against an independent reference (`scipy.integrate.solve_ivp`, `python-control`, analytic solutions) with tolerances stated and justified in the test.
3. Tests are deterministic and need no network. API tests use `TestClient`.
4. A phase is not done until the acceptance criteria in `Build_Steps.md` are each demonstrated, not claimed.

## 6. API rules
1. Version prefix `/api/v1`. Pydantic models for every request and response. Business logic in `src/sysid`, never in routers.
2. Validate and bound inputs (sample count, amplitude, frequency range). Return the error schema from the spec.
3. Load artifacts once at startup. NumPy only; no PyTorch import anywhere under `api/`.
4. No secrets, no absolute paths, no runtime writes to disk.
5. Document every field in the OpenAPI schema.

## 7. Dashboard rules
1. Data comes only from the API. No mock data, no hard-coded results, no imports from `sysid` or `torch`.
2. Handle a sleeping API: retries with a visible "waking the API" state, timeouts on every request, readable error messages.
3. Cache GET calls with a TTL. Keep the app light for Community Cloud limits.
4. Show units on every axis and in every table. Show the API's out-of-range warnings.

## 8. Repository hygiene
1. Separate requirement files per deployment target as in the spec. Pin compatible versions.
2. Keep committed artifacts small. Never commit `data/`, virtual environments, caches, or secrets.
3. Notebooks, if any, are exploratory only and never the source of a reported number.
4. Keep `.agent/rules/rules.md` identical to `docs/Rules.md`.

## 9. Definition of done for any task
Code written, formatted, linted, type-checked, tested, acceptance criteria demonstrated with real output, summary posted, nothing beyond the phase scope added, and the user's approval received before the next phase starts.
