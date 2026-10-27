# AGENTS.md

Project: classical vs. learned (Neural ODE) system identification of three simple plants, served by a FastAPI API (Render) and explored in a Streamlit dashboard (Streamlit Community Cloud).

Read in this order before doing anything: `docs/Decisions.md`, `docs/Project_Spec.md`, `docs/Rules.md`, `docs/Build_Steps.md`, `docs/Deployment.md`.

Non-negotiables (full list in `docs/Rules.md`):
1. Do not assume. If anything is unclear, stop and ask the user with options and a recommended default.
2. One phase at a time: plan, wait for approval, implement, verify, summarize, wait.
3. Never invent or hand-type a result. Every reported number comes from a script and a file in `artifacts/`.
4. Never tune on test or out-of-distribution data.
5. No dead code, no placeholders, no emoji, no filler wording. Format, lint, type-check, and test before reporting a phase done.
6. The API uses NumPy only. The dashboard talks only to the API.
7. Do only what the current phase lists.
