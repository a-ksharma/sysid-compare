# Master Prompt for Antigravity

Paste everything below the line into a new Antigravity agent session, opened on the repository folder that contains `docs/`, `AGENTS.md`, and `.agent/rules/rules.md`. Use planning mode.

---

You are building the project described in `docs/`. It is a comparative study of classical (ARX least squares and Prony) and learned (Neural ODE) system identification on three simple plants, plus a FastAPI service and a Streamlit dashboard, deployed on Render and Streamlit Community Cloud.

Read these files completely, in this order: `docs/Decisions.md`, `docs/Project_Spec.md`, `docs/Rules.md`, `docs/Build_Steps.md`, `docs/Deployment.md`. Do not write any code yet.

Step 1, readiness review. List every ambiguity, contradiction, missing input, or technical risk you find in the documents, at most ten items. For each one give two or three options and your recommended default. Include the open gates in `Decisions.md` that apply to Phase 0. Ask me. Stop and wait. When I answer, record my answers under "Resolved at kickoff" in `docs/Decisions.md`.

Step 2, build. Execute `docs/Build_Steps.md` one phase at a time, starting at Phase 0. For each phase: produce an implementation plan and wait for my approval; implement only what the phase lists; run ruff, mypy, and pytest; then post a walkthrough that contains the real command output, the acceptance criteria each with its evidence, and anything that surprised you. Stop and wait for my go-ahead before the next phase.

Constraints:
- Follow `docs/Rules.md` exactly. Do not assume anything. If something is unclear, ask me before deciding.
- If a requirement seems wrong or infeasible, tell me and stop. Do not substitute a different design.
- Never invent, estimate, or hand-type a result. Every number comes from a script and a file in `artifacts/`.
- Do not tune on test or out-of-distribution data.
- Do not add features, dependencies, files, or abstractions the documents do not call for.
- The finished repository must read as careful engineering by a student who understands signals and systems: domain names from the spec, comments that explain why, plain wording, no filler, no emoji, no dead code, no placeholders.
- Items marked [VERIFY] in `docs/Deployment.md` must be checked against the vendor's current documentation during Phase 9 and the file updated.

Begin with Step 1 now.
