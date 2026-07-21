# Project: Iterative Network for Parameter Estimation in ODEs

## What this is
Amortized parameter estimation for nonlinear ODE systems (SIR, Lotka–Volterra, OGTT)
from sparse, partial observations. Core method: a fixed-point operator
T(theta; x_obs) = P_psi(x_obs, H_phi(x_obs, theta)), iterated to a fixed point.
Target venue: a SIAM journal. Paper draft lives in <paper/draft.pdf>.

## Initial Setup & Bug Handling (For Claude's First Run)
- **First Execution:** To understand the whole project, comprehensively analyze the `src/`, `systems/`, and `tools/` directories, along with `main.py` and `config.py`.
- **Bug Reporting:** Document the architecture and script functions. If you find any functional bugs or logic flaws that could distort experimental results, report and document them for reference.
- **Top Priority:** Fixing bugs that distort results is the absolute highest priority. Create a fix-checklist, implement the fix, and test it before moving on to any experiments.

## The scientific paradigm (read before designing any experiment)
The thesis is a BIAS–VARIANCE / ROBUSTNESS result, NOT a clean-accuracy result.
- Do NOT optimize for or celebrate lower in-distribution parameter RMSE. At the MSE
  optimum the iterative operator and a direct regressor estimate the SAME target
  E[theta | x_obs]; they are tied there by construction.
- The claim we support with code is: bounded input-sensitivity / robustness under noise,
  sparsity, and distribution shift, plus a controllable bias–variance trade-off via the
  spectral margin and the unroll depth K.
- When reporting results, always include sensitivity, across-noise variance, and tail
  (95th-pct) error — not RMSE alone.

## Hard constraints (do not violate)
- **Environment:** ALWAYS use `conda activate vision_task`.
- **Execution:** Single-model training is exclusively configured via `config.py`. Use `main.py` to run the model based on those settings. Before writing any experiment script, inspect `config.py` and `main.py` to confirm how per-run configuration actually works (e.g., whether `main.py` reads `config.py` directly, or accepts a path/overrides) — do not assume a CLI-args interface that may not exist.
- **Restricted Code (DO NOT TOUCH):** `prob_models/` directory and `master_train.py` are for follow-up papers and encode the CVAE / probabilistic-model work that is explicitly out of scope here. Do not modify, import from, or take design cues from them. This paper's estimator must remain fully deterministic.
- **Restricted Features (DO NOT USE):** Any SDE-related settings in `config.py` and `data_loader.py`, as well as `USE_DERIVATIVE` in `config.py`, are excluded from this paper.
- **Restricted Directories (DO NOT READ legacy contents):** `results/`, `docs/`, `analysis/`, and `tests/`. Do not read pre-existing/legacy contents from these.
  - **Exception:** `docs/experiment-plan.md` is the one file under `docs/` that Claude Code should read and update every session — it is the live experiment checklist, not legacy content. Nothing else under `docs/` should be read.
  - For new experiments, define a new output directory name in `config.py` (e.g., named after the experiment), and only write — never read — under `results/`.

## NEVER do (these are the audit red lines)
- Never fit scalers/normalizers on anything but the train split.
- Never feed theta_true or x_hid_true into the INFERENCE path (teacher forcing is train-only).
- Never drop failed/diverged NLS runs from metrics — count them as failures.
- Never split train/test by trajectory index; split by parameter and initial condition.

## New Experiment Workflow (The 5-Step Rule)
When executing a new experiment, strictly follow this procedure:
1. **Create Directory:** Make a dedicated directory for the experiment under `experiments/`.
2. **Write Proposal:** Write a `proposal.md` in this directory detailing the experiment's method and target goals.
3. **Write Script:** Create an executable script inside this directory that runs the experiment (driving `config.py` + `main.py` per the inspection above — do not bypass them).
4. **Tuning & Logging:** Keep in mind that hyperparameter tuning (params, model structure) might be needed. Write and maintain execution logs.
5. **Save & Update:** Upon completion, save figures in `.pdf` format and numerical metrics in `.json` format under the experiment's configured output directory. Update the experiment's own `proposal.md` with interpreted results, AND update the corresponding entry in `docs/experiment-plan.md` (status + a one-line finding).

## Repo layout
- `<src/models/>`   H_phi, P_psi, the operator T, spectral-norm wrappers
- `<src/data/>`     ODE simulators, sampling, normalization
- `<src/train/>`    decoupled teacher-forced training
- `<src/eval/>`     inference (fixed-point iteration), metrics
- `experiments/`    NEW: sweep drivers (keep experiment code here, not in src/)
- `results/`        JSON metrics, one subdir per experiment (Do not read legacy, write only)
- `figures/`        generated paper figures
- `docs/experiment-plan.md`  the live checklist (see @docs/experiment-plan.md) — the sole readable file under `docs/`

## Commands
- Env:   `conda activate vision_task`
- Train/Eval: Modify `config.py` and run `python main.py`
- Test:  `pytest -q`
- Lint/format: `ruff / black`

## Conventions
- Config-driven runs: every experiment configures `config.py` and writes that config next to
  its results. No hard-coded hyperparameters in scripts.
- Reproducibility: set and log all seeds; every metric is reported as mean ± std over ≥5 seeds.
- Metrics are saved as JSON keyed by (method, sweep_value, seed); parameter errors reported
  in PHYSICAL units (not normalized space).
- Make minimal changes; do not refactor src/ while building an experiment. New logic goes in
  the specific experiment directory. Show a plan and a diff before large edits.

## Definition of done for an experiment
A dedicated experiment folder exists with a `proposal.md`. Metrics JSON and PDF figures are written to the configured output directory. A plotting entry point regenerates the figure, `proposal.md` is updated with the final findings, and `docs/experiment-plan.md` reflects the updated status.