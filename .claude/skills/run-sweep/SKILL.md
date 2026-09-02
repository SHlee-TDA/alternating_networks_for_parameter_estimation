---
name: run-sweep
description: Run a reproducible experiment sweep (noise, K, or init-distance) over >=5 seeds and save metrics JSON. Use when starting or re-running any parameter-estimation experiment.
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit
---

## Repo state
!`git status --short && echo "---" && ls results/ 2>/dev/null`

## Instructions
Run the sweep named by $ARGUMENTS (one of: noise, k, init). Steps:
1. Read CLAUDE.md and confirm the audit red lines are not violated by the run config.
2. Load `experiments/config/<sweep>.yaml`; if missing, scaffold a minimal one and show it to me.
3. For each sweep value and each method (iterative_sn, direct, nls), loop over >=5 seeds.
4. Record RMSE (physical units), input sensitivity ||d theta_hat / d x_obs||_2, across-seed/noise
   variance, and 95th-pct error. For iterative_sn also record measured Lip_T and the bound
   Lip_x/(1-Lip_T). Count NLS non-convergences; never drop them.
5. Write results/<sweep>/metrics.json keyed by (method, value, seed); copy the config beside it.
6. Print a compact summary table and append a findings note to docs/experiment-plan.md.
Do not modify src/ beyond what the sweep requires.