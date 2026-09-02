---
name: run-experiment
description: Run one gate or experiment (G0-G2, E1-E6) from docs/experiment-plan.md end to end, following the 5-Step Rule and driving config.py + main.py. Use when starting or resuming a specific plan item.
disable-model-invocation: true
allowed-tools: Bash, Read, Write, Edit
---

## Repo state
!`git status --short && echo "---" && cat docs/experiment-plan.md 2>/dev/null | head -40`

## Instructions
$ARGUMENTS is a plan item ID (e.g., G0, G1, G2, E1 ... E6). Steps:

1. Read CLAUDE.md and the full entry for $ARGUMENTS in docs/experiment-plan.md (its
   objective/done-when, config knobs, and constraints). This is the only file under docs/
   you may read — do not read anything else there, and do not read results/, analysis/,
   or tests/.
2. If experiments/<id_slug>/ does not exist, create it (Step 1 of the 5-Step Rule).
3. If experiments/<id_slug>/proposal.md does not exist, write it: state the method, the
   target config knobs, and the pass/done criterion copied from experiment-plan.md
   (Step 2).
4. Before writing any script, inspect config.py and main.py to confirm how a run is
   actually configured and launched (do not assume a CLI-args interface). Confirm this
   run will not touch prob_models/, master_train.py, USE_DERIVATIVE, or any SDE-related
   setting.
5. Write an executable script in experiments/<id_slug>/ that edits config.py (or writes
   a per-run config the way main.py expects) and invokes main.py for each condition in
   the sweep and each of >=5 seeds (Step 3). Log every run.
6. Follow the audit red lines: scalers fit on train split only; inference path never
   receives theta_true or x_hid_true; NLS non-convergences are counted, never dropped;
   splits are by parameter/initial-condition, not trajectory index.
7. Save metrics as JSON (physical units, keyed by method/sweep-value/seed) and figures as
   PDF under the experiment's own configured output directory (never write into legacy
   results/ paths) (Step 4-5).
8. Update experiments/<id_slug>/proposal.md with the interpreted findings, AND update the
   matching entry in docs/experiment-plan.md: tick its status box, tick the 5-step
   checklist, and append one dated bullet to the Cumulative findings log (Step 5).

Keep changes minimal and confined to experiments/<id_slug>/; do not modify src/. Show a
plan and the file diff before executing anything that trains or runs main.py.