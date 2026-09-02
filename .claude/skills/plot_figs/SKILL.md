---
name: plot-figs
description: Regenerate PDF figures from an experiment's saved metrics.json under experiments/<id>/. Use after run-experiment finishes or when a figure needs updating.
allowed-tools: Bash, Read, Write
---

## Instructions
$ARGUMENTS names a plan item ID (e.g., E1). Read experiments/<id_slug>/metrics.json (or
its configured output path from proposal.md) — never read outside that experiment's own
directory or from legacy results/ contents. Produce figures/<id>_<name>.pdf per the plot
spec below, using vector PDF, colorblind-safe colors, and font size >= 8pt:

- E1 (noise/sensitivity): sensitivity + RMSE vs. noise sigma, with the theoretical bound
  Lip_x/(1-Lip_T) overlaid.
- E2 (init robustness): final error vs. init distance, SN-on vs. SN-off.
- E3 (K-sweep): bias and sensitivity vs. K, with/without SN.
- E5 (non-identifiability): (s_i, sigma) plane with the fiber, true points, predictions,
  and drift arrows to the conditional mean.
Print a caption stub to stdout, and remind me to paste the real caption into the paper's
figure placeholder once reviewed.