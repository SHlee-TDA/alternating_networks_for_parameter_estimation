---
name: audit
description: Run gates G0-G2 in order via run-experiment and report pass/fail, updating docs/experiment-plan.md. Use before starting any E-numbered experiment.
allowed-tools: Bash, Read
---

## Instructions
For G0, then G1, then G2, in order:
1. Check docs/experiment-plan.md — if the gate is already [x], skip it and move to the
   next.
2. Otherwise, run it via the run-experiment skill logic for that ID.
3. Stop immediately and report FAIL if a gate does not meet its done-when criterion; do
   not proceed to the next gate or to any E-numbered experiment.
Print a final PASS/FAIL table for G0-G2. Only report overall PASS if all three are [x] in
docs/experiment-plan.md.