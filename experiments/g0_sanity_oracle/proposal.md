# G0 — Sanity Oracle

## Objective (done-when)
A linear, fully identifiable, low-noise synthetic system is recovered to **< 1e-3
relative parameter error by both the iterative estimator and NLS**. If either
method fails this, treat it as a code bug, not a scientific finding, and stop.

## System
`systems/linear_oracle.py` (`LinearOracle`): a linear 2-state ODE

```
dx/dt = -a x + y      (observed,  x0 = 1)
dy/dt = -b y          (hidden,    y0 ~ U[0.5, 1.5])
```

Closed form: `x(t) = (1 - y0/(a-b)) e^{-a t} + (y0/(a-b)) e^{-b t}`, a sum of two
decaying exponentials. Observing `x` on a time grid identifies `(a, b, y0)`
whenever `a != b`, so `(a, b)` is fully identifiable.

- `param_ranges`: `a ∈ [0.1, 0.35]`, `b ∈ [0.6, 0.9]` — well separated (gap ≥ 0.25),
  and deliberately **inside (−1, 1)**. Non-OGTT systems run with
  `use_normalization=False`, and `ParameterEstimator` ends in a `Tanh` that caps
  its raw output to `[−1, 1]`; keeping targets inside that interval avoids the cap.
  (This Tanh/normalization coupling is a latent bug for physical-space systems —
  logged separately in the bug report.)
- `t_points`: 8 points on `[0, 5]`. Deterministic ODE, `AUGMENTATION_FACTOR=0`,
  `USE_SDE=False` → effectively zero observation noise.

## Method
- **Iterative:** train the decoupled teacher-forced operator `T(θ;x) = P_ψ(x, H_φ(x,θ))`
  via `main.py` (supervised loss only), then run the fixed-point iteration at
  inference (`K = ITERATIONS = 10`). No spectral norm (G0 is an accuracy check).
- **NLS:** `scipy.least_squares` fit of the exact linear ODE model over `[a, b, y0]`
  (with `x0` taken from the observation), given the **same `θ^(0)` (`p_init`)** the
  network receives. Non-convergences are counted, never dropped.

## Config knobs (from `config.py`, pushed by `run_g0.py`)
`SYSTEM_NAME='linear_oracle'`, `NUM_SAMPLES=20000`, `EPOCHS=800`, `LEARNING_RATE=1e-3`,
`BATCH_SIZE=256`, `ITERATIONS=10`, `USE_SPECTRAL_NORM=False`, `USE_DERIVATIVE=False`,
`AUGMENTATION_FACTOR=0`; seeds `[0,1,2,3,4]`.

## Files
- `g0_settings.py` — shared constants (seeds, HPs, paths).
- `run_g0.py` — patches `config.py` defaults per seed, runs `python main.py`, restores.
- `evaluate_g0.py` — reloads each trained operator, reproduces the test split, runs
  the iterative + NLS estimators, computes relative error, writes `g0_metrics.json`
  and `g0_relative_error.pdf`.

## Pass criterion (as finalized)
Original done-when asked for `<1e-3` from **both** estimators. Empirically the
iterative operator cannot reach `<1e-3` on this (or any tuned) clean oracle — its
fixed point is biased ~5% (see below and `FINDINGS.md`). Per the user's decision
(2026-07-04), G0's **pass gate is NLS** (pipeline / identifiability validity); the
iterative estimator's error is **recorded as a documented method characterization**,
not gated. `param_ranges` were set to `a∈[0.10,0.35], b∈[0.60,0.90]`: well separated
so the two-exponential fit is well conditioned for NLS (a narrower `b∈[0.40,0.55]` was
tried but made NLS ill-conditioned — max rel err 2.1 in the tail — and was reverted).

## Results (5 seeds; `results/g0_metrics.json`, `results/g0_relative_error.pdf`)
- **NLS — PASS.** Mean relative parameter error **2.54e-7 ± 7.3e-9**, **0**
  non-convergences. The pipeline, identifiability, splits, and NLS harness are correct.
- **Iterative — documented.** At inference depth K=10: **5.4% ± 0.4%**; best over K
  (≈K=15–20): **4.2%**. K-curve descends from 9.8% (K=1) to ≈4.6% then rises again —
  it never approaches 1e-3.
- **Root cause (method finding):** teacher-forced/recurrent training makes the
  one-step map near θ* accurate (0.25%), but **θ* is not an attracting fixed point of
  T**. Without spectral norm the iteration is non-contractive and drifts; with it, the
  iteration contracts to a *biased* fixed point (~7%). Robust across a wide sweep
  (SN on/off, spectral margin 0.95/0.99, recurrent depth 2/5/10, nets 256³/512³, data
  20k/40k, epochs ≤1000). This is the bias phenomenon the paper studies — carry it to
  **G2** (operator instrumentation: it predicts `Lip_T<1` will hold only *with* SN, at
  the cost of a biased fixed point).

**G0 verdict: PASS** (pipeline gate). Iterative fixed-point bias documented, not a bug.
