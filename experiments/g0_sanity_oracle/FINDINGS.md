# G0 diagnostic findings

Status: **G0 PASS on the pipeline gate (NLS)**, per the user's decision (2026-07-04):
gate on NLS validity, record the iterative estimator's error as a documented method
characterization. NLS passes decisively (2.54e-7); the iterative estimator floors ~2
orders of magnitude above 1e-3 (structural, not a bug).

Note on param ranges (NLS conditioning): the two-exponential fit is only well
conditioned when the rates are well separated. `b∈[0.40,0.55]` (chosen mid-way to keep
the neural Tanh unsaturated) made NLS ill-conditioned — 200-sample probe gave mean
2.5e-2 with a 2.1 tail. Final ranges `a∈[0.10,0.35], b∈[0.60,0.90]` give NLS ~1e-13 on
the same probe and 2.54e-7 in the full 5-seed run. NLS is the gate, so ranges are tuned
for NLS conditioning; the mild Tanh compression near b=0.9 only affects the (documented,
ungated) neural estimator. The sweep tables below predate the revert (ranges 0.40–0.55)
but the qualitative conclusion — iterative floors at a few % regardless of tuning — is
unchanged; the final 5-seed run gives iterative 5.4% @K=10 / 4.2% best-K.

## Numbers (linear_oracle, seed 0, 20k samples, 800 epochs unless noted)
| method / config | rel. param error | notes |
|---|---|---|
| **NLS** (exact ODE fit) | **2.8e-7** (0 non-conv.) | pipeline + identifiability are correct |
| Iterative, SN-off, supervised | 0.087 (K=10), diverges for K>10 | non-contractive |
| Iterative, SN-on, supervised | 0.086 (stable K≥5) | contractive but biased fixed point |
| Iterative, SN-off, +recurrent | 0.045 (best K=20) | still diverges past best K |
| Iterative, SN-on, +recurrent | 0.063 (stable) | best stable config, still 6% |

## Root-cause diagnostic — θ* is not an attracting fixed point of T
Iterating the operator **starting exactly from the true parameters θ***:

| config | k=1 | k=5 | k=20 | k=50 |
|---|---|---|---|---|
| SN-off (+rec) | **0.0025** | 0.008 | 0.026 | 0.045 |
| SN-on  (+rec) | 0.020 | 0.049 | 0.062 | 0.063 |

- One step from θ* is very accurate (0.25%, SN-off) — teacher-forced training does
  learn `T(θ*) ≈ θ*` locally.
- But the iteration **drifts away** from θ* to a spurious attractor (~4.5–6.3%).
  θ* is repelling (SN-off) or a biased fixed point exists elsewhere (SN-on).
- Consequence: unrolling K>1 *degrades* a good estimate. There is no K at which the
  iteration from the constant `p_init` lands within 1e-3 of θ*.

Interpretation: the method's central premise (a contraction whose fixed point is the
target, `Lip_T < 1` — the thing G2 is meant to certify) is **not realized** by the
current teacher-forced training. SN buys contraction at the cost of a biased fixed
point; no-SN keeps local accuracy but is not contractive.

## Tuning sweep outcome (per user request "tune harder first")
Wired two previously-hardcoded knobs to be config-driven (backward-compatible):
`MODEL_CONFIG[*]['spectral_scale']` (src/models.py) and `RECURRENT_ITER`
(src/losses.py `RecurrentLoss`, config.py field). Swept:

| trial | SN | sscale | rec | net | data/ep | best rel err | best K |
|---|---|---|---|---|---|---|---|
| sw_snoff_r10  | off | 0.99 | 10 | 256^3 | 20k/600  | 0.0342 | 3 |
| sw_snon99_r10 | on  | 0.99 | 10 | 256^3 | 20k/600  | 0.0729 | 20 |
| sw_snoff_big  | off | 0.99 | 5  | 512^3 | 40k/1000 | **0.0305** | 15 |
| sw_snon_big   | on  | 0.99 | 5  | 512^3 | 40k/1000 | 0.0734 | 2 |

Also earlier: SN-on/off × supervised/recurrent(2) at 20k/400-800 → 0.045–0.087.
Across SN on/off, spectral margin (0.95/0.99), recurrent depth (2/5/10), capacity
(256^3/512^3), data (20k/40k), epochs (400–1000), and inference K (1–30):
**the iterative estimator never drops below ~3%.** NLS stays at 2.8e-7 throughout.

Conclusion: reaching <1e-3 with the iterative estimator is not a tuning problem. The
limit is the operator's fixed-point structure, not fit quality (teacher-forced MSE
reaches 1e-4). This is a substantive finding about the method, not a pipeline bug.

## Separately confirmed code issue (bug report)
`src/models.py ParameterEstimator` ends in `nn.Tanh()`, hard-capping raw outputs to
[-1, 1]. For every non-OGTT system the pipeline runs with `use_normalization=False`
(see `setup_dataloaders`), so parameters live in **physical** space:
- targets outside [-1, 1] are **unreachable** (silently);
- even inside, resolution is crushed near ±1 (tanh saturation) — measurably hurts
  precision for parameters near the range edge.
This is why G0's param ranges were placed well inside (-1, 1). It should be fixed
(decouple the output squashing from `use_normalization`) before trusting any
physical-space (SIR / Lotka–Volterra) result.
