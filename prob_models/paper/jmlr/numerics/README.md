# T2-3 — Numerical illustrations for the companion theory paper

Controlled, closed-form-first experiments validating `../main.tex`. Design rationale in
`../../T2-3_NUMERICS_DESIGN.md`. No training, no OGTT model, no checkpoints — pure NumPy/SciPy
(+matplotlib). Independent of Track E; the `sliced_w2` estimator matches Track E's
`prob_models/paper/experiments/_metrics.py` so the `eps_inc` diagnostic is the same object on both
the toys (here) and the real A-DCVAE model (Track E B7).

## Reproduce
```bash
conda activate vision_task            # numpy 1.26, scipy 1.15, matplotlib 3.8
cd prob_models/paper/jmlr/numerics
python e_a_rate.py                    # E-a: rate kappa=rho^2, exactness
python e_b_bounds.py                  # E-b: Thm B + Thm incbound bounds, eps_inc tracking
python e_c_ridge.py                   # E-c: ridge recovery vs collapse, eps_inc~0  (~90s)
```
Figures -> `../figures/num_E{a,b,c}_*.pdf`; metrics JSON -> `results/e_{a,b,c}.json`.

## What each validates (and observed results, default settings)

| Exp | Theorem | Key observed result |
|---|---|---|
| **E-a** | `Thm:rate`, `Thm:B` (exactness) | `kappa = rho^2 = 0.640` (= classical bivariate-Gaussian Gibbs rate); closed-form `W_2` decays at slope `= log kappa` exactly; `L_H = 1.60 > 1` yet `kappa < 1`; stationary `W_2(nu*, nu_dagger) = 1e-10` (exact). |
| **E-b** | `Thm:B`, `Thm:incbound` | Both bounds hold for all `delta` in `[0, 0.18]`; `eps_inc(delta=0) ~ 0` (compatible); `eps_inc/eps_P in [0.035, 0.039]` (bounded => tracking); single-run empirical-cov Bures estimate matches the closed-form `eps_inc` (0.109 vs 0.105 at `delta=0.18`); sliced-`W_2` is a lower bound above its `delta=0` resolution floor. |
| **E-c** | `Prop:inc`, non-collapse | Stochastic sweep covers the ridge `ab=1`: HPD95 coverage `0.95`, sliced-`W_2(nu*, ref) = 0.046`; along-ridge std **1.19 (stochastic) vs 0.000 (deterministic ping-pong)**; `eps_inc = 0.008 ~ 0` certifies compatibility. |

## Notes
- E-a/E-b are fully closed form (Gaussian `W_2` via the Bures metric, `_w2.py`); the Monte-Carlo run
  only confirms the closed forms and demonstrates single-run estimability of `eps_inc`.
- The finite-sample floor of the general nonparametric `sliced_w2` estimator is reported explicitly
  (E-a plateau; E-b `delta=0` floor line) — the honest resolution limit of the in-run diagnostic.
- Seeds default to 5; pass `--seeds`, `--n_chains`, etc. to rescale.
