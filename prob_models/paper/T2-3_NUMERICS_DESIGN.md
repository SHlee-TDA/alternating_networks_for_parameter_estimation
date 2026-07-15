# T2-3 — Numerical illustration design (companion JMLR theory paper)

> Controlled, minimal experiments that show the theory of `jmlr/main.tex` *works*: the constants are
> real, `κ<1` holds, and the bounds are meaningful. Deliberately general (no OGTT specifics);
> E-c is the pure-math reduction of the application's `S_I·σ` fiber. Shared with **Track E B7/B3**
> (see §Sharing). Created 2026-07-14. Refs: `jmlr/main.tex`, `THEORY_PAPER_PLAN.md §6`.

## 0. Design principles
- **Closed-form-first.** E-a/E-b are linear-Gaussian, so the true posterior, `κ`, `ε_H`, `ε_P`,
  `ε_inc`, and every bound are available in closed form; the Monte-Carlo run only *confirms* them and
  demonstrates **single-run estimability** of `ε_inc`.
- **One claim per experiment.** E-a = exactness + rate; E-b = steering bound + `ε_inc` bound; E-c =
  collapse-free support recovery + compatibility certificate.
- **Reproducibility.** ≥5 seeds, mean±std; fixed chain length / burn-in; NumPy/SciPy only
  (`W_2` between Gaussians via the Bures formula). Figures as PDF, metrics as JSON.
- Each experiment names the exact theorem label it validates in `jmlr/main.tex`.

---

## 1. E-a — Linear-Gaussian, compatible  → validates `Thm:rate`, `Thm:B` (exactness)

**Setup.** A genuine bivariate Gaussian is the ground truth, and the learned conditionals are set
*equal* to the true ones (so `ε_H=ε_P=0`, the compatible case).
Let `(u,θ) ~ N(0, Σ)`, `Σ = [[σ_u², ρσ_uσ_θ],[ρσ_uσ_θ, σ_θ²]]`, `|ρ|<1`. Its conditionals:
- `p†_H(u|θ) = N(a_H θ, s_H²)`, `a_H = ρσ_u/σ_θ`, `s_H² = σ_u²(1−ρ²)`.
- `p†_P(θ|u) = N(a_P u, s_P²)`, `a_P = ρσ_θ/σ_u`, `s_P² = σ_θ²(1−ρ²)`.

Realize each learned conditional in the A-DCVAE form `Q_H = B_H#law(g_φ(θ,z_H)+σ_Hξ_H)` by
`g_φ(θ,z_H)=a_Hθ+c_H z_H` with `c_H²+σ_H²=s_H²` (latent + injection noise split; identical for `Q_P`).
Choose the box `Ω` wide enough (≳6·sd) that projection is inactive w.p. ≈1, so the Gaussian algebra holds.

**Closed-form quantities.**
- Condition-Lipschitz: `L_H=|a_H|`, `L_P=|a_P|`, so **`κ = L_H L_P = ρ²`** — matches the classical
  bivariate-Gaussian Gibbs autocorrelation, an independent sanity check.
- Deliberately pick `σ_u≠σ_θ` (e.g. `σ_u=2, σ_θ=1, ρ=0.8` → `a_H=1.6>1`, `a_P=0.4`, `κ=0.64`) so that
  **one individual `L` exceeds 1 while `κ<1`** — illustrates `Rem:noises` (only the product matters).
- The θ-chain is exactly the AR(1) `θ_{n+1} = ρ² θ_n + ζ`, `ζ~N(0, a_P²s_H²+s_P²)`; stationary
  `ν* = N(0, σ_θ²)` (equals `ν†`, i.e. `Thm:B` exactness).

**What we measure / plot.**
1. **Rate (`Thm:rate`).** Start `μ_0=δ_{θ_0}`, `θ_0` far. Law of `θ_n` is `N(ρ^{2n}θ_0, v_n)`;
   plot `log W_2(law(θ_n), ν*)` vs `n`, overlay the line of slope `log κ = 2log ρ`. Closed-form
   `W_2(N(m,v),N(0,σ_θ²)) = sqrt(m² + (√v−σ_θ)²)`; MC estimate overlaid.
2. **Exactness (`Thm:B`, `ε_H=ε_P=0`).** Confirm stationary samples match `N(0,σ_θ²)` (W_2→MC-floor).
3. **(optional) Dimension robustness.** Isotropic `d`-dim version: `κ=ρ²` independent of `d`, while the
   TV constant `ε` (=`δ_Hδ_P Leb(Ω)`) collapses with `d` — a one-plot contrast supporting `Rem:scope`.

**Expected outcome.** Geometric decay at rate exactly `κ=ρ²`; stationary law indistinguishable from
`N(0,σ_θ²)` up to MC error.

---

## 2. E-b — Controlled incompatibility  → validates `Thm:B` (bound), `Thm:incbound`

**Setup.** Take E-a's compatible model, then perturb **one** conditional. Keep `Q_H=p†_H` (so
`ε_H=0`) and set `Q_P^δ(θ|u)=N((a_P+δ)u, s_P²)` — a dial `δ` on the P-slope. For `δ≠0` the pair is
**incompatible** (linear-Gaussian conditionals are jointly Gaussian only if
`slope_H/slope_P = τ_u²/τ_θ²`, which the perturbation breaks), yet the chain is still the AR(1)
`θ_{n+1}=a_H(a_P+δ)θ_n+ζ_δ`.

**Closed-form quantities** (`Ω_H=[−U,U]`, `U≈6·sd_u`):
- `κ_δ = |a_H(a_P+δ)|`; `Var(ζ_δ)=(a_P+δ)²s_H²+s_P²`; `v*_δ=Var(ζ_δ)/(1−κ_δ²)`; `ν*_δ=N(0,v*_δ)`.
- **`ε_H=0`**, **`ε_P = sup_{u∈Ω_H} W_2(Q_P^δ,p†_P) = δU`** (linear in δ).
- **Actual** target error `W_2(ν*_δ, ν†) = |√v*_δ − σ_θ|`.
- **`Thm:B` bound** `= (L_P ε_H + ε_P)/(1−κ_δ) = δU/(1−κ_δ)`.
- **`ε_inc`** `= W_2(Π^→, Π^←)`, both centered bivariate Gaussians on `(θ,u)` sharing the same
  u-variance `V_u=a_H²v*_δ+s_H²`:
  - `Σ^→ = [[v*_δ, a_H v*_δ],[a_H v*_δ, V_u]]`,
  - `Σ^← = [[(a_P+δ)²V_u+s_P², (a_P+δ)V_u],[(a_P+δ)V_u, V_u]]`,
  - `ε_inc = ( tr(Σ^→+Σ^← − 2(Σ^{←1/2}Σ^→Σ^{←1/2})^{1/2}) )^{1/2}` (Bures; 2×2, closed form).
  At `δ=0`, `Σ^→=Σ^←` ⇒ `ε_inc=0`.
- **`Thm:incbound` bound** `= C·ε_P`, `C=max(C_H,C_P)` with `A=2+L_P†`, `L_P=|a_P+δ|`,
  `L_H†=|a_H|`, `L_P†=|a_P|`, `κ=κ_δ` (all plugged from above).

**What we measure / plot** (sweep `δ` over e.g. `[0, δ_max]`, `κ_δ<1` maintained):
1. **Panel A (`Thm:B`).** Actual `W_2(ν*_δ,ν†)` vs the `Thm:B` bound `δU/(1−κ_δ)`; the bound must lie
   above for all δ.
2. **Panel B (`Thm:incbound`).** Actual `ε_inc(δ)` vs the bound `C(δ)·ε_P(δ)`; must lie above, both
   →0 as δ→0, and `ε_inc≈(const)·δ` **tracking** `ε_P` — the "computable shadow."
3. **Single-run estimability.** Estimate `ε_inc` from one MC run (pair `(θ,u~Q_H)` vs the stationary
   `(θ,u)`; empirical `W_2` via POT/`scipy`), overlay on the closed-form curve.

**Expected outcome.** Both bounds hold with slack; `ε_inc` and `ε_P` vanish together and co-scale near
`δ=0`; MC `ε_inc` matches closed form (validates the single-run diagnostic claim).

---

## 3. E-c — Non-identifiable toy (`a·b=c`)  → validates `Prop:inc`, `Rem:edist`, non-collapse

**Setup.** Parameters `θ=(a,b)∈[a_lo,a_hi]²`; deterministic forward `u=a·b` (scalar hidden);
observation `x_obs` fixed with likelihood `N(x_obs; u, σ_obs²)`. True posterior
`p†(a,b|x_obs) ∝ exp(−(ab−x_obs)²/2σ_obs²)·prior(a,b)` — a **hyperbolic ridge** `{ab≈x_obs}` (only the
product is identified; the pure-math reduction of the OGTT `S_I·σ` fiber). Set learned conditionals =
true (`ε_H=ε_P=0`) so this experiment isolates **non-identifiability**, not approximation:
- `Q_H(u|a,b)=N(ab, σ_H²)`,
- `Q_P(a,b|u)=p†(a,b|u,x_obs)` (sample along `{ab≈u}` with prior; e.g. draw `a`, set `b=u/a`+noise, project).

**Reference posterior.** 2-D grid (e.g. 200×200) over the box, evaluate the unnormalized posterior,
normalize → `ν†(a,b)` (the "analytic reference," same method as **Track E B3**).

**What we measure / plot.**
1. **Support recovery.** Scatter of `ν*` samples over `ν†` contours; along-ridge coordinate histogram
   vs reference; across-ridge width vs the σ-floor `O(σ_H+σ_P)` (`Rem:edist`). Metric: 2-D / sliced
   `W_2(ν*, ν†)` and ridge coverage fraction.
2. **Certificate.** Monitor `ε_inc` across noise scales `σ∈{...}`; should stay ≈0 (compatible,
   `Prop:inc`) even though the target is a degenerate ridge.
3. **Collapse contrast (non-collapse).** Replace stochastic draws by conditional **means**
   (deterministic ping-pong) → converges to a single point on/near the ridge; overlay to show the
   deterministic estimator is *conceptually* uninformative (regression-to-mean) while the stochastic
   dual sweep covers the ridge. (Distribution-level convergence is `Thm:A`; the point here is the
   *shape* of what is recovered.)

**Expected outcome.** Stochastic sampler covers the ridge with width ≈ σ-floor ≪ ridge length;
`ε_inc≈0`; deterministic baseline collapses to a point. Report `a,b` in native units.

---

## 4. Shared config & reproducibility
- Seeds `{0..4}` (≥5), report mean±std. Chain: burn-in `B`, kept `N` samples, thinning as needed;
  fix per experiment and record in JSON.
- `W_2`: Gaussians via Bures (E-a/b); empirical via `POT`/`scipy` for MC checks and E-c (sliced-W_2 in 2-D).
- Outputs: `experiments/theory_Ea/…`, `…_Eb/…`, `…_Ec/…` with `metrics.json` + figure PDFs.
- Implementation is light (NumPy/SciPy); no training needed for E-a/b (analytic conditionals), minimal
  for E-c (analytic conditionals + grid). Full A-DCVAE nets are *not* required for the theory checks.

## 5. Sharing with Track E (avoid duplication)
| Companion (T2-3) | Track E | Shared artifact |
|---|---|---|
| E-b (`ε_inc` vs δ, target-noise dial) | B7(b) target-noise ablation + `ε_inc` monitor | `ε_inc` estimator, single-run W_2 |
| E-c (ridge recovery, stochastic vs deterministic) | B7(c) mode coverage + B3 grid reference | 2-D grid reference builder, collapse baseline |
| E-a | — (theory-only) | — |
Track E runs these on the real OGTT model; T2-3 runs the controlled toys. Estimator code is shared.

## 6. Theorem ↔ experiment map (for §8 of the paper)
- `Thm:rate` (rate `κ^n`) → **E-a**.
- `Thm:B` (steering bound; exactness at `ε=0`) → **E-a** (exactness), **E-b** (bound in the incompatible regime).
- `Thm:incbound` (`ε_inc≤C(ε_H+ε_P)`) → **E-b**.
- `Prop:inc` (`ε_inc=0` ⇔ compatible) → **E-b** (δ=0), **E-c** (`ε_inc≈0` on a ridge).
- `Rem:edist` (σ-floor), non-collapse → **E-c**.

### Log
- 2026-07-14 — 최초 설계. E-a/b closed-form(Bures W_2), E-c grid reference+collapse contrast.
  Track E B7/B3와 공유 매핑. 다음: §8 산문 반영 + 구현(Track E와 공유).
