# Deterministic Paper (SIMODS/`paper/simods/`) — Session Handoff Notes

> Written 2026-07-21 at the end of a long working session. This note records how the
> deterministic paper's framing evolved during the session, what was changed and confirmed,
> and what remains to do (Ha Joon reference + how this attaches to Paper 2). Read this first
> before continuing work on `paper/simods/`.

---

## 0. One-line status

The paper is a **finished, honest, arXiv-ready draft** — it compiles cleanly (tectonic, 26 pages,
no errors / no overfull hboxes / no undefined refs) and its thesis, theory, experiments, figures,
and narrative are consistent. Remaining work is a real citation (Ha Joon's OGTT model), template
placeholders (title/funding/MSC), and the strategic decision of how to connect it to Paper 2.

---

## 1. How the framing evolved this session (the story)

The draft began as a **"we propose an iterative method"** paper (positive / superiority tone), with a
§Correctness that *claimed* teacher forcing aligns the fixed point with the ground truth. Through
discussion we established this was a **bait-and-switch**: the experiments show it does NOT align.

The pivot, confirmed by both analysis and the user's judgment, is to an **honest CHARACTERIZATION /
limitation paper**. The load-bearing insight (the "conditional-mean ceiling"):

> The fixed point `θ*(x_obs)` is a deterministic **function of `x_obs`**. Therefore it cannot beat the
> conditional mean `E[θ|x_obs]` — reconstructing the hidden state gives **no information advantage at
> inference**, because the reconstructed state is itself a function of `x_obs`. A plain direct
> regressor reaches this ceiling *more closely* (the decoupling's reconstruction detour only inflates
> the error). Under non-identifiability the conditional mean is moreover **off the physical fiber**
> (Jensen / extreme-point), so the estimate is not even physically admissible.

This resolves the user's key worry ("if reconstructing the hidden state should help, why doesn't it?"):
it doesn't, because the true hidden state is unavailable at inference and its surrogate is an image of
`x_obs`. The limitation is **structural (a property of the problem), not of our method** — which is
exactly what motivates the probabilistic successor (Paper 2).

Empirically this was confirmed hard: the **direct regressor beats the iterative estimator everywhere**
(SIR wide β: r=0.98 vs 0.66), and an **observation-density sweep** shows accuracy improves with density
but **plateaus below the ceiling** — correctness is approached only outside the sparse premise.

---

## 2. Confirmed decisions

- **Thesis:** honest characterization (convergence + conditional-mean ceiling + off-fiber collapse +
  observability-predicts-identifiability). **No performance/superiority claim. No "correctness" promise.**
- **Venue:** SIAM is off (needs a positive result). **Realistic: TMLR** (accepts negative/characterization
  results). **Plan: finish the arXiv version now, decide merge-vs-separate with Paper 2 afterward.**
- **SIR experiment:** wide range `[0.01,0.5]` (the code's adversarial design) is primary in the table;
  narrow `[0.08,0.12]` mentioned as a note. **Random (unknown-hidden-state) ICs** are the valid,
  OGTT-consistent setting — do NOT fix ICs just for a prettier figure.
- **Direct regressor** = clean control (isolates the structural collapse without the iterative operator's
  own approximation gap). The **iterative estimator remains the studied object / protagonist.**
- **Related work + EM/NLLS empirical baselines: deferred.** But a §Discussion paragraph now makes the
  positioning point in prose (see §3).
- **Figure 3 (LV) and Figure 4 (OGTT ill-conditioning):** regenerated at vector-PDF quality (were
  low-res PNGs imported from a note).

---

## 3. Concrete changes made this session

**Theory (`sections/correctness.tex`)** — reframed §Correctness from a "correctness" claim to a
characterization:
- Added **Prop:ceiling** (x_obs-measurability ⇒ bounded by conditional mean; hidden-state reconstruction
  adds no info). This is the paper's new theoretical heart.
- Generalized off-fiber result: **Prop:off_fiber** (any strictly-convex-position fiber ⇒ conditional
  mean off-fiber, via Bauer's extreme-point characterization) + **Cor:off_fiber_product** (the OGTT
  product `S_I·σ` case, with the explicit Jensen bound `E[S_I|c]E[σ|c] ≥ c`).
- **Rem:decomposition**: error `ε` = structural part `E[Var(θ|x)]` + operator-approximation part.
- Reframed the teacher-forcing paragraph honestly ("targets ε=0, generally unreachable").

**Intro / abstract (`sections/intro.tex`, `main.tex`)** — rewritten to the honest thesis (ceiling +
off-fiber + "motivates distributions"); **fixed the false disparagement of direct regression** (it was
called "overfitting/memorizing" in both intro and method, contradicting the finding that it is the
strong baseline).

**Experiments (`sections/09_experiments.tex`)** — rewritten with real numbers and honest roles:
Table 1 (iterative vs baseline RMSE/r), SIR = identifiability control (operator gap), LV = observability
predicts β is unrecoverable, OGTT = off-fiber collapse (product stiff, individuals both sloppy), plus the
**density-sweep figure** and the SN-margin ablation. Distribution-shift note kept as a non-thesis remark.

**Identifiability (`sections/04_feasibility.tex`)** — activated as §6; Hermann–Krener observability is the
a-priori organizing principle predicting the SIR/LV/OGTT spectrum. Fixed the σ-notation collision:
singular values are now written **`ς` (`\varsigma`)** to avoid clashing with the parameter σ.

**Discussion (`sections/11_discussion.tex`) — NEW §8**: positions the limitation as not-method-specific;
distinguishes failure modes (squared-loss → off-fiber mean; **EM / NLLS → non-unique fiber point**); and
motivates lifting inference to distributions (forward-reference to Paper 2 as "subsequent work").

**Figures** — OGTT joint-collapse regenerated with the corrected panel 3 (both `S_I` and `σ` shown, both
sloppy — the old "S_I is THE sloppy direction" was a hard-coded, unjustified label); new density-sweep
figure; **Fig 3 (`figures/lv_nonident.pdf`)** and **Fig 4 (`figures/ogtt_illcond.pdf`)** regenerated as
clean vector PDFs (generation scripts: `experiments/det_meanlocus/{run_paper_exp.py, ogtt_illcond_fig.py}`
and an inline LV script).

**Bugs fixed (code + paper):**
- `src/analyzer.py`: Lip_T instrumentation ignored the `ExcludeLambda` scale (default config was actually
  `Lip_T≈1.004>1`, i.e., *not* contractive) — now measures the true effective Lipschitz factor; also
  panel-3 of the collapse plot now shows both coordinates.
- `src/data_loader.py`: normalization was gated to OGTT only → SIR/LV hit the output-`Tanh` cap; now
  enabled for all systems.
- `sections/05_method.tex`: operator definition typo `H_ψ → H_φ`; empty `\cite{}` removed; overstated UAT
  claim softened; 231 lines of commented-out clutter removed; map equations compacted (also fixed
  flattened-dimension notation `ℝ^{(N+1)×d} → ℝ^{(N+1)d}`).
- Fixed `eq:(6.7)` abnormal size (a stray `\resizebox` I had wrongly added) and all visible overfull
  hboxes; removed all TODO/placeholder text that had been rendering in the PDF.

**Citations** — SIR `\cite{kermack1927contribution}`, LV `\cite{lotka1925elements,volterra1927fluctuations}`
added (all real, already in `references.bib`). OGTT is a clearly-marked **placeholder [6]** awaiting the
correct reference (see §4).

**Build** — the SIAM class (`siamonline250211.cls`) is pdfLaTeX/dvips-oriented and breaks under XeTeX/
tectonic (loads dvips-only `breakurl`/`hypdvips`). Fixes applied so tectonic compiles: the `nohypdvips`
class option in `\documentclass`, and a local patch to the class forcing its pdf branch (harmless under
pdfLaTeX / Overleaf, which already take that branch). tectonic lives in conda env `texbuild`
(`/home/shlee/miniconda3/envs/texbuild/bin/tectonic`).

---

## 4. TODO / next steps

1. **Ha Joon OGTT reference (blocking a clean bibliography).** Replace the placeholder entry
   `ha2025disposition` in `references.bib` with the correct BibTeX for collaborator Ha Joon's OGTT
   glucose–insulin minimal model — the paper that (a) defines the 6-state model `[G,I,N5,N6,S_I,σ]` and
   (b) establishes the dimensional-analysis result that glucose identifies only `mDI = S_I·σ`. If these
   are two papers, cite both. The key is cited in `sections/04_feasibility.tex` (model definition) and
   `sections/09_experiments.tex` (the `mDI` claim). Rendered as `[6]` and clearly marked "PLACEHOLDER".

2. **Template placeholders.** Title still reads as a "method" paper — consider a characterization-oriented
   title (deferred by the user). Funding line ("Fog Research Institute FRI-454") and MSC codes are the
   SIAM template's placeholders — fill with real values.

3. **How to attach to Paper 2 (strategic decision, deferred until after arXiv).** Two options discussed:
   - **(a) Merge** the characterization into Paper 2 as its motivation/analysis section → one strong TMLR
     paper. Safest and highest-quality, but loses the standalone deterministic draft. Paper 2 then does
     not need to cite an external deterministic paper.
   - **(b) Two companion TMLR papers** cross-citing: deterministic (characterization) on arXiv first,
     Paper 2 (probabilistic method) citing it. Preserves the draft and gives a second paper, at the cost
     of some salami-slicing optics and the weaker paper's rejection risk.
   The user leans toward keeping two papers (paper count matters) but agreed one combined mega-paper
   would be an unreadable "thick novel." **§8 Discussion already forward-references Paper 2 as
   "subsequent work"**, so either path is supported. Decide after the arXiv post.

4. **Optional strengthening (deferred).** Empirically run EM / NLLS / other iterative-latent baselines to
   show "all deterministic methods fail" and add a Related Work section. Not necessary (Prop:ceiling
   already carries the generality, and §8 makes the point in prose), but would harden the paper.

5. **Reproducibility.** Experiments are driven by `experiments/det_meanlocus/run_paper_exp.py`
   (env knobs: `SIR_RANGE`=wide|narrow, `SIR_FIXED_IC`=0|1, `SIR_NPOINTS`=<n>; CLI: `--system`,
   `--baseline`, `--spectral_scale`, `--ood`). Prediction extraction: `extract_preds.py`. Figure
   generation: `ogtt_illcond_fig.py`, plus the figures under `results/det_paper/.../` copied into
   `paper/simods/figures/exp/`. Compile: the `texbuild` conda env's tectonic.
