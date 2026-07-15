"""
B7 — Three-guarantee isolation ablation (empirical validation of the theory).

Maps the three noise sources (DISCUSSION.md §E taxonomy) to the three guarantees
and ablates each in isolation:

  N1  condition noise  (train, CONDITION_NOISE_STD_*)  -> Thm 1 / sweep-contraction
        Jacobian/Tikhonov  ->  STABILITY (kappa < 1, no divergence)
  N2  target noise     (train, TARGET_NOISE_STD_*)     -> Thm 2 denoising score
        matching  ->  DIRECTION / correctness (unbiased mDI fiber)
  N4  injection noise  (inference, INFER_NOISE_*)      -> Thm A minorization + pi*
        width  ->  MODE COVERAGE (stochastic vs deterministic ping-pong; Thm 3)

Experimental design: the data split, normalizer and analytic reference posteriors
are held FIXED (seed 42). Only the training randomness (weight init + minibatch
noise) varies across ``--seeds`` runs, so ablation effects are isolated from data
variation. Every trained model is evaluated on the SAME held-out observations.

For each (variant, seed) we report:
  - divergence_rate, kappa_det       (stability; N1)
  - mDI_relerr, along_fiber_std      (direction/collapse; N2, vs reference)
  - coverage_ref_hpd95 (stoch & det) (mode coverage; N4/Thm3)
  - eps_inc = sliced-W2(forward-sweep(pi*), pi*)   (self-consistency certificate)

Outputs (results/paper2_experiments/b7/):
  b7_metrics.json           per-(variant,seed) + aggregate mean±std
  ../figures/figure_b7_ablation.pdf

Usage:
    conda activate vision_task
    python -m prob_models.paper.experiments.b7_ablation --seeds 5 --epochs 400
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prob_models.paper.experiments import _context as C
from prob_models.paper.experiments import _metrics as M
from prob_models.paper.experiments import b3_reference_posterior as B3
from prob_models.models import HiddenStateCVAE, ParameterCVAE
from prob_models.trainer import ProbTrainer

OUT_DIR = Path("results/paper2_experiments/b7")
FIG_DIR = Path("prob_models/paper/figures")

# training-noise ablation variants: (condition N1, target N2) std at train time
VARIANTS = {
    "full":       dict(cond=0.05, target=0.05),
    "no_condition": dict(cond=0.0, target=0.05),   # N1 off -> expect instability / kappa up
    "no_target":  dict(cond=0.05, target=0.0),     # N2 off -> expect wrong direction / collapse
}


# --------------------------- local pseudo-Gibbs (full control) ---------------------------
@torch.no_grad()
def pgs(Hnet, Pnet, x_obs, num_chains, steps, burn_in, latent="sample", inject=0.05,
        bounds=(-3.0, 3.0), record=False):
    """Alternating sampler with explicit control of latent (N3) and injection (N4) noise.

    latent='sample' -> z~N(0,I); latent='zero' -> decoder mean (deterministic map).
    inject -> N4 std added to decoder output. Returns final theta samples (C,2) for a
    single observation, and optionally the theta history (steps, C, 2).
    """
    device = x_obs.device
    xr = x_obs.repeat_interleave(num_chains, dim=0)
    B = xr.shape[0]
    theta = torch.randn(B, Pnet.theta_dim, device=device)
    hist = []
    def z(net):
        return torch.zeros(B, net.latent_dim, device=device) if latent == "zero" \
            else torch.randn(B, net.latent_dim, device=device)
    for t in range(steps):
        y = Hnet.decode(z(Hnet), xr, theta)
        if inject > 0:
            y = y + torch.randn_like(y) * inject
        theta = Pnet.decode(z(Pnet), xr, y)
        if inject > 0:
            theta = theta + torch.randn_like(theta) * inject
        if bounds is not None:
            theta = torch.clamp(theta, bounds[0], bounds[1])
        if record:
            hist.append(theta.detach().clone())
    keep = theta  # final state
    H = torch.stack(hist, 0) if record else None
    return keep, H


@torch.no_grad()
def det_sweep(Hnet, Pnet, x_obs, theta):
    """One deterministic mean sweep m(theta) = P.decode(0, x, H.decode(0, x, theta))."""
    B = theta.shape[0]
    zH = torch.zeros(B, Hnet.latent_dim, device=theta.device)
    zP = torch.zeros(B, Pnet.latent_dim, device=theta.device)
    y = Hnet.decode(zH, x_obs.repeat_interleave(B, dim=0) if x_obs.shape[0] == 1 else x_obs, theta)
    return Pnet.decode(zP, x_obs.repeat_interleave(B, dim=0) if x_obs.shape[0] == 1 else x_obs, y)


@torch.no_grad()
def estimate_kappa(Hnet, Pnet, x_obs, n_pairs=256, n_steps=15, delta=0.05):
    """Contraction rate of the deterministic mean map via synchronous coupling.

    kappa = geometric mean over steps of ||m(a)-m(b)|| / ||a-b||. <1 => contraction.
    """
    device = x_obs.device
    xr = x_obs.repeat_interleave(n_pairs, dim=0)
    a = torch.randn(n_pairs, Pnet.theta_dim, device=device)
    b = a + torch.randn_like(a) * delta
    ratios = []
    for _ in range(n_steps):
        d0 = (a - b).norm(dim=1) + 1e-8
        zH = torch.zeros(n_pairs, Hnet.latent_dim, device=device)
        zP = torch.zeros(n_pairs, Pnet.latent_dim, device=device)
        ya = Hnet.decode(zH, xr, a); yb = Hnet.decode(zH, xr, b)
        a2 = Pnet.decode(zP, xr, ya); b2 = Pnet.decode(zP, xr, yb)
        d1 = (a2 - b2).norm(dim=1) + 1e-8
        ratios.append((d1 / d0).clamp(1e-6, 1e6).log().mean().item())
        a, b = a2, b2
    return float(np.exp(np.mean(ratios)))


# --------------------------- reference posteriors (fixed eval obs) ---------------------------
def build_references(ctx, device, n_ref, grid, seed):
    """Analytic reference posteriors for n_ref fixed eval observations (mDI percentiles)."""
    normalizer, system = ctx["normalizer"], ctx["system"]
    dp = B3.load_priors()
    xs, ys, ps = [], [], []
    for xb, yb, pb in ctx["test_l"]:
        xs.append(xb); ys.append(yb); ps.append(pb)
    X = torch.cat(xs); Yh = torch.cat(ys); P = torch.cat(ps)
    P_phys = normalizer.denormalize_params(P.to(device)).cpu().numpy()
    mDI = P_phys[:, 0] * P_phys[:, 1]
    pcts = np.linspace(25, 75, n_ref)
    idxs = list(dict.fromkeys(int(np.argmin(np.abs(mDI - np.percentile(mDI, q)))) for q in pcts))

    class _A:  # lightweight args holder for B3.compute_reference
        pass
    a = _A(); a.grid = grid; a.si_max = 3.0; a.sigma_max = 3.0; a.rel_noise = 0.03; a.floor_noise = 2.0; a.seed = seed

    refs = []
    for idx in idxs:
        x_norm = X[idx:idx + 1].to(device)
        x_phys = normalizer.denormalize_inputs(x_norm, "observed").cpu().numpy().ravel()
        y_phys = normalizer.denormalize_inputs(Yh[idx:idx + 1].to(device), "hidden").cpu().numpy().ravel()
        G_obs = x_phys[0::2]; I0 = float(y_phys[0])
        si_ax, sg_ax, logL, post, _ = B3.compute_reference(system, G_obs, I0, dp, a)
        rng = np.random.default_rng(seed + idx)
        ref_s = M.sample_grid_density(si_ax, sg_ax, post, 12000, rng)
        refs.append(dict(idx=idx, x_norm=x_norm, si_ax=si_ax, sg_ax=sg_ax, post=post,
                         ref_samples=ref_s, true=P_phys[idx].copy(),
                         mDI_true=float(P_phys[idx, 0] * P_phys[idx, 1])))
    print(f"[B7] reference eval indices: {idxs}")
    return refs, X


# --------------------------- diagnostics for one trained model ---------------------------
def _uv(s):
    ls = np.log(np.clip(s[:, 0], 1e-6, None)); lg = np.log(np.clip(s[:, 1], 1e-6, None))
    return (ls - lg), (ls + lg)


def diagnose(Hnet, Pnet, refs, X_all, normalizer, device, args):
    Hnet.eval(); Pnet.eval()
    si_max = sg_max = 3.0

    # ---- stability on a broad batch of observations ----
    n_stab = min(args.n_stab, X_all.shape[0])
    xb = X_all[:n_stab].to(device)
    _, Hst = pgs(Hnet, Pnet, xb, num_chains=1, steps=args.steps, burn_in=0,
                 latent="sample", inject=args.inject, bounds=None, record=True)  # (steps, n_stab, 2)
    Hst = Hst.cpu().numpy()
    finite = np.isfinite(Hst).all(axis=-1)                       # (steps, n_stab)
    blew = (np.abs(Hst) > args.blowup).any(axis=-1) | (~finite)  # chain-step exceeded bounds
    divergence_rate = float(blew[-1].mean())                     # fraction diverged at final step
    late = np.nan_to_num(Hst[-5:], nan=args.blowup)
    dispersion = float(np.nanstd(np.clip(late, -args.blowup, args.blowup)))

    # kappa (deterministic contraction) averaged over a few obs
    kappas = [estimate_kappa(Hnet, Pnet, X_all[i:i + 1].to(device)) for i in [0, 1, 2, 3]]
    kappa_det = float(np.mean(kappas))

    # ---- direction / collapse / coverage / eps_inc on reference obs ----
    per = []
    for r in refs:
        x_norm = r["x_norm"]
        s_stoch, _ = pgs(Hnet, Pnet, x_norm, args.num_chains, args.steps, args.burn_in,
                         latent="sample", inject=args.inject)
        s_det, _ = pgs(Hnet, Pnet, x_norm, args.num_chains, args.steps, args.burn_in,
                       latent="zero", inject=0.0)
        st = normalizer.denormalize_params(s_stoch).cpu().numpy()
        dt = normalizer.denormalize_params(s_det).cpu().numpy()

        def clip_in(a):
            m = (a[:, 0] > 0) & (a[:, 0] < si_max) & (a[:, 1] > 0) & (a[:, 1] < sg_max)
            return a[m]
        st_i, dt_i = clip_in(st), clip_in(dt)
        rec = {"idx": r["idx"], "mDI_true": r["mDI_true"]}
        if st_i.shape[0] > 10:
            u_s, v_s = _uv(st_i); u_r, v_r = _uv(r["ref_samples"])
            mdi_s = st_i[:, 0] * st_i[:, 1]
            rec["mDI_relerr_stoch"] = float(abs(np.median(mdi_s) - r["mDI_true"]) / (r["mDI_true"] + 1e-9))
            rec["along_fiber_std_stoch"] = float(np.std(u_s))
            rec["coverage_stoch"] = float(M.coverage_in_grid_hpd(st_i, r["si_ax"], r["sg_ax"], r["post"], 0.95))
            rec["w1_along"] = float(M.w1_1d(u_s, u_r))
            # eps_inc: one extra stochastic forward sweep (H then P) applied to pi* samples;
            # small W2(sweep(pi*), pi*) => pi* is (near-)invariant / self-consistent.
            theta_t = s_stoch  # (C,2) normalized tensor on device
            xr = x_norm.repeat_interleave(theta_t.shape[0], dim=0)
            zH = torch.randn(theta_t.shape[0], Hnet.latent_dim, device=device)
            zP = torch.randn(theta_t.shape[0], Pnet.latent_dim, device=device)
            with torch.no_grad():
                y2 = Hnet.decode(zH, xr, theta_t)
                y2 = y2 + torch.randn_like(y2) * args.inject
                th2 = Pnet.decode(zP, xr, y2)
                th2 = th2 + torch.randn_like(th2) * args.inject
            th2_p = normalizer.denormalize_params(torch.clamp(th2, -3, 3)).cpu().numpy()
            th2_i = clip_in(th2_p)
            if th2_i.shape[0] > 10:
                rec["eps_inc"] = float(M.sliced_w2(st_i, th2_i))
        if dt_i.shape[0] > 5:
            u_d, _ = _uv(dt_i)
            rec["along_fiber_std_det"] = float(np.std(u_d))
            rec["coverage_det"] = float(M.coverage_in_grid_hpd(dt_i, r["si_ax"], r["sg_ax"], r["post"], 0.95))
        per.append(rec)

    def avg(key):
        vals = [p[key] for p in per if key in p and np.isfinite(p[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "divergence_rate": divergence_rate,
        "dispersion": dispersion,
        "kappa_det": kappa_det,
        "mDI_relerr_stoch": avg("mDI_relerr_stoch"),
        "along_fiber_std_stoch": avg("along_fiber_std_stoch"),
        "along_fiber_std_det": avg("along_fiber_std_det"),
        "coverage_stoch": avg("coverage_stoch"),
        "coverage_det": avg("coverage_det"),
        "w1_along": avg("w1_along"),
        "eps_inc": avg("eps_inc"),
    }


# --------------------------- train one model ---------------------------
def train_variant(ctx, cfg, cond_std, target_std, seed, results_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    device = cfg.DEVICE
    lh = cfg.LATENT_DIM_HIDDEN; lp = cfg.LATENT_DIM_PARAM; hd = cfg.HIDDEN_DIMS
    Hnet = HiddenStateCVAE(ctx["x_dim"], ctx["theta_dim"], ctx["y_dim"], latent_dim=lh, hidden_dims=hd).to(device)
    Pnet = ParameterCVAE(ctx["x_dim"], ctx["y_dim"], ctx["theta_dim"], latent_dim=lp, hidden_dims=hd).to(device)
    Pnet.theta_dim = ctx["theta_dim"]

    run_cfg = C.make_config(seed=seed, num_samples=cfg.NUM_SAMPLES, device=device)
    run_cfg.EPOCHS = cfg.EPOCHS
    run_cfg.USE_EARLY_STOPPING = True
    run_cfg.EARLY_STOPPING_PATIENCE = cfg.EARLY_STOPPING_PATIENCE
    run_cfg.CONDITION_NOISE_STD_Y = cond_std; run_cfg.CONDITION_NOISE_STD_P = cond_std
    run_cfg.TARGET_NOISE_STD_Y = target_std; run_cfg.TARGET_NOISE_STD_P = target_std
    run_cfg.RESULTS_DIR = results_dir
    trainer = ProbTrainer(ctx["train_l"], ctx["val_l"], run_cfg, hidden_cvae=Hnet, param_cvae=Pnet)
    Hnet, Pnet, _ = trainer.train()
    return Hnet, Pnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--n_ref", type=int, default=5)
    ap.add_argument("--n_stab", type=int, default=200)
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--num_chains", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--burn_in", type=int, default=30)
    ap.add_argument("--inject", type=float, default=0.05)
    ap.add_argument("--blowup", type=float, default=6.0)
    ap.add_argument("--variants", type=str, default="full,no_condition,no_target")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = C.make_config(seed=42, num_samples=args.num_samples, device=device)
    base.EPOCHS = args.epochs; base.EARLY_STOPPING_PATIENCE = args.patience
    ctx = C.load_data_context(base)                       # fixed data/split/normalizer
    refs, X_all = build_references(ctx, device, args.n_ref, args.grid, seed=42)

    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    results = {v: [] for v in variants}
    for v in variants:
        vc = VARIANTS[v]
        for s in range(1, args.seeds + 1):
            rd = f"/tmp/claude-1001/b7_ckpt/{v}_seed{s}"
            print(f"\n===== B7 variant={v} seed={s}  (cond={vc['cond']}, target={vc['target']}) =====")
            Hnet, Pnet = train_variant(ctx, base, vc["cond"], vc["target"], s, rd)
            diag = diagnose(Hnet, Pnet, refs, X_all, ctx["normalizer"], device, args)
            diag["variant"] = v; diag["seed"] = s
            results[v].append(diag)
            print(f"[B7] {v} s{s}: div={diag['divergence_rate']:.2f} kappa={diag['kappa_det']:.3f} "
                  f"mDIerr={diag['mDI_relerr_stoch']:.3f} along(stoch/det)={diag['along_fiber_std_stoch']:.3f}/"
                  f"{diag['along_fiber_std_det']:.3f} cov(stoch/det)={diag['coverage_stoch']:.2f}/"
                  f"{diag['coverage_det']:.2f} eps_inc={diag['eps_inc']:.3f}")
            # checkpoint metrics incrementally
            _dump(results)

    _dump(results)
    _plot(results)
    print(f"\n[B7] done -> {OUT_DIR}/b7_metrics.json  and  {FIG_DIR}/figure_b7_ablation.pdf")


def _agg(runs):
    keys = ["divergence_rate", "kappa_det", "mDI_relerr_stoch", "along_fiber_std_stoch",
            "along_fiber_std_det", "coverage_stoch", "coverage_det", "w1_along", "eps_inc", "dispersion"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and np.isfinite(r[k])]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else None
    return out


def _dump(results):
    payload = {"aggregate": {v: _agg(runs) for v, runs in results.items() if runs},
               "per_run": results}
    with open(OUT_DIR / "b7_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = {v: _agg(runs) for v, runs in results.items() if runs}
    variants = list(agg.keys())
    panels = [
        ("kappa_det", "kappa (contraction)\nN1: stability", 1.0),
        ("divergence_rate", "divergence rate\nN1: stability", None),
        ("mDI_relerr_stoch", "mDI rel. error\nN2: direction", None),
        ("along_fiber_std_stoch", "along-fiber std (stoch)\nN2/N4: non-collapse", None),
        ("coverage_stoch", "ref-HPD95 coverage (stoch)\nN4: mode coverage", None),
        ("eps_inc", "eps_inc = W2(sweep,pi*)\nself-consistency", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    colors = {"full": "#2a9d8f", "no_condition": "#e76f51", "no_target": "#e9c46a"}
    for ax, (key, title, hline) in zip(axes.ravel(), panels):
        means = [agg[v][key]["mean"] if agg[v].get(key) else np.nan for v in variants]
        stds = [agg[v][key]["std"] if agg[v].get(key) else 0 for v in variants]
        ax.bar(variants, means, yerr=stds, capsize=5,
               color=[colors.get(v, "gray") for v in variants], alpha=0.85)
        if hline is not None:
            ax.axhline(hline, color="red", ls="--", lw=1, label=f"={hline}")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    # add det-coverage comparison as text on coverage panel
    fig.suptitle("B7: three-guarantee isolation ablation (mean±std over seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "figure_b7_ablation.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
