"""
B3 — Analytic reference (gold-standard) posterior for OGTT.

Because glucose G(t) depends on the parameters (S_I, sigma) only through the
product  mDI = S_I * sigma  (Ha et al., dimensional-analysis proof; DISCUSSION.md
B2), the likelihood of a glucose-only observation is a function of mDI alone.  We
therefore obtain a cheap, exact 2-D reference posterior by brute force:

    p(S_I, sigma | G_obs)  proportional to  L(G_obs | forward_sim(S_I, sigma)) * prior(S_I, sigma)

evaluated on a dense physical grid via forward simulation of the OGTT ODE.  This
requires no ABC / long-MCMC.  We then compare the trained A-DCVAE posterior
(pi*, pseudo-Gibbs over the iter_cvae checkpoint) against this reference in
sliced-2-Wasserstein, HPD coverage, and along-/across-fiber marginals.

Outputs (results/paper2_experiments/b3/):
    reference_posterior.npz    grid, logL, logprior, posterior, chosen observation
    b3_metrics.json            reference summary + pi* comparison metrics
    ../figures/figure_b3_reference_posterior.pdf

Usage:
    conda activate vision_task
    python -m prob_models.paper.experiments.b3_reference_posterior [--sample_idx N] [--grid 80]
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.stats import lognorm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prob_models.paper.experiments import _context as C
from prob_models.paper.experiments import _metrics as M
from prob_models.infer import pseudo_gibbs_sampling
from systems.ogtt_simul import OGTTModel, ODE_PARAMS, SYS_PARAMS

OUT_DIR = Path("results/paper2_experiments/b3")
FIG_DIR = Path("prob_models/paper/figures")


def load_priors():
    with open("data/parameters/distribution_params.json") as f:
        dp = json.load(f)
    return dp


def prior_logpdf(si, sigma, dp):
    lsi = lognorm.logpdf(si, s=dp["si"]["s"], loc=dp["si"]["loc"], scale=dp["si"]["scale"])
    lsg = lognorm.logpdf(sigma, s=dp["sigma"]["s"], loc=dp["sigma"]["loc"], scale=dp["sigma"]["scale"])
    return lsi + lsg


def forward_glucose(system, si, sigma, y0, t_points):
    """Forward-sim the OGTT ODE, return G at t_points (or None on failure)."""
    try:
        sol = solve_ivp(
            fun=lambda t, y: system.ode_func(t, y, [si, sigma]),
            t_span=[float(t_points[0]), float(t_points[-1])],
            y0=y0, t_eval=t_points, method="BDF",  # stiff solver (matches OGTTModel.simulate)
        )
        if not sol.success or sol.y.shape[1] != len(t_points):
            return None
        return sol.y[0]  # glucose row
    except Exception:
        return None


def steady_init(G0, I0, si, sigma):
    model = OGTTModel(ODE_PARAMS, SYS_PARAMS, {"si": si, "sigma": sigma})
    n5, n6 = model.find_steady_state_N(G0)
    return [G0, I0, n5, n6]


def compute_reference(system, G_obs, I0, dp, args):
    """Analytic reference posterior on the (S_I, sigma) grid for one observation."""
    n = args.grid
    si_ax = np.linspace(0.02, args.si_max, n)
    sg_ax = np.linspace(0.02, args.sigma_max, n)
    sigma_obs = np.maximum(args.rel_noise * G_obs, args.floor_noise)
    logL = np.full((n, n), -np.inf)   # [iy=sigma, ix=si]
    for ix, si in enumerate(si_ax):
        for iy, sg in enumerate(sg_ax):
            y0 = steady_init(G_obs[0], I0, si, sg)
            Gsim = forward_glucose(system, si, sg, y0, np.asarray(system.t_points, float))
            if Gsim is None or not np.all(np.isfinite(Gsim)):
                continue
            logL[iy, ix] = -0.5 * np.sum(((Gsim - G_obs) / sigma_obs) ** 2)
    SI, SG = np.meshgrid(si_ax, sg_ax)
    logpost = logL + prior_logpdf(SI, SG, dp)
    logpost -= np.nanmax(logpost)
    post = np.exp(logpost); post[~np.isfinite(post)] = 0.0; post /= post.sum()
    return si_ax, sg_ax, logL, post, sigma_obs


def _uv(s):
    ls = np.log(np.clip(s[:, 0], 1e-6, None)); lg = np.log(np.clip(s[:, 1], 1e-6, None))
    return (ls - lg), (ls + lg)   # (along-fiber, across-fiber=log mDI)


def evaluate_sample(idx, X, Yh, P_phys, normalizer, system, dp, args, net, device):
    """Full per-observation pipeline: reference + pi* comparison metrics."""
    x_obs_norm = X[idx:idx + 1].to(device)
    x_obs_phys = normalizer.denormalize_inputs(x_obs_norm, "observed").cpu().numpy().ravel()
    y_hid_phys = normalizer.denormalize_inputs(Yh[idx:idx + 1].to(device), "hidden").cpu().numpy().ravel()
    G_obs = x_obs_phys[0::2]
    I0 = float(y_hid_phys[0])
    si_true, sigma_true = float(P_phys[idx, 0]), float(P_phys[idx, 1])
    mDI_true = si_true * sigma_true

    si_ax, sg_ax, logL, post, sigma_obs = compute_reference(system, G_obs, I0, dp, args)
    rng = np.random.default_rng(args.seed + idx)
    ref_samples = M.sample_grid_density(si_ax, sg_ax, post, 20000, rng)
    ref_mDI = ref_samples[:, 0] * ref_samples[:, 1]
    map_iy, map_ix = np.unravel_index(int(np.argmax(post)), post.shape)

    rec = {
        "sample_idx": int(idx), "true_SI": si_true, "true_sigma": sigma_true, "true_mDI": mDI_true,
        "MAP_SI": float(si_ax[map_ix]), "MAP_sigma": float(sg_ax[map_iy]),
        "MAP_mDI": float(si_ax[map_ix] * sg_ax[map_iy]),
        "ref_mDI_mean": float(ref_mDI.mean()), "ref_mDI_std": float(ref_mDI.std()),
        "ref_mDI_cv": float(ref_mDI.std() / (ref_mDI.mean() + 1e-9)),
        "MAP_mDI_rel_err": float(abs(si_ax[map_ix] * sg_ax[map_iy] - mDI_true) / (mDI_true + 1e-9)),
    }

    net_samples = None
    if net is not None:
        Hnet, Pnet = net
        with torch.no_grad():
            _, theta_s, _ = pseudo_gibbs_sampling(Hnet, Pnet, x_obs_norm,
                                                  num_chains=args.num_chains, num_steps=args.steps, burn_in=args.burn_in)
        theta_s = theta_s.squeeze(0)
        net_samples = normalizer.denormalize_params(theta_s.to(device)).cpu().numpy()
        m = (net_samples[:, 0] > 0) & (net_samples[:, 0] < args.si_max) & \
            (net_samples[:, 1] > 0) & (net_samples[:, 1] < args.sigma_max)
        net_in = net_samples[m]
        if net_in.shape[0] > 10:
            u_net, v_net = _uv(net_in); u_ref, v_ref = _uv(ref_samples)
            net_mDI = net_in[:, 0] * net_in[:, 1]
            rec.update({
                "n_net_samples": int(net_in.shape[0]),
                "sliced_w2_joint": M.sliced_w2(net_in, ref_samples),
                "coverage_net_in_ref_hpd95": M.coverage_in_grid_hpd(net_in, si_ax, sg_ax, post, 0.95),
                "w1_across_fiber_logmDI": M.w1_1d(v_net, v_ref),
                "w1_along_fiber_logratio": M.w1_1d(u_net, u_ref),
                "net_mDI_mean": float(net_mDI.mean()), "net_mDI_std": float(net_mDI.std()),
                "net_mDI_cv": float(net_mDI.std() / (net_mDI.mean() + 1e-9)),
            })
    return rec, (si_ax, sg_ax, post, si_true, sigma_true, mDI_true, net_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_idx", type=int, default=-1, help="single index; overrides --n_samples")
    ap.add_argument("--n_samples", type=int, default=8, help="number of observations spanning mDI percentiles")
    ap.add_argument("--grid", type=int, default=70)
    ap.add_argument("--si_max", type=float, default=3.0)
    ap.add_argument("--sigma_max", type=float, default=3.0)
    ap.add_argument("--rel_noise", type=float, default=0.03)
    ap.add_argument("--floor_noise", type=float, default=2.0)
    ap.add_argument("--num_chains", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--burn_in", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_net", action="store_true", help="skip pi* comparison (reference only)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = C.make_config(seed=args.seed, device=device)
    ctx = C.load_data_context(cfg)
    normalizer, system = ctx["normalizer"], ctx["system"]

    xs, ys, ps = [], [], []
    for xb, yb, pb in ctx["test_l"]:
        xs.append(xb); ys.append(yb); ps.append(pb)
    X = torch.cat(xs); Yh = torch.cat(ys); P = torch.cat(ps)
    P_phys = normalizer.denormalize_params(P.to(device)).cpu().numpy()
    mDI_all = P_phys[:, 0] * P_phys[:, 1]
    dp = load_priors()

    # choose observation indices: single, or spread across mDI percentiles
    if args.sample_idx >= 0:
        indices = [args.sample_idx]
    else:
        pcts = np.linspace(20, 80, args.n_samples)
        indices = [int(np.argmin(np.abs(mDI_all - np.percentile(mDI_all, q)))) for q in pcts]
        indices = list(dict.fromkeys(indices))  # dedupe, keep order

    net = None if args.no_net else C.load_dual_cvae(cfg, ctx)

    records, first_plot = [], None
    for k, idx in enumerate(indices):
        rec, plotdata = evaluate_sample(idx, X, Yh, P_phys, normalizer, system, dp, args, net, device)
        records.append(rec)
        if k == len(indices) // 2:   # representative (median mDI) for the figure
            first_plot = plotdata
        print(f"[B3] idx={idx:5d} trueMDI={rec['true_mDI']:.3f} MAPmDI={rec['MAP_mDI']:.3f} "
              f"(relerr={rec['MAP_mDI_rel_err']*100:.1f}%) refCV={rec['ref_mDI_cv']:.3f}"
              + (f" | W2={rec.get('sliced_w2_joint',float('nan')):.3f} cov95={rec.get('coverage_net_in_ref_hpd95',float('nan'))*100:.0f}%"
                 f" W1mDI={rec.get('w1_across_fiber_logmDI',float('nan')):.3f}" if net is not None else ""))

    # aggregate mean±std over observations
    def agg(key):
        vals = [r[key] for r in records if key in r and np.isfinite(r[key])]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else None
    agg_keys = ["MAP_mDI_rel_err", "ref_mDI_cv", "sliced_w2_joint", "coverage_net_in_ref_hpd95",
                "w1_across_fiber_logmDI", "w1_along_fiber_logratio", "net_mDI_cv"]
    aggregate = {k: agg(k) for k in agg_keys}

    with open(OUT_DIR / "b3_metrics.json", "w") as f:
        json.dump({"config": vars(args), "aggregate": aggregate, "per_sample": records}, f, indent=2)

    # save reference grid of the representative sample + figure
    if first_plot is not None:
        si_ax, sg_ax, post, si_true, sigma_true, mDI_true, net_samples = first_plot
        np.savez(OUT_DIR / "reference_posterior.npz",
                 si_ax=si_ax, sg_ax=sg_ax, posterior=post,
                 true_theta=np.array([si_true, sigma_true]), mDI_true=mDI_true)
        _plot(si_ax, sg_ax, post, si_true, sigma_true, mDI_true, net_samples, args)

    print("\n[B3] AGGREGATE over", len(records), "observations:")
    for k, v in aggregate.items():
        if v:
            print(f"   {k:28s} = {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
    print(f"[B3] done -> {OUT_DIR}/b3_metrics.json  and  {FIG_DIR}/figure_b3_reference_posterior.pdf")


def _plot(si_ax, sg_ax, post, si_true, sigma_true, mDI_true, net_samples, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    cf = ax.contourf(si_ax, sg_ax, post, levels=12, cmap="magma")
    plt.colorbar(cf, ax=ax, label="reference posterior density")
    sr = np.linspace(max(0.02, mDI_true / args.sigma_max), args.si_max, 400)
    ax.plot(sr, mDI_true / sr, color="cyan", ls="--", lw=2, label=r"valley $S_I\!\cdot\!\sigma=\mathrm{mDI}$")
    if net_samples is not None:
        ax.scatter(net_samples[:, 0], net_samples[:, 1], s=4, c="lime", alpha=0.25, label=r"A-DCVAE $\pi^*$ samples")
    ax.scatter([si_true], [sigma_true], marker="*", s=320, c="white", edgecolor="black", lw=1.4, zorder=10, label="ground truth")
    ax.set_xlim(0, args.si_max); ax.set_ylim(0, args.sigma_max)
    ax.set_xlabel(r"Insulin sensitivity $S_I$"); ax.set_ylabel(r"Secretion capacity $\sigma$")
    ax.set_title("B3: analytic reference posterior vs A-DCVAE")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_b3_reference_posterior.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
