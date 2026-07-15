"""
B4 — Strong baselines: is decoupling the net effect?

The current paper only compares Single-CVAE vs Dual-CVAE (both CVAEs), so a
reviewer can attribute any difference to "CVAE vs CVAE" rather than to the
decoupling.  This script adds two self-contained (no external install) baselines
and compares all methods against the B3 analytic reference posterior on a fixed
set of observations:

  1. NPE-flow : a single-network conditional normalizing flow p(theta | x_obs)
                (RealNVP), the canonical amortized-posterior estimator.  If this
                recovers the S_I*sigma crescent *without* decoupling, the
                decoupling motivation is weakened -- so we must check it.
  2. det-reg  : a deterministic MSE regressor theta = f(x_obs).  A well-converged
                point estimator returns the conditional MEAN of the fiber -> a
                single uninformative point (conceptual collapse).  We show this
                collapse WITHOUT invoking the deterministic paper's unrelated
                optimisation pathology (G0 non-attracting fixed point).
  3. A-DCVAE  : the proposed dual-CVAE pseudo-Gibbs posterior pi* (canonical ckpt).

Metrics vs reference (mean±std over seeds and observations): sliced-W2, ref-HPD95
coverage, along-fiber std (structure), across-fiber mDI error.

Outputs (results/paper2_experiments/b4/):
  b4_metrics.json ; ../figures/figure_b4_baselines.pdf

Usage:
    conda activate vision_task
    python -m prob_models.paper.experiments.b4_baselines --seeds 5 --epochs 400
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prob_models.paper.experiments import _context as C
from prob_models.paper.experiments import _metrics as M
from prob_models.paper.experiments import b7_ablation as B7
from prob_models.infer import pseudo_gibbs_sampling

OUT_DIR = Path("results/paper2_experiments/b4")
FIG_DIR = Path("prob_models/paper/figures")


# --------------------------- conditional RealNVP (NPE) ---------------------------
class CondAffineCoupling(nn.Module):
    """Transform one of the 2 theta dims, conditioned on the other + context c."""
    def __init__(self, ctx_dim, keep_index, hidden=128):
        super().__init__()
        self.keep = keep_index          # dim passed through; other dim is transformed
        self.trans = 1 - keep_index
        self.net = nn.Sequential(
            nn.Linear(1 + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),       # (s, t) for the transformed dim
        )

    def _st(self, theta, c):
        h = self.net(torch.cat([theta[:, self.keep:self.keep + 1], c], dim=-1))
        s, t = h[:, 0:1], h[:, 1:2]
        return torch.tanh(s), t          # bounded log-scale for stability

    def forward(self, theta, c):         # theta -> z, returns logdet
        s, t = self._st(theta, c)
        z = theta.clone()
        z[:, self.trans:self.trans + 1] = (theta[:, self.trans:self.trans + 1] - t) * torch.exp(-s)
        return z, -s.squeeze(-1)

    def inverse(self, z, c):
        s, t = self._st(z, c)
        theta = z.clone()
        theta[:, self.trans:self.trans + 1] = z[:, self.trans:self.trans + 1] * torch.exp(s) + t
        return theta


class CondRealNVP(nn.Module):
    def __init__(self, ctx_dim, n_layers=8, hidden=128):
        super().__init__()
        self.layers = nn.ModuleList([CondAffineCoupling(ctx_dim, i % 2, hidden) for i in range(n_layers)])

    def log_prob(self, theta, c):
        z = theta; logdet = torch.zeros(theta.shape[0], device=theta.device)
        for L in self.layers:
            z, ld = L(z, c); logdet = logdet + ld
        base = -0.5 * (z ** 2).sum(-1) - theta.shape[1] * 0.5 * np.log(2 * np.pi)
        return base + logdet

    @torch.no_grad()
    def sample(self, c, n):
        c_rep = c.repeat_interleave(n, dim=0)
        z = torch.randn(c_rep.shape[0], 2, device=c.device)
        for L in reversed(self.layers):
            z = L.inverse(z, c_rep)
        return z


def train_npe_flow(ctx, cfg, seed, epochs, patience=40):
    torch.manual_seed(seed); np.random.seed(seed)
    device = cfg.DEVICE
    flow = CondRealNVP(ctx["x_dim"], n_layers=8, hidden=128).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(5, patience // 2))
    best, wait = np.inf, 0
    for ep in range(epochs):
        flow.train(); tot = 0.0; nb = 0
        for xb, _, pb in ctx["train_l"]:
            xb = xb.to(device); pb = pb.to(device)
            loss = -flow.log_prob(pb, xb).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        # validation
        flow.eval(); vtot = 0.0; vnb = 0
        with torch.no_grad():
            for xb, _, pb in ctx["val_l"]:
                xb = xb.to(device); pb = pb.to(device)
                vtot += (-flow.log_prob(pb, xb).mean()).item(); vnb += 1
        vloss = vtot / max(vnb, 1); sched.step(vloss)
        if vloss < best - 1e-4:
            best = vloss; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return flow


# --------------------------- deterministic MSE regressor ---------------------------
def train_det_regressor(ctx, cfg, seed, epochs, patience=40):
    from prob_models.models import build_mlp
    torch.manual_seed(seed); np.random.seed(seed)
    device = cfg.DEVICE
    net = nn.Sequential(build_mlp(ctx["x_dim"], ctx["theta_dim"], cfg.HIDDEN_DIMS), nn.Tanh()).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, wait = np.inf, 0
    for ep in range(epochs):
        net.train()
        for xb, _, pb in ctx["train_l"]:
            xb = xb.to(device); pb = pb.to(device)
            loss = nn.functional.mse_loss(net(xb), pb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); vtot = 0.0; vnb = 0
        with torch.no_grad():
            for xb, _, pb in ctx["val_l"]:
                xb = xb.to(device); pb = pb.to(device)
                vtot += nn.functional.mse_loss(net(xb), pb).item(); vnb += 1
        v = vtot / max(vnb, 1)
        if v < best - 1e-5:
            best = v; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return net


# --------------------------- evaluation ---------------------------
def _uv(s):
    ls = np.log(np.clip(s[:, 0], 1e-6, None)); lg = np.log(np.clip(s[:, 1], 1e-6, None))
    return (ls - lg), (ls + lg)


def eval_samples_vs_ref(samples_phys, r, si_max=3.0, sg_max=3.0):
    m = (samples_phys[:, 0] > 0) & (samples_phys[:, 0] < si_max) & \
        (samples_phys[:, 1] > 0) & (samples_phys[:, 1] < sg_max)
    s = samples_phys[m]
    if s.shape[0] < 5:
        return {}
    u, v = _uv(s); mdi = s[:, 0] * s[:, 1]
    return {
        "sliced_w2": M.sliced_w2(s, r["ref_samples"]),
        "coverage95": M.coverage_in_grid_hpd(s, r["si_ax"], r["sg_ax"], r["post"], 0.95),
        "along_fiber_std": float(np.std(u)),
        "mDI_relerr": float(abs(np.median(mdi) - r["mDI_true"]) / (r["mDI_true"] + 1e-9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--n_ref", type=int, default=5)
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--n_post", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=40,
                    help="early-stopping patience, applied uniformly to A-DCVAE, NPE-flow and det-reg")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = C.make_config(seed=42, num_samples=args.num_samples, device=device)
    base.EPOCHS = args.epochs
    base.EARLY_STOPPING_PATIENCE = args.patience
    ctx = C.load_data_context(base)
    normalizer = ctx["normalizer"]
    refs, _ = B7.build_references(ctx, device, args.n_ref, args.grid, seed=42)

    # All three methods are trained per-seed on the SAME 10k context (identical data,
    # split and normalizer) so the comparison is fair. A-DCVAE (dual, full noise) is
    # trained here rather than loaded from the canonical 50k checkpoint, which would
    # otherwise be denormalized with the wrong (10k) normalizer.
    methods = {m: [] for m in ["npe_flow", "det_reg", "adcvae"]}
    last = {}
    for s in range(1, args.seeds + 1):
        print(f"\n===== B4 seed {s} : training A-DCVAE + NPE-flow + det-regressor =====")
        Hnet, Pnet = B7.train_variant(ctx, base, 0.05, 0.05, s, f"/tmp/claude-1001/b4_ckpt/adcvae_seed{s}")
        flow = train_npe_flow(ctx, base, s, args.epochs, patience=args.patience)
        reg = train_det_regressor(ctx, base, s, args.epochs, patience=args.patience)
        for r in refs:
            x = r["x_norm"]
            with torch.no_grad():
                _, th, _ = pseudo_gibbs_sampling(Hnet, Pnet, x, num_chains=args.n_post, num_steps=60, burn_in=30)
            ap = normalizer.denormalize_params(th.squeeze(0).to(device)).cpu().numpy()
            e0 = eval_samples_vs_ref(ap, r)
            if e0:
                e0["idx"] = r["idx"]; e0["seed"] = s; methods["adcvae"].append(e0)
            fp = normalizer.denormalize_params(flow.sample(x, args.n_post).to(device)).cpu().numpy()
            e = eval_samples_vs_ref(fp, r)
            if e:
                e["idx"] = r["idx"]; e["seed"] = s; methods["npe_flow"].append(e)
            with torch.no_grad():
                pt = normalizer.denormalize_params(reg(x).to(device)).cpu().numpy()  # (1,2) point
            pt_cloud = pt + np.random.normal(0, 1e-4, size=(args.n_post, 2))
            e2 = eval_samples_vs_ref(pt_cloud, r)
            if e2:
                e2["idx"] = r["idx"]; e2["seed"] = s; methods["det_reg"].append(e2)
        _dump(methods)
        last = {"Hnet": Hnet, "Pnet": Pnet, "flow": flow, "reg": reg}
        # keep a representative sample-set for the figure from the last seed
        if s == args.seeds:
            _save_figure_data(ctx, base, refs, last, normalizer, device, args)

    _dump(methods)
    _plot(methods)
    print(f"\n[B4] done -> {OUT_DIR}/b4_metrics.json  and  {FIG_DIR}/figure_b4_baselines.pdf")


def _agg(runs):
    keys = ["sliced_w2", "coverage95", "along_fiber_std", "mDI_relerr"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and np.isfinite(r[k])]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else None
    return out


def _dump(methods):
    payload = {"aggregate": {m: _agg(runs) for m, runs in methods.items() if runs}, "per_run": methods}
    with open(OUT_DIR / "b4_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def _save_figure_data(ctx, cfg, refs, last, normalizer, device, args):
    r = refs[len(refs) // 2]
    x = r["x_norm"]
    npe = normalizer.denormalize_params(last["flow"].sample(x, args.n_post).to(device)).cpu().numpy()
    with torch.no_grad():
        pt = normalizer.denormalize_params(last["reg"](x).to(device)).cpu().numpy()
        _, th, _ = pseudo_gibbs_sampling(last["Hnet"], last["Pnet"], x, num_chains=args.n_post, num_steps=60, burn_in=30)
    ad = normalizer.denormalize_params(th.squeeze(0).to(device)).cpu().numpy()
    np.savez(OUT_DIR / "figure_data.npz",
             si_ax=r["si_ax"], sg_ax=r["sg_ax"], post=r["post"], true=r["true"],
             mDI_true=r["mDI_true"], npe=npe, det_pt=pt, adcvae=ad if ad is not None else np.zeros((0, 2)))


def _plot(methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = {m: _agg(runs) for m, runs in methods.items() if runs}
    order = [m for m in ["adcvae", "npe_flow", "det_reg"] if m in agg]
    labels = {"adcvae": "A-DCVAE\n(dual, ours)", "npe_flow": "NPE-flow\n(single)", "det_reg": "det-reg\n(point)"}
    colors = {"adcvae": "#2a9d8f", "npe_flow": "#457b9d", "det_reg": "#e76f51"}
    panels = [("coverage95", "ref-HPD95 coverage (↑)"),
              ("along_fiber_std", "along-fiber std (structure, ↑)"),
              ("sliced_w2", "sliced-W2 to reference (↓)"),
              ("mDI_relerr", "across-fiber mDI rel.err (↓)")]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3)
    # left: scatter overlay if figure_data exists
    ax0 = fig.add_subplot(gs[:, 0])
    fd_path = OUT_DIR / "figure_data.npz"
    if fd_path.exists():
        fd = np.load(str(fd_path))
        ax0.contourf(fd["si_ax"], fd["sg_ax"], fd["post"], levels=10, cmap="magma")
        mdi = float(fd["mDI_true"]); sr = np.linspace(max(0.02, mdi / 3.0), 3.0, 300)
        ax0.plot(sr, mdi / sr, "--", color="cyan", lw=1.5, label="valley")
        if fd["adcvae"].shape[0] > 0:
            ax0.scatter(fd["adcvae"][:, 0], fd["adcvae"][:, 1], s=3, c="lime", alpha=0.25, label="A-DCVAE")
        ax0.scatter(fd["npe"][:, 0], fd["npe"][:, 1], s=3, c="deepskyblue", alpha=0.25, label="NPE-flow")
        ax0.scatter(fd["det_pt"][:, 0], fd["det_pt"][:, 1], marker="X", s=160, c="red",
                    edgecolor="black", zorder=10, label="det-reg (point)")
        ax0.scatter(fd["true"][0], fd["true"][1], marker="*", s=260, c="white",
                    edgecolor="black", zorder=11, label="truth")
        ax0.set_xlim(0, 3); ax0.set_ylim(0, 3)
        ax0.set_xlabel(r"$S_I$"); ax0.set_ylabel(r"$\sigma$")
        ax0.set_title("Posterior vs reference (one obs)"); ax0.legend(fontsize=8, loc="upper right")

    axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
           fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, (key, title) in zip(axs, panels):
        means = [agg[m][key]["mean"] if agg[m].get(key) else np.nan for m in order]
        stds = [agg[m][key]["std"] if agg[m].get(key) else 0 for m in order]
        ax.bar([labels[m] for m in order], means, yerr=stds, capsize=5,
               color=[colors[m] for m in order], alpha=0.85)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3); ax.tick_params(labelsize=8)
    fig.suptitle("B4: does decoupling help? A-DCVAE vs single-net NPE vs deterministic point",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "figure_b4_baselines.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
