"""Regenerate the B7 ablation figure from results/paper2_experiments/b7/b7_metrics.json.

Standalone so it can be re-run after the training driver finishes (or to restyle
without recomputing). The definitive collapse panel is the grouped along-fiber std
(stochastic vs deterministic ping-pong): deterministic collapses to ~0 across all
variants (regression-to-the-mean), stochastic retains the fiber spread (Thm 3 / N4).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/paper2_experiments/b7/b7_metrics.json")
FIG = Path("prob_models/paper/figures/figure_b7_ablation.pdf")
COLORS = {"full": "#2a9d8f", "no_condition": "#e76f51", "no_target": "#e9c46a"}
NICE = {"full": "full", "no_condition": "N1 off\n(no cond.)", "no_target": "N2 off\n(no target)"}


def _m(agg, v, k):
    return (agg[v][k]["mean"], agg[v][k]["std"]) if agg[v].get(k) else (np.nan, 0)


def main():
    d = json.load(open(OUT))
    agg = d["aggregate"]
    variants = [v for v in ["full", "no_condition", "no_target"] if v in agg]
    x = np.arange(len(variants))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))

    # (1) kappa — N1 stability
    ax = axes[0, 0]
    m, s = zip(*[_m(agg, v, "kappa_det") for v in variants])
    ax.bar(x, m, yerr=s, capsize=5, color=[COLORS[v] for v in variants], alpha=.85)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="κ=1 (stability limit)")
    ax.set_xticks(x); ax.set_xticklabels([NICE[v] for v in variants], fontsize=8)
    ax.set_title("κ: deterministic contraction\n(N1 → Thm 1 stability)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)

    # (2) divergence + dispersion — N1 stability
    ax = axes[0, 1]
    m, s = zip(*[_m(agg, v, "divergence_rate") for v in variants])
    ax.bar(x, m, yerr=s, capsize=5, color=[COLORS[v] for v in variants], alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels([NICE[v] for v in variants], fontsize=8)
    ax.set_title("chain divergence rate\n(N1 → stability)", fontsize=10)
    ax.grid(axis="y", alpha=.3)

    # (3) mDI rel err — N2 direction
    ax = axes[0, 2]
    m, s = zip(*[_m(agg, v, "mDI_relerr_stoch") for v in variants])
    ax.bar(x, m, yerr=s, capsize=5, color=[COLORS[v] for v in variants], alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels([NICE[v] for v in variants], fontsize=8)
    ax.set_title("across-fiber mDI rel. error\n(N2 → Thm 2 direction)", fontsize=10)
    ax.grid(axis="y", alpha=.3)

    # (4) along-fiber std: stochastic vs deterministic — THE collapse panel (N4 / Thm 3)
    ax = axes[1, 0]
    w = 0.38
    ms, ss = zip(*[_m(agg, v, "along_fiber_std_stoch") for v in variants])
    md, sd = zip(*[_m(agg, v, "along_fiber_std_det") for v in variants])
    ax.bar(x - w / 2, ms, w, yerr=ss, capsize=4, color="#2a9d8f", label="stochastic PGS")
    ax.bar(x + w / 2, md, w, yerr=sd, capsize=4, color="#adb5bd", label="deterministic PGS")
    ax.set_xticks(x); ax.set_xticklabels([NICE[v] for v in variants], fontsize=8)
    ax.set_title("along-fiber spread: stoch vs det\n(N4 → Thm 3 non-collapse)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)

    # (5) eps_inc — self-consistency certificate
    ax = axes[1, 1]
    m, s = zip(*[_m(agg, v, "eps_inc") for v in variants])
    ax.bar(x, m, yerr=s, capsize=5, color=[COLORS[v] for v in variants], alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels([NICE[v] for v in variants], fontsize=8)
    ax.set_title(r"$\varepsilon_{\mathrm{inc}}=W_2(\mathrm{sweep}(\pi^*),\pi^*)$"
                 "\nself-consistency", fontsize=10)
    ax.grid(axis="y", alpha=.3)

    # (6) coverage (stochastic) — reference HPD, with caveat note
    ax = axes[1, 2]
    m, s = zip(*[_m(agg, v, "coverage_stoch") for v in variants])
    ax.bar(x, m, yerr=s, capsize=5, color=[COLORS[v] for v in variants], alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels([NICE[v] for v in variants], fontsize=8)
    ax.set_title("stoch. π* mass in ref-HPD95\n(low ⇒ π* over-dispersed)", fontsize=10)
    ax.grid(axis="y", alpha=.3)

    fig.suptitle("B7: three-guarantee isolation ablation (mean ± std over 5 seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[b7_replot] -> {FIG}")
    # also print a compact table
    print("\nvariant        kappa   div   mDIerr  along(st/det)  eps_inc")
    for v in variants:
        g = agg[v]
        def gv(k): return g[k]["mean"] if g.get(k) else float("nan")
        print(f"{v:14s} {gv('kappa_det'):.3f}  {gv('divergence_rate'):.2f}  "
              f"{gv('mDI_relerr_stoch'):.3f}   {gv('along_fiber_std_stoch'):.2f}/"
              f"{gv('along_fiber_std_det'):.2f}      {gv('eps_inc'):.3f}")


if __name__ == "__main__":
    main()
