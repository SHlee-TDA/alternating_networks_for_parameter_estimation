"""Fig 1 teaser (conceptual, self-contained — no data/checkpoints).
(a) Collapse to the mean: single-net / deterministic decoupled -> tight blob at the
    conditional mean E[theta|x_obs], off the fiber, in the low-density interior.
(b) Track the fiber: A-DCVAE -> crescent cloud hugging the non-identifiable fiber
    S_I * sigma = C through the truth, with alternating pseudo-Gibbs arrows.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

rng = np.random.default_rng(7)

C = 0.55                      # fiber constant  S_I * sigma = C
si = np.linspace(0.22, 2.5, 400)
sig = C / si                  # fiber curve
# truth on the fiber
si_star, sig_star = 0.7, C / 0.7

# conditional mean = centroid of fiber arc (lies inside the convex hyperbola -> off-fiber)
mask = (sig >= 0.1) & (sig <= 2.6)
mean_si = si[mask].mean()
mean_sig = sig[mask].mean()

def fiber_cloud(n, along=0.55, across=0.06):
    """points scattered along the fiber (crescent), small across-fiber width."""
    t = rng.normal(np.log(si_star), along, n)         # move along fiber in log-S_I
    s = np.exp(t)
    s = np.clip(s, 0.25, 2.4)
    base = C / s
    base = base * np.exp(rng.normal(0, across, n))     # across-fiber jitter
    return s, base

fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharex=True, sharey=True)
XLIM, YLIM = (0, 2.6), (0, 2.6)
teal = "#2a9d8f"
grey = "#9aa0a6"

for ax in axes:
    ax.plot(si, sig, "--", color=grey, lw=2.2, zorder=1)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"insulin sensitivity  $S_I$", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
axes[0].set_ylabel(r"secretion capacity  $\sigma$", fontsize=11)

# ---- Panel (a): collapse to the mean ----
axa = axes[0]
blob_si = rng.normal(mean_si, 0.045, 400)
blob_sig = rng.normal(mean_sig, 0.045, 400)
axa.scatter(blob_si, blob_sig, s=7, color="#e76f51", alpha=0.5, zorder=2, edgecolors="none")
axa.scatter([si_star], [sig_star], marker="*", s=290, color="#ffd166",
            edgecolors="black", linewidths=1.1, zorder=5)
axa.annotate(r"$\mathbb{E}[\theta\,|\,X_{\mathrm{obs}}]$",
             xy=(mean_si, mean_sig), xytext=(mean_si + 0.35, mean_sig + 0.65),
             fontsize=11, color="#c1440e",
             arrowprops=dict(arrowstyle="->", color="#c1440e", lw=1.4))
axa.text(0.03, 0.965, "(a) single-net / deterministic:\ncollapse to the mean",
         transform=axa.transAxes, va="top", ha="left", fontsize=10.5, fontweight="bold")
axa.text(1.35, 0.35, "non-identifiable\nfiber  $S_I\\,\\sigma=C$",
         color="#6b7075", fontsize=9.5, rotation=-33)

# ---- Panel (b): track the fiber ----
axb = axes[1]
cs_si, cs_sig = fiber_cloud(1300)
axb.scatter(cs_si, cs_sig, s=6, color=teal, alpha=0.4, zorder=2, edgecolors="none")
axb.scatter([si_star], [sig_star], marker="*", s=290, color="#ffd166",
            edgecolors="black", linewidths=1.1, zorder=5)
# alternating pseudo-Gibbs arrows along the fiber
pts = [(0.45, C/0.45), (0.62, C/0.62), (0.95, C/0.95), (1.35, C/1.35)]
for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
    axb.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                  arrowstyle="-|>", mutation_scale=12, color="#1d3557",
                  lw=1.3, alpha=0.9, zorder=4,
                  connectionstyle="arc3,rad=0.25"))
axb.text(0.03, 0.965, "(b) A-DCVAE:\ntrack the fiber",
         transform=axb.transAxes, va="top", ha="left", fontsize=10.5, fontweight="bold")
axb.text(1.15, 1.35, "pseudo-Gibbs\nsweeps", color="#1d3557", fontsize=9.0)

# shared legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffd166",
           markeredgecolor="black", markersize=15, label="truth"),
    Line2D([0], [0], ls="--", color=grey, lw=2.2, label=r"fiber $S_I\sigma=C$"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=teal, markersize=8,
           label="posterior samples"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))

from pathlib import Path
out = Path(__file__).resolve().parent.parent / "figures" / "figure1_teaser.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
print("conditional mean (off-fiber):", round(mean_si, 3), round(mean_sig, 3),
      "| product S_I*sigma =", round(mean_si * mean_sig, 3), "vs C =", C)
