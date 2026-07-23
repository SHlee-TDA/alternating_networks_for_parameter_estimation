"""
Part 2 (SIMODS review): conditional-mean locus figure + quantitative metrics.

Builds a single figure showing, in the (S_I, sigma) plane:
  (1) the ground-truth joint distribution,
  (2) representative constant-S_I*sigma fibers,
  (3) the estimated conditional-mean locus  c |-> E[(S_I,sigma) | S_I*sigma=c],
  (4) iterative fixed-point predictions,
  (5) direct-regression (matched-capacity baseline) predictions.

All quantities come from the SAVED predictions of the real pipeline; nothing is synthesized.
Iterative and baseline share one test split (same p_true), so both are compared against the
SAME ground-truth cloud and the SAME conditional-mean locus.

Reported metrics (per estimator), written to results/det_meanlocus/condmean_metrics.json:
  - dist_to_condmean : mean over product-bins of || mean(pred|bin) - E[theta|bin] ||
  - dist_to_midpoint : same, but to the on-fiber geometric point (sqrt c, sqrt c)  (selector check)
  - var_explained_by_locus : 1 - sum||pred_i - m(c_i)||^2 / sum||pred_i - mean(pred)||^2
  - dist_to_nearest_fiber : mean Euclidean distance from each prediction to its true fiber {S_I*sigma=c}
  - cond_product_bias : mean over bins of ( prod(mean(pred|bin)) - c )      (Jensen off-fiber signature)
  - mDI_relerr : mean |pred_prod - c| / c
Usage:  python experiments/det_meanlocus/condmean_locus_fig.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})

ITER = 'results/det_paper/ogtt_simul/main_sn0p9/extracted_preds.npz'
BASE = 'results/det_paper/ogtt_simul/baseline/extracted_preds.npz'
OUTFIG = 'paper/simods/figures/exp/ogtt_condmean_locus.pdf'
OUTJSON = 'results/det_meanlocus/condmean_metrics.json'
NBINS = 24
RNG = np.random.default_rng(0)


def load(p):
    d = np.load(p, allow_pickle=True)
    return d['p_true'].astype(float), d['p_pred'].astype(float)


def nearest_fiber_dist(pred, c):
    """Euclidean distance from pred=(a,b) to hyperbola {(t, c/t): t>0}, minimized over t on a refined grid."""
    a, b = pred[:, 0], pred[:, 1]
    # coarse grid in t around the geometric scale sqrt(c), then refine
    t = np.geomspace(np.maximum(1e-3, np.sqrt(np.maximum(c, 1e-9)) / 8),
                     np.sqrt(np.maximum(c, 1e-9)) * 8, 400)  # (400,) per-sample below
    dmin = np.empty(len(pred))
    for i in range(len(pred)):
        ti = t[:, i] if t.ndim == 2 else t
        # build per-sample grid
        s = np.sqrt(max(c[i], 1e-9))
        ti = np.geomspace(max(1e-3, s / 8), s * 8, 400)
        d2 = (ti - a[i]) ** 2 + (c[i] / ti - b[i]) ** 2
        dmin[i] = np.sqrt(d2.min())
    return dmin


def condmean_locus(pt, nb=NBINS):
    """Bin ground truth by product c and return (c_bin, mean_theta_bin) sorted by c."""
    c = pt[:, 0] * pt[:, 1]
    qs = np.quantile(c, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-9
    bid = np.clip(np.digitize(c, qs) - 1, 0, nb - 1)
    cb, mb = [], []
    for b in range(nb):
        m = bid == b
        if m.sum() < 20: continue
        cb.append(c[m].mean()); mb.append(pt[m].mean(0))
    return np.array(cb), np.array(mb), qs, bid


def metrics(pt, pp, qs):
    c = pt[:, 0] * pt[:, 1]
    bid = np.clip(np.digitize(c, qs) - 1, 0, len(qs) - 2)
    dcm, dmid, jbias, cbins = [], [], [], []
    m_of_c = np.zeros_like(pt)  # conditional mean at each sample's true c-bin
    for b in range(len(qs) - 1):
        m = bid == b
        if m.sum() < 20: continue
        mbin = pt[m].mean(0); pbin = pp[m].mean(0); cc = c[m].mean()
        dcm.append(np.linalg.norm(pbin - mbin))
        dmid.append(np.linalg.norm(pbin - np.array([np.sqrt(cc), np.sqrt(cc)])))
        jbias.append(mbin[0] * mbin[1] - cc)
        cbins.append(cc)
        m_of_c[m] = mbin
    dcm, dmid = np.array(dcm), np.array(dmid)
    # variance explained by the conditional-mean locus
    ss_res = np.sum((pp - m_of_c) ** 2)
    ss_tot = np.sum((pp - pp.mean(0)) ** 2)
    var_expl = 1.0 - ss_res / ss_tot
    # nearest-fiber Euclidean distance (subsample for speed)
    idx = RNG.choice(len(pp), size=min(3000, len(pp)), replace=False)
    nfd = nearest_fiber_dist(pp[idx], c[idx])
    cprod = pp[:, 0] * pp[:, 1]
    return dict(
        dist_to_condmean=float(dcm.mean()),
        dist_to_midpoint=float(dmid.mean()),
        selector=('CONDITIONAL-MEAN' if dcm.mean() < dmid.mean() else 'MIDPOINT/other'),
        var_explained_by_locus=float(var_expl),
        dist_to_nearest_fiber=float(nfd.mean()),
        cond_product_bias_mean=float(np.mean(jbias)),
        cond_product_bias_frac_pos=float(np.mean(np.array(jbias) > 0)),
        mDI_relerr=float(np.mean(np.abs(cprod - c) / np.maximum(c, 1e-9))),
        n=int(len(pp)),
    )


def main():
    pt_i, pp_i = load(ITER)
    pt_b, pp_b = load(BASE)
    assert np.allclose(pt_i, pt_b), "iter/baseline splits differ"
    pt = pt_i  # shared ground truth
    cb, mb, qs, _ = condmean_locus(pt)

    m_iter = metrics(pt, pp_i, qs)
    m_base = metrics(pt, pp_b, qs)
    out = {'iterative': m_iter, 'direct_baseline': m_base,
           'note': 'iterative and direct baseline share one test split; locus computed from shared ground truth.'}
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    json.dump(out, open(OUTJSON, 'w'), indent=2)
    print(json.dumps(out, indent=2))

    # ---- figure: two panels (iterative | direct baseline) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    c_all = pt[:, 0] * pt[:, 1]
    fiber_cs = np.quantile(c_all, [0.1, 0.3, 0.5, 0.7, 0.9])
    lim = (0, np.quantile(pt[:, 0], 0.995) * 1.05, 0, np.quantile(pt[:, 1], 0.995) * 1.05)
    for ax, (pp, name, r) in zip(axes, [(pp_i, 'Iterative fixed point', m_iter),
                                        (pp_b, 'Direct estimator (matched)', m_base)]):
        ax.scatter(pt[:, 0], pt[:, 1], s=3, c='0.78', alpha=0.35, label='ground truth', rasterized=True)
        # constant-product fibers
        for cc in fiber_cs:
            t = np.geomspace(max(1e-2, cc / (lim[3] + 1e-9)), min(lim[1], cc / 1e-2 + 1e-9), 200)
            ax.plot(t, cc / t, color='0.55', lw=0.8, ls=':')
        ax.scatter(pp[:, 0], pp[:, 1], s=4, c='#1f77b4', alpha=0.45, label='predictions', rasterized=True)
        ax.plot(mb[:, 0], mb[:, 1], '-', color='#d62728', lw=2.6,
                label='product-cond. arith. mean\n(reference geometry)', zorder=5)
        ax.set_title(name)
        ax.set_xlabel(r'$S_I$'); ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
        txt = (f"dist to arith.-mean locus {r['dist_to_condmean']:.3f}\n"
               f"dist to on-fiber midpoint {r['dist_to_midpoint']:.3f}\n"
               f"dist to nearest fiber {r['dist_to_nearest_fiber']:.3f}\n"
               f"reference off-fiber bias {r['cond_product_bias_mean']:+.3f}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.88))
    axes[0].set_ylabel(r'$\sigma$')
    axes[0].legend(loc='lower right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)
    plt.savefig(OUTFIG, dpi=200, bbox_inches='tight')
    print("saved", OUTFIG)


if __name__ == '__main__':
    main()
