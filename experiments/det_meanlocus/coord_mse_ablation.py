"""
Physical- vs log-coordinate MSE ablation for OGTT (validates Remark 5.13 / Corollary 5.12 empirically).

A direct MSE regressor x_obs -> theta=(S_I,sigma) is trained under two loss coordinates:
  * 'log'      : target = minmax-normalized log(theta)  (the pipeline's coordinate)
  * 'physical' : target = minmax-normalized theta        (physical coordinate)
Prediction: denormalize back to physical theta in both cases.

Theory (Remark 5.13): under a product degeneracy S_I*sigma=c the fiber is
  * curved in physical coordinates  -> arithmetic conditional mean is OFF the fiber (product > c, Cor 5.12),
  * affine in log coordinates       -> geometric conditional mean stays ON the fiber (product = c).
So we expect physical-MSE predictions to sit off the fiber with product biased ABOVE c, and log-MSE
predictions to sit on the fiber with the product preserved.

Runs multiple seeds and reports mean +/- std. CPU by default (small model; avoids GPU contention).
Usage: python experiments/det_meanlocus/coord_mse_ablation.py [--seeds 0,1,2] [--epochs 300] [--device cpu]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 14,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})
from scipy.stats import pearsonr

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', default='0,1,2,3,4')
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--device', default='cpu')
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(',')]
DEV = args.device
OUT = 'results/det_meanlocus'; os.makedirs(OUT, exist_ok=True)
DATA = 'data/ogtt_simul/augmented_data_ode_noderiv_100000.npz'
NBINS = 20


class Reg(nn.Module):
    def __init__(s, h=256):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(5, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                              nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2), nn.Tanh())
    def forward(s, x): return s.net(x)


def run_one(coord, seed, epochs):
    np.random.seed(seed); torch.manual_seed(seed)
    d = np.load(DATA)
    X = d['observed_data'].reshape(d['observed_data'].shape[0], -1).astype(np.float32)  # (N,5)
    P = d['params'].astype(np.float32)                                                  # (N,2) physical
    N = len(X); idx = np.random.permutation(N); ntr = int(0.85 * N)
    tr, te = idx[:ntr], idx[ntr:]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xn = (X - xm) / xs
    # target coordinate
    U = np.log(np.clip(P, 1e-8, None)) if coord == 'log' else P.copy()
    umin, umax = U[tr].min(0), U[tr].max(0)
    rng = (umax - umin) + 1e-8
    Un = (2 * (U - umin) / rng - 1).astype(np.float32)
    def denorm(un):
        u = (un + 1) / 2 * rng + umin
        return np.exp(u) if coord == 'log' else u
    Xtr = torch.tensor(Xn[tr], device=DEV); Utr = torch.tensor(Un[tr], device=DEV)
    Xte = torch.tensor(Xn[te], device=DEV)
    m = Reg().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lossf = nn.MSELoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(Xtr), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad(); lossf(m(Xtr[b]), Utr[b]).backward(); opt.step()
    with torch.no_grad():
        pred = denorm(m(Xte).cpu().numpy())
    return P[te].astype(float), pred.astype(float)


def _fiber_dist(pred, c, rng):
    """Per-point Euclidean distance from pred=(a,b) to the hyperbola {s*t=c}, minimized over s."""
    idx = rng.choice(len(pred), size=min(3000, len(pred)), replace=False)
    d = np.empty(len(idx))
    for j, i in enumerate(idx):
        s = np.sqrt(max(c[i], 1e-9))
        ti = np.geomspace(max(1e-3, s / 8), s * 8, 300)
        d[j] = np.sqrt(np.min((ti - pred[i, 0]) ** 2 + (c[i] / ti - pred[i, 1]) ** 2))
    return float(d.mean())


def metrics(pt, pp, rng):
    # Per-POINT indicators (avoid the Jensen-of-averaging artifact of per-bin means).
    c = pt[:, 0] * pt[:, 1]; cp = pp[:, 0] * pp[:, 1]
    return dict(
        r_si=float(pearsonr(pt[:, 0], pp[:, 0])[0]),
        r_sigma=float(pearsonr(pt[:, 1], pp[:, 1])[0]),
        r_logprod=float(pearsonr(np.log(np.clip(c, 1e-8, None)), np.log(np.clip(cp, 1e-8, None)))[0]),
        mDI_relerr=float(np.mean(np.abs(cp - c) / np.maximum(c, 1e-9))),           # per-point product error
        prod_signed_relbias=float(np.mean(cp / np.maximum(c, 1e-9) - 1.0)),         # >0 => off-fiber convex side
        fiber_dist=_fiber_dist(pp, c, rng),                                        # per-point distance to fiber
    )


def main():
    rng = np.random.default_rng(0)
    agg = {}
    last = {}
    for coord in ['log', 'physical']:
        per = []
        for s in SEEDS:
            pt, pp = run_one(coord, s, args.epochs)
            mk = metrics(pt, pp, rng); per.append(mk)
            print(f"[{coord} seed {s}] r_logprod={mk['r_logprod']:.3f} mDI_err={mk['mDI_relerr']:.3f} "
                  f"prod_signed_relbias={mk['prod_signed_relbias']:+.3f} fiber_dist={mk['fiber_dist']:.3f} "
                  f"r(S_I)={mk['r_si']:.3f} r(sig)={mk['r_sigma']:.3f}", flush=True)
            if s == SEEDS[-1]: last[coord] = (pt, pp)
        keys = per[0].keys()
        agg[coord] = {k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))] for k in keys}
        agg[coord]['n_seeds'] = len(SEEDS)
    json.dump(agg, open(f'{OUT}/coord_mse_ablation.json', 'w'), indent=2)
    print("\n=== SUMMARY (mean +/- std over %d seeds) ===" % len(SEEDS))
    for coord in ['log', 'physical']:
        a = agg[coord]
        print(f"[{coord:8s}] mDI_relerr={a['mDI_relerr'][0]:.3f}+/-{a['mDI_relerr'][1]:.3f} "
              f"signed_relbias={a['prod_signed_relbias'][0]:+.3f}+/-{a['prod_signed_relbias'][1]:.3f} "
              f"fiber_dist={a['fiber_dist'][0]:.3f}+/-{a['fiber_dist'][1]:.3f} "
              f"r(S_I)={a['r_si'][0]:.3f} r(sig)={a['r_sigma'][0]:.3f}")

    # ---- figure: physical vs log predictions vs fibers ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True, sharey=True)
    for a, coord, ttl in [(ax[0], 'log', 'log-coordinate MSE'), (ax[1], 'physical', 'physical-coordinate MSE')]:
        pt, pp = last[coord]
        c = pt[:, 0] * pt[:, 1]
        lim = (0, np.quantile(pt[:, 0], 0.99) * 1.05, 0, np.quantile(pt[:, 1], 0.99) * 1.05)
        a.scatter(pt[:, 0], pt[:, 1], s=3, c='0.8', alpha=0.3, rasterized=True, label='ground truth')
        for cc in np.quantile(c, [0.15, 0.4, 0.65, 0.9]):
            t = np.geomspace(max(1e-2, cc / (lim[3] + 1e-9)), min(lim[1], cc / 1e-2), 200)
            a.plot(t, cc / t, color='0.55', lw=0.8, ls=':')
        a.scatter(pp[:, 0], pp[:, 1], s=4, c='#1f77b4', alpha=0.4, rasterized=True, label='predictions')
        mk = metrics(pt, pp, np.random.default_rng(0))
        onoff = 'off-fiber' if mk['fiber_dist'] > 0.02 else 'on-fiber'
        a.set_title(f'{ttl}\nmDI err {mk["mDI_relerr"]:.2f}, signed prod. bias {mk["prod_signed_relbias"]:+.2f} ({onoff})')
        a.set_xlabel(r'$S_I$'); a.set_xlim(*lim[:2]); a.set_ylim(*lim[2:])
    ax[0].set_ylabel(r'$\sigma$'); ax[0].legend(loc='upper right', fontsize=10)
    plt.tight_layout(); fig.savefig('paper/simods/figures/exp/ogtt_coord_ablation.pdf', dpi=200, bbox_inches='tight')
    print("saved paper/simods/figures/exp/ogtt_coord_ablation.pdf")


if __name__ == '__main__':
    main()
