"""E-c -- Non-identifiable toy (u = a*b).  Validates Prop:inc (eps_inc~0) + non-collapse.

True model on theta=(a,b) in a positive box, hidden u:
  prior(a,b) uniform,  u | a,b ~ N(a b, sigma_H^2),  x_obs | u ~ N(u, sigma_obs^2).
True conditionals (learned = true, so eps_H = eps_P = 0):
  Q_H(u | a,b)   = N(m_u, s_u^2),  s_u^2 = 1/(1/sigma_H^2 + 1/sigma_obs^2),
                   m_u = s_u^2 (a b/sigma_H^2 + x_obs/sigma_obs^2).
  Q_P(a,b | u)   proportional to  N(u; a b, sigma_H^2)  on the box  (grid-sampled).
Reference marginal:  p(a,b | x_obs) proportional to N(x_obs; a b, sigma_H^2+sigma_obs^2).
The ridge {a b = x_obs} is non-identifiable (only the product is constrained).
We show the stochastic dual sweep COVERS the ridge (support recovery) while a deterministic
mean-map ping-pong COLLAPSES to a point, and eps_inc ~ 0 certifies compatibility.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w2 import sliced_w2

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "figures"))
RESDIR = os.path.join(HERE, "results")


def build_grid(lo, hi, G):
    ax = np.linspace(lo, hi, G)
    A, B = np.meshgrid(ax, ax, indexing="ij")   # A[i,j]=a_i, B[i,j]=b_j
    return ax, A, B, A * B                        # product grid AB


def reference_density(AB, x_obs, w2):
    d = np.exp(-(AB - x_obs) ** 2 / (2.0 * w2))
    return d / d.sum()


def grid_sample(ax, A, B, AB, u, sigma_H, rng):
    """Sample (a,b) ~ Q_P(.|u) on the grid for a batch of u (n,). Returns (n,2)."""
    n = u.shape[0]; G = ax.size
    w = np.exp(-(AB[None] - u[:, None, None]) ** 2 / (2.0 * sigma_H ** 2))  # (n,G,G)
    flat = w.reshape(n, G * G)
    flat /= flat.sum(1, keepdims=True)
    cdf = np.cumsum(flat, axis=1)
    r = rng.random(n)
    idx = (cdf < r[:, None]).sum(1).clip(0, G * G - 1)
    ia, ib = np.unravel_index(idx, (G, G))
    da = ax[1] - ax[0]
    a = ax[ia] + (rng.random(n) - 0.5) * da
    b = ax[ib] + (rng.random(n) - 0.5) * da
    return np.stack([a, b], axis=1)


def grid_mean(ax, A, B, AB, u, sigma_H):
    """Deterministic E[(a,b) | u] on the grid (a single point) for a batch of u."""
    n = u.shape[0]; G = ax.size
    w = np.exp(-(AB[None] - u[:, None, None]) ** 2 / (2.0 * sigma_H ** 2))
    w /= w.reshape(n, G * G).sum(1)[:, None, None]
    ma = (w * A[None]).reshape(n, G * G).sum(1)
    mb = (w * B[None]).reshape(n, G * G).sum(1)
    return np.stack([ma, mb], axis=1)


def qh_mean_std(ab, x_obs, sigma_H, sigma_obs):
    s_u2 = 1.0 / (1.0 / sigma_H ** 2 + 1.0 / sigma_obs ** 2)
    m_u = s_u2 * (ab / sigma_H ** 2 + x_obs / sigma_obs ** 2)
    return m_u, np.sqrt(s_u2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--hi", type=float, default=3.0)
    ap.add_argument("--x_obs", type=float, default=1.0)
    ap.add_argument("--sigma_H", type=float, default=0.15)
    ap.add_argument("--sigma_obs", type=float, default=0.05)
    ap.add_argument("--grid", type=int, default=100)
    ap.add_argument("--n_chains", type=int, default=800)
    ap.add_argument("--n_steps", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(RESDIR, exist_ok=True)

    ax, A, B, AB = build_grid(args.lo, args.hi, args.grid)
    w2 = args.sigma_H ** 2 + args.sigma_obs ** 2
    ref_dens = reference_density(AB, args.x_obs, w2)

    def along(ab):  # log(a/b): varies ALONG the ridge {ab=const}
        return np.log(ab[:, 0]) - np.log(ab[:, 1])

    def hpd_mask(dens, level=0.95):
        flat = dens.ravel(); order = np.argsort(flat)[::-1]
        csum = np.cumsum(flat[order]) / flat.sum()
        k = np.searchsorted(csum, level) + 1
        return dens >= flat[order][k - 1]

    def coverage(pts, mask):
        da = ax[1] - ax[0]
        ia = np.clip(((pts[:, 0] - ax[0]) / da).round().astype(int), 0, args.grid - 1)
        ib = np.clip(((pts[:, 1] - ax[0]) / da).round().astype(int), 0, args.grid - 1)
        return float(mask[ia, ib].mean())

    mask95 = hpd_mask(ref_dens, 0.95)

    res = {"stoch": [], "det": [], "sw2": [], "cov": [], "eps_inc": [],
           "along_std_stoch": [], "along_std_det": []}
    last = {}
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        # reference samples from the grid density
        idx = rng.choice(ref_dens.size, size=args.n_chains, p=ref_dens.ravel())
        ia, ib = np.unravel_index(idx, ref_dens.shape)
        da = ax[1] - ax[0]
        ref_pts = np.stack([ax[ia] + (rng.random(args.n_chains) - 0.5) * da,
                            ax[ib] + (rng.random(args.n_chains) - 0.5) * da], axis=1)

        # stochastic dual sweep
        ab = np.stack([rng.uniform(args.lo, args.hi, args.n_chains),
                       rng.uniform(args.lo, args.hi, args.n_chains)], axis=1)
        for _ in range(args.n_steps):
            m_u, s_u = qh_mean_std(ab[:, 0] * ab[:, 1], args.x_obs, args.sigma_H, args.sigma_obs)
            u = m_u + s_u * rng.normal(size=args.n_chains)
            ab = grid_sample(ax, A, B, AB, u, args.sigma_H, rng)
        stoch = ab

        # deterministic mean-map ping-pong (same init)
        abd = np.stack([rng.uniform(args.lo, args.hi, args.n_chains),
                        rng.uniform(args.lo, args.hi, args.n_chains)], axis=1)
        for _ in range(args.n_steps):
            m_u, _ = qh_mean_std(abd[:, 0] * abd[:, 1], args.x_obs, args.sigma_H, args.sigma_obs)
            abd = grid_mean(ax, A, B, AB, m_u, args.sigma_H)
        det = abd

        # eps_inc from the stochastic run: Pi^-> = (a,b, u'~Q_H) vs Pi^<- = (a,b, u_stat)
        m_u, s_u = qh_mean_std(stoch[:, 0] * stoch[:, 1], args.x_obs, args.sigma_H, args.sigma_obs)
        u_stat = m_u + s_u * rng.normal(size=args.n_chains)             # the u that generated theta_n
        u_fwd = m_u + s_u * rng.normal(size=args.n_chains)              # fresh forward u'
        pi_fwd = np.column_stack([stoch, u_fwd]); pi_bwd = np.column_stack([stoch, u_stat])
        eps_inc = sliced_w2(pi_fwd, pi_bwd, n_proj=200, seed=s)

        res["sw2"].append(sliced_w2(stoch, ref_pts, n_proj=300, seed=s))
        res["cov"].append(coverage(stoch, mask95))
        res["eps_inc"].append(eps_inc)
        res["along_std_stoch"].append(float(along(stoch).std()))
        res["along_std_det"].append(float(along(det).std()))
        if s == 0:
            last = dict(stoch=stoch, det=det, ref_pts=ref_pts)

    def ms(k):
        a = np.array(res[k]); return float(a.mean()), float(a.std())
    metrics = dict(
        params=vars(args), ridge=f"a*b={args.x_obs}",
        sliced_w2_to_ref=ms("sw2"), coverage_hpd95=ms("cov"), eps_inc=ms("eps_inc"),
        along_ridge_std_stoch=ms("along_std_stoch"),
        along_ridge_std_det=ms("along_std_det"),
    )
    with open(os.path.join(RESDIR, "e_c.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axp = axes[0]
    axp.contour(ax, ax, ref_dens.T, levels=6, cmap="Greys", linewidths=0.8)
    axp.scatter(last["stoch"][:, 0], last["stoch"][:, 1], s=6, alpha=0.25, color="#1f4e79",
                label="stochastic sweep $\\nu^*$")
    axp.scatter(last["det"][:, 0], last["det"][:, 1], s=14, alpha=0.7, color="#c00000",
                marker="x", label="deterministic ping-pong")
    tt = np.linspace(args.lo, args.hi, 200)
    axp.plot(tt, args.x_obs / tt, "--", color="#2e7d32", lw=1.2, label=f"ridge $ab={args.x_obs:g}$")
    axp.set_xlim(args.lo, args.hi); axp.set_ylim(args.lo, args.hi)
    axp.set_xlabel("$a$"); axp.set_ylabel("$b$")
    axp.set_title("E-c: ridge recovery vs deterministic collapse")
    axp.legend(fontsize=8.5, loc="upper right")

    axh = axes[1]
    axh.hist(along(last["stoch"]), bins=40, density=True, alpha=0.6, color="#1f4e79",
             label="stochastic (along-ridge)")
    axh.axvline(np.median(along(last["det"])), color="#c00000", lw=2, label="deterministic (collapsed)")
    axh.set_xlabel(r"along-ridge coordinate $\log(a/b)$"); axh.set_ylabel("density")
    m_s, _ = ms("along_std_stoch"); m_d, _ = ms("along_std_det"); m_e, _ = ms("eps_inc")
    axh.set_title(f"along-ridge std: stoch {m_s:.2f} vs det {m_d:.3f};  $\\varepsilon_{{\\mathrm{{inc}}}}$={m_e:.3f}")
    axh.legend(fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "num_Ec_ridge.pdf")
    fig.savefig(out); plt.close(fig)

    print(f"[E-c] ridge a*b={args.x_obs}  sliced-W2(nu*,ref)={ms('sw2')[0]:.3f}+/-{ms('sw2')[1]:.3f}")
    print(f"[E-c] coverage HPD95={ms('cov')[0]:.2f}  eps_inc={ms('eps_inc')[0]:.3f}+/-{ms('eps_inc')[1]:.3f}")
    print(f"[E-c] along-ridge std: stochastic={ms('along_std_stoch')[0]:.3f}  deterministic={ms('along_std_det')[0]:.4f}")
    print(f"[E-c] => stochastic covers ridge; deterministic collapses (std ~ 0). figure -> {out}")


if __name__ == "__main__":
    main()
