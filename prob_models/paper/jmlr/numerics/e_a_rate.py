"""E-a -- Linear-Gaussian, compatible.  Validates Thm:rate (kappa^n) + Thm:B (exactness).

A genuine bivariate Gaussian (u, theta) ~ N(0, Sigma) supplies the truth; the learned
conditionals are set equal to the true ones (eps_H = eps_P = 0). Everything is closed form:
  a_H = rho sigma_u/sigma_theta,  a_P = rho sigma_theta/sigma_u,  kappa = a_H a_P = rho^2,
  s_H^2 = sigma_u^2 (1-rho^2),    s_P^2 = sigma_theta^2 (1-rho^2).
The theta-chain is the AR(1)  theta_{n+1} = rho^2 theta_n + zeta,  Var(zeta) = a_P^2 s_H^2 + s_P^2,
with stationary law N(0, sigma_theta^2) = nu^dagger  (=> Thm:B exactness).
Law(theta_n | theta_0) = N(rho^{2n} theta_0,  sigma_theta^2 (1 - rho^{4n})).
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w2 import gaussian_w2_1d, emp_w2_1d

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "figures"))
RESDIR = os.path.join(HERE, "results")


def model(sigma_u, sigma_theta, rho):
    a_H = rho * sigma_u / sigma_theta
    a_P = rho * sigma_theta / sigma_u
    s_H2 = sigma_u ** 2 * (1 - rho ** 2)
    s_P2 = sigma_theta ** 2 * (1 - rho ** 2)
    kappa = a_H * a_P                       # = rho^2
    var_zeta = a_P ** 2 * s_H2 + s_P2       # = sigma_theta^2 (1 - rho^4)
    return dict(a_H=a_H, a_P=a_P, s_H2=s_H2, s_P2=s_P2, kappa=kappa, var_zeta=var_zeta)


def run_chain(m, theta0, n_chains, n_steps, rng):
    a_H, a_P = m["a_H"], m["a_P"]
    s_H, s_P = np.sqrt(m["s_H2"]), np.sqrt(m["s_P2"])
    th = np.full(n_chains, float(theta0))
    traj = np.empty((n_steps + 1, n_chains))
    traj[0] = th
    for n in range(1, n_steps + 1):
        u = a_H * th + s_H * rng.normal(size=n_chains)
        th = a_P * u + s_P * rng.normal(size=n_chains)
        traj[n] = th
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma_u", type=float, default=2.0)
    ap.add_argument("--sigma_theta", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--theta0", type=float, default=6.0)
    ap.add_argument("--n_steps", type=int, default=25)
    ap.add_argument("--n_chains", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESDIR, exist_ok=True)

    sig_t = args.sigma_theta
    m = model(args.sigma_u, sig_t, args.rho)
    kappa = m["kappa"]
    ns = np.arange(args.n_steps + 1)

    # ---- closed form: law(theta_n) = N(m_n, v_n) ------------------------------
    m_n = args.theta0 * kappa ** ns
    v_n = sig_t ** 2 * (1.0 - (kappa ** 2) ** ns)            # rho^{4n} = kappa^{2n}
    w2_cf = np.array([gaussian_w2_1d(mm, vv, 0.0, sig_t ** 2) for mm, vv in zip(m_n, v_n)])
    w2_mu0 = gaussian_w2_1d(args.theta0, 0.0, 0.0, sig_t ** 2)   # W2(delta_theta0, nu*)
    thm_bound = kappa ** ns * w2_mu0                          # Thm:rate: <= kappa^n W2(mu0, nu*)

    # ---- Monte-Carlo estimate across seeds ------------------------------------
    mc = np.zeros((args.seeds, args.n_steps + 1))
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        traj = run_chain(m, args.theta0, args.n_chains, args.n_steps, rng)
        ref = sig_t * rng.normal(size=args.n_chains)          # nu^dagger = N(0, sigma_theta^2)
        mc[s] = [emp_w2_1d(traj[n], ref) for n in range(args.n_steps + 1)]
    mc_mean, mc_std = mc.mean(0), mc.std(0)

    # ---- checks ---------------------------------------------------------------
    exactness = abs(np.sqrt(v_n[-1]) - sig_t)                 # -> 0 (Thm:B exactness)
    bound_ok = bool(np.all(w2_cf <= thm_bound + 1e-9))
    # empirical decay slope of closed-form W2 (skip n=0), vs log kappa
    good = w2_cf[1:] > 1e-12
    slope = float(np.polyfit(ns[1:][good], np.log(w2_cf[1:][good]), 1)[0])

    metrics = dict(
        params=dict(sigma_u=args.sigma_u, sigma_theta=sig_t, rho=args.rho, theta0=args.theta0),
        a_H=m["a_H"], a_P=m["a_P"], kappa=kappa, rho2=args.rho ** 2,
        L_H_gt_1=bool(abs(m["a_H"]) > 1.0),
        stationary_var=float(v_n[-1]), target_var=sig_t ** 2,
        exactness_w2=float(exactness),
        thm_rate_bound_holds=bound_ok,
        cf_decay_slope=slope, log_kappa=float(np.log(kappa)),
        w2_closed_form=w2_cf.tolist(), thm_bound=thm_bound.tolist(),
        mc_mean=mc_mean.tolist(), mc_std=mc_std.tolist(),
    )
    with open(os.path.join(RESDIR, "e_a.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.semilogy(ns, np.maximum(w2_cf, 1e-16), "o-", color="#1f4e79", lw=2, ms=4,
                label=r"$W_2(\mathrm{law}(\theta_n),\nu^*)$ (closed form)")
    ax.semilogy(ns, np.maximum(thm_bound, 1e-16), "--", color="#c00000", lw=2,
                label=r"Thm rate bound $\kappa^n W_2(\mu_0,\nu^*)$")
    ax.errorbar(ns, np.maximum(mc_mean, 1e-16), yerr=mc_std, fmt="s", color="#2e7d32",
                ms=3.5, capsize=2, alpha=0.8, label=r"MC estimate ($\pm$std, %d seeds)" % args.seeds)
    ax.set_xlabel("sweep $n$"); ax.set_ylabel(r"$W_2$ to stationary $\nu^*$")
    ax.set_title(r"E-a: geometric rate $\kappa=\rho^2=%.3f$  ($L_H=%.2f>1,\ L_P=%.2f$)"
                 % (kappa, m["a_H"], m["a_P"]))
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "num_Ea_rate.pdf")
    fig.savefig(out); plt.close(fig)

    print(f"[E-a] kappa=rho^2={kappa:.4f}  a_H={m['a_H']:.3f} (>1: {metrics['L_H_gt_1']})  a_P={m['a_P']:.3f}")
    print(f"[E-a] exactness W2(nu*,nu_dagger)={exactness:.2e}  Thm-rate bound holds: {bound_ok}")
    print(f"[E-a] closed-form decay slope={slope:.4f}  vs  log kappa={np.log(kappa):.4f}")
    print(f"[E-a] figure -> {out}")
    print(f"[E-a] metrics -> {os.path.join(RESDIR, 'e_a.json')}")


if __name__ == "__main__":
    main()
