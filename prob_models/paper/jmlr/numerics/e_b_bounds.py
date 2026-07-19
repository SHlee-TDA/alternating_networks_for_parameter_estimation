"""E-b -- Controlled incompatibility.  Validates Thm:B (bound) + Thm:incbound.

Start from E-a's compatible Gaussian; keep Q_H = p_H^dagger (eps_H = 0) and perturb the
P-slope: Q_P^delta(theta|u) = N((a_P+delta) u, s_P^2). For delta != 0 the pair is
incompatible, but the chain stays Gaussian:
  kappa_delta = |a_H (a_P+delta)|,   v*_delta = Var(zeta_delta)/(1-kappa_delta^2),
  Var(zeta_delta) = (a_P+delta)^2 s_H^2 + s_P^2,   nu*_delta = N(0, v*_delta).
Closed forms (box radius U, so eps_P = |delta| U):
  eps_P               = |delta| U
  actual target error = |sqrt(v*_delta) - sigma_theta|          [ = W2(nu*_delta, nu^dagger) ]
  Thm:B bound         = eps_P / (1 - kappa_delta)
  eps_inc             = Bures W2 between the two centered joints Pi^-> and Pi^<-
  Thm:incbound bound  = C_P eps_P,  C_P = 1 + A(1+L_H^dagger)/(1-kappa_delta), A = 2+L_P^dagger,
                        with L_H^dagger=|a_H|, L_P^dagger=|a_P|, L_P=|a_P+delta|  (eps_H=0).
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _w2 import bures_w2, sliced_w2

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "figures"))
RESDIR = os.path.join(HERE, "results")


def base_model(sigma_u, sigma_theta, rho):
    return dict(a_H=rho * sigma_u / sigma_theta, a_P=rho * sigma_theta / sigma_u,
                s_H2=sigma_u ** 2 * (1 - rho ** 2), s_P2=sigma_theta ** 2 * (1 - rho ** 2),
                sigma_theta=sigma_theta)


def perturbed(m, delta):
    a_H, a_P, s_H2, s_P2 = m["a_H"], m["a_P"], m["s_H2"], m["s_P2"]
    aPd = a_P + delta
    kappa = abs(a_H * aPd)
    var_zeta = aPd ** 2 * s_H2 + s_P2
    v_star = var_zeta / (1.0 - kappa ** 2)
    V_u = a_H ** 2 * v_star + s_H2                       # stationary u-variance
    # centered joints on (theta, u):
    Sig_fwd = np.array([[v_star, a_H * v_star], [a_H * v_star, V_u]])
    Sig_bwd = np.array([[aPd ** 2 * V_u + s_P2, aPd * V_u], [aPd * V_u, V_u]])
    return dict(aPd=aPd, kappa=kappa, v_star=v_star, V_u=V_u, Sig_fwd=Sig_fwd, Sig_bwd=Sig_bwd)


def C_constants(m, delta, kappa):
    """Explicit Thm:incbound constants (eps_H = 0 here, so bound = C_P eps_P)."""
    L_H_d, L_P_d = abs(m["a_H"]), abs(m["a_P"])          # true-conditional W2-Lipschitz
    A = 2.0 + L_P_d
    C_P = 1.0 + A * (1.0 + L_H_d) / (1.0 - kappa)
    L_P = abs(m["a_P"] + delta)                          # learned P-slope
    C_H = A + A * (1.0 + L_H_d) * L_P / (1.0 - kappa)
    return C_H, C_P, max(C_H, C_P)


def mc_eps_inc(m, pert, delta, n_chains, n_steps, rng):
    """Single-run sliced-W2 estimate of eps_inc = W2(Pi^->, Pi^<-) from a trained-model
    run. The estimator's finite-sample resolution FLOOR is obtained by running the same
    estimator on the compatible model (delta=0), where the true value is 0."""
    a_H, aPd = m["a_H"], pert["aPd"]
    s_H, s_P = np.sqrt(m["s_H2"]), np.sqrt(m["s_P2"])
    th = rng.normal(size=n_chains) * np.sqrt(pert["v_star"])
    u = None
    for n in range(n_steps):
        u = a_H * th + s_H * rng.normal(size=n_chains)
        th = aPd * u + s_P * rng.normal(size=n_chains)
    # stationary backward joint pi* = Pi^<- : the pair (theta_n, u_n)
    pi_bwd = np.stack([th, u], axis=1)
    # forward joint Pi^-> : (theta, fresh u' ~ Q_H(.|theta))
    u_fwd = a_H * th + s_H * rng.normal(size=n_chains)
    pi_fwd = np.stack([th, u_fwd], axis=1)
    sliced = sliced_w2(pi_fwd, pi_bwd, n_proj=200, seed=int(rng.integers(1 << 30)))
    # Gaussian plug-in: empirical covariances -> Bures (converges to true W2, no floor)
    emp = bures_w2(pi_fwd.mean(0), np.cov(pi_fwd.T), pi_bwd.mean(0), np.cov(pi_bwd.T))
    return sliced, emp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma_u", type=float, default=2.0)
    ap.add_argument("--sigma_theta", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--delta_max", type=float, default=0.18)
    ap.add_argument("--n_delta", type=int, default=19)
    ap.add_argument("--U_sd", type=float, default=4.0, help="box radius in stationary u-sd")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_chains", type=int, default=20000)
    ap.add_argument("--n_steps", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(RESDIR, exist_ok=True)

    m = base_model(args.sigma_u, args.sigma_theta, args.rho)
    sig_t = m["sigma_theta"]
    deltas = np.linspace(0.0, args.delta_max, args.n_delta)

    # fixed box radius U (constant across the sweep): cover support at delta_max
    p_max = perturbed(m, args.delta_max)
    U = args.U_sd * np.sqrt(p_max["V_u"])

    rows = []
    for d in deltas:
        p = perturbed(m, d)
        kap = p["kappa"]
        eps_P = abs(d) * U
        target_err = abs(np.sqrt(p["v_star"]) - sig_t)
        thmB = eps_P / (1.0 - kap)
        eps_inc = bures_w2([0, 0], p["Sig_fwd"], [0, 0], p["Sig_bwd"])
        C_H, C_P, C = C_constants(m, d, kap)
        incbound = C_P * eps_P                       # = C_H eps_H + C_P eps_P, eps_H=0
        rows.append(dict(delta=float(d), kappa=float(kap), eps_P=float(eps_P),
                         target_err=float(target_err), thmB_bound=float(thmB),
                         eps_inc=float(eps_inc), incbound=float(incbound),
                         C_P=float(C_P), C=float(C)))

    # ---- MC estimate of eps_inc (single-run diagnostic) -----------------------
    # includes delta=0 (compatible), whose estimate is the estimator's resolution floor.
    mc_deltas = deltas[:: max(1, args.n_delta // 6)]
    if mc_deltas[0] != 0.0:
        mc_deltas = np.concatenate([[0.0], mc_deltas])
    mc = {}
    for d in mc_deltas:
        p = perturbed(m, d)
        pairs = [mc_eps_inc(m, p, d, args.n_chains, args.n_steps,
                            np.random.default_rng(1000 + s)) for s in range(args.seeds)]
        sl = np.array([x[0] for x in pairs]); em = np.array([x[1] for x in pairs])
        mc[float(d)] = dict(sliced=float(sl.mean()), sliced_std=float(sl.std()),
                            bures=float(em.mean()), bures_std=float(em.std()))
    mc_floor = mc[0.0]["sliced"]   # resolution floor of the sliced-W2 estimator

    da = np.array([r["delta"] for r in rows])
    # tol absorbs the Bures numerical zero at delta=0 (eps_inc, eps_P both -> 0 there)
    thmB_ok = all(r["target_err"] <= r["thmB_bound"] + 1e-6 for r in rows)
    inc_ok = all(r["eps_inc"] <= r["incbound"] + 1e-6 for r in rows)
    # tracking: eps_inc / eps_P ratio (finite, bounded) for delta>0
    ratios = [r["eps_inc"] / r["eps_P"] for r in rows if r["eps_P"] > 0]

    metrics = dict(
        params=dict(sigma_u=args.sigma_u, sigma_theta=sig_t, rho=args.rho,
                    U=float(U), U_sd=args.U_sd, delta_max=args.delta_max),
        a_H=m["a_H"], a_P=m["a_P"], kappa0=abs(m["a_H"] * m["a_P"]),
        thmB_bound_holds=thmB_ok, incbound_holds=inc_ok,
        eps_inc_over_eps_P_range=[float(min(ratios)), float(max(ratios))],
        mc_floor=float(mc_floor), rows=rows, mc_eps_inc=mc,
    )
    with open(os.path.join(RESDIR, "e_b.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- figure: two panels ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    target = np.array([r["target_err"] for r in rows])
    thmB = np.array([r["thmB_bound"] for r in rows])
    einc = np.array([r["eps_inc"] for r in rows])
    incb = np.array([r["incbound"] for r in rows])

    pos = da > 0   # skip delta=0 for bound curves (bound=0 -> -inf on log axis)
    ax = axes[0]
    ax.plot(da[pos], thmB[pos], "--", color="#c00000", lw=2, label=r"Thm B bound $\varepsilon_P/(1-\kappa_\delta)$")
    ax.plot(da, target, "o-", color="#1f4e79", lw=2, ms=4, label=r"actual $W_2(\nu^*_\delta,\nu^\dagger)$")
    ax.set_xlabel(r"incompatibility dial $\delta$"); ax.set_ylabel(r"$W_2$")
    ax.set_title("Panel A: steering bound (Thm B)"); ax.set_yscale("log")
    ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(da[pos], incb[pos], "--", color="#c00000", lw=2, label=r"Thm incbound $C_P\,\varepsilon_P$")
    ax.plot(da[pos], einc[pos], "o-", color="#1f4e79", lw=2, ms=4, label=r"closed-form $\varepsilon_{\mathrm{inc}}$")
    xs = np.array(sorted(mc.keys()))
    b_mean = np.array([mc[x]["bures"] for x in xs]); b_std = np.array([mc[x]["bures_std"] for x in xs])
    s_mean = np.array([mc[x]["sliced"] for x in xs])
    ax.errorbar(xs, b_mean, yerr=b_std, fmt="s", color="#2e7d32", ms=4, capsize=2,
                label=r"MC empirical-cov Bures (Gaussian plug-in)")
    ax.plot(xs, s_mean, "^", color="#7030a0", ms=5, alpha=0.85,
            label=r"MC sliced-$W_2$ (general; lower bound)")
    ax.axhline(mc_floor, ls=":", color="#777777", lw=1.5,
               label="sliced-$W_2$ floor ($\\delta{=}0$)")
    ax.set_xlabel(r"incompatibility dial $\delta$"); ax.set_ylabel(r"$W_2$")
    ax.set_title("Panel B: self-consistency bound (Thm incbound)"); ax.set_yscale("log")
    ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(r"E-b: controlled incompatibility ($\varepsilon_H=0$, dial $\delta$; both bounds hold, $\varepsilon_{\mathrm{inc}}\!\to\!0$ with $\delta$)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "num_Eb_bounds.pdf")
    fig.savefig(out); plt.close(fig)

    print(f"[E-b] box radius U={U:.2f}  kappa range=[{da[0]:.2f}->{rows[0]['kappa']:.3f}, {rows[-1]['kappa']:.3f}]")
    print(f"[E-b] Thm B bound holds for all delta: {thmB_ok}")
    print(f"[E-b] Thm incbound holds for all delta: {inc_ok}")
    print(f"[E-b] eps_inc/eps_P in [{min(ratios):.3f}, {max(ratios):.3f}] (bounded => tracking)")
    print(f"[E-b] eps_inc(delta=0)={rows[0]['eps_inc']:.2e} (compatible; sliced-W2 floor={mc_floor:.4f})")
    d0 = sorted(mc.keys())[-1]
    cf0 = float(np.interp(d0, da, np.array([r['eps_inc'] for r in rows])))
    print(f"[E-b] at delta={d0:.3f}: CF={cf0:.4f}  MC-Bures={mc[d0]['bures']:.4f}  MC-sliced={mc[d0]['sliced']:.4f}")
    print(f"[E-b] figure -> {out}")


if __name__ == "__main__":
    main()
