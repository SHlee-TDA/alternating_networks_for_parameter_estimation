"""
OGTT sampled-sensitivity / identifiability experiments (restricted free-variable manifold), full ISS-HGP
vs constant-HGP.

The actual OGTT record is glucose at t in {0,30,60,90,120} (5 points). The generator draws G0,I0,S_I,sigma
and slaves the incretin initial states to (N5_0,N6_0)=SS(G0,sigma). Since G0=G(t_0) is observed, the
quantities a glucose-only record can determine are I0 and the parameters, i.e. the free-variable manifold
    xi=(I0,S_I,sigma) in R^3,     G(xi)=(G(t_0),...,G(t_4)) in R^5.
We study the *log-relative* sampled-sensitivity Jacobian
    J = D_xi G(xi) diag(xi)  in R^{5x3}     (i.e. columns are dG/d log xi_k),
a scale-invariant diagnostic in relative coordinates (distinct from the network training coordinate of
Appendix B.2). J is 5x3, so it has at most three singular values.

The constant-HGP model has the exact one-parameter scaling symmetry (log-relative tangent t=(+1,-1,+1)):
    I0->cI0, sigma->c sigma, S_I->S_I/c   (N5,N6 auto-scale via SS(G0,c sigma)=c SS(G0,sigma)),
which leaves glucose invariant. We quantify, with step-size-stable central differences:
  (1) full vs constant-HGP singular values of the 5x3 J (three values), column norms, condition number;
  (2) the sensitivity ||J t|| along the scaling tangent, and the alignment of the smallest right singular
      vector with t (=1 for constant HGP: its null direction IS the scaling tangent);
  (3) ||J t|| vs number of glucose samples (the symmetry-breaking signal);
  (4) a finite-scaling glucose-invariance check (exact symmetry <=> constant HGP);
  (5) the pointwise symmetry generator I dHGP/dI - S_I dHGP/dS_I (nonzero full / zero constant);
  (6) multi-point statistics of the third singular value and condition number over prior draws.

Usage: python experiments/det_meanlocus/ogtt_observability.py
Outputs: results/det_meanlocus/ogtt_observability.json  and  paper/simods/figures/exp/ogtt_hgp_ablation.pdf
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 14, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13})
from scipy.integrate import solve_ivp
import scipy.stats as st
from systems.ogtt_simul import OgttSimul, OGTTModel, SYS_PARAMS, ODE_PARAMS

OUTJSON = 'results/det_meanlocus/ogtt_observability.json'
OUTFIG = 'paper/simods/figures/exp/ogtt_hgp_ablation.pdf'

sysm = OgttSimul()
G0_NOM, I0_NOM, SI_NOM, SIG_NOM = 91.0, 6.0, 0.5, 0.5    # nominal G0 (mg/dL) and xi
xi0 = np.array([I0_NOM, SI_NOM, SIG_NOM])
TANG = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)         # log-relative scaling tangent
_orig = OGTTModel.calculate_HGP
# constant-HGP value: a SINGLE fixed scalar (nominal point), independent of S_I and I.
_HGP_CONST = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': SI_NOM, 'sigma': SIG_NOM}).calculate_HGP(I0_NOM)


def set_variant(const):
    OGTTModel.calculate_HGP = (lambda self, I: _HGP_CONST) if const else _orig


def glucose(t_eval, xi, G0, const):
    set_variant(const)
    m = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': xi[1], 'sigma': xi[2]})
    n5, n6 = m.find_steady_state_N(G0)                   # (N5,N6)=SS(G0,sigma), scales with sigma
    return solve_ivp(lambda t, y: sysm.ode_func(t, y, (xi[1], xi[2])), (0, 120), [G0, xi[0], n5, n6],
                     t_eval=t_eval, rtol=1e-10, atol=1e-10, method='LSODA').y[0]


def jac_logrel(t_obs, xi, G0, const, rel=1e-4, absol=1e-6):
    """J = D_xi G diag(xi): columns are dG/d log xi_k."""
    J = np.zeros((len(t_obs), 3))
    for k in range(3):
        h = rel * abs(xi[k]) + absol
        xp = xi.copy(); xp[k] += h
        xm = xi.copy(); xm[k] -= h
        J[:, k] = (glucose(t_obs, xp, G0, const) - glucose(t_obs, xm, G0, const)) / (2 * h) * abs(xi[k])
    return J


def diag(t_obs, xi, G0, const, rel=1e-4):
    J = jac_logrel(t_obs, xi, G0, const, rel=rel)
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    Jt = float(np.linalg.norm(J @ TANG))
    return dict(sv=sv, sigma_min=float(sv[-1]), sigma_max=float(sv[0]),
                cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
                col_norms=np.linalg.norm(J, axis=0).tolist(),
                Jt=Jt, Jt_rel=float(Jt / np.linalg.norm(J)),
                vmin_cos_tangent=float(abs(Vt[-1] @ TANG)))


def symmetry_generator(const, eps=1e-5):
    set_variant(const)
    m = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': SI_NOM, 'sigma': SIG_NOM})
    hI = eps * abs(I0_NOM) + 1e-8
    dI = (m.calculate_HGP(I0_NOM + hI) - m.calculate_HGP(I0_NOM - hI)) / (2 * hI)
    hs = eps * abs(SI_NOM) + 1e-8
    m.theta['si'] = SI_NOM + hs; hp = m.calculate_HGP(I0_NOM)
    m.theta['si'] = SI_NOM - hs; hm = m.calculate_HGP(I0_NOM)
    dS = (hp - hm) / (2 * hs)
    return float(I0_NOM * dI - SI_NOM * dS)


def main():
    t5 = np.array([0, 30, 60, 90, 120.0])
    res = {'nominal': {'G0': G0_NOM, 'I0': I0_NOM, 'S_I': SI_NOM, 'sigma': SIG_NOM},
           'coordinate': 'J = D_xi G diag(xi) (log-relative); glucose in mg/dL',
           'grid_5pt': t5.tolist(), 'scaling_tangent_logrel': TANG.tolist(),
           'rank_tolerance': '1e-3 * sigma_max'}
    for tag, ch in [('full', False), ('constant', True)]:
        by_step = {rel: diag(t5, xi0, G0_NOM, ch, rel=rel)['sigma_min'] for rel in (1e-3, 1e-4, 1e-5)}
        d = diag(t5, xi0, G0_NOM, ch)
        res[tag] = dict(singular_values=d['sv'].tolist(), sigma_min=d['sigma_min'], sigma_max=d['sigma_max'],
                        cond=d['cond'], col_norms=d['col_norms'], Jt=d['Jt'], Jt_rel=d['Jt_rel'],
                        vmin_cos_tangent=d['vmin_cos_tangent'], sigma_min_by_relstep=by_step)
        print(f"[{tag:8s}] sv={np.array2string(d['sv'],precision=4)} cond={d['cond']:.1f} "
              f"||Jt||={d['Jt']:.3e} |cos(vmin,t)|={d['vmin_cos_tangent']:.3f} colnorms={np.array2string(np.array(d['col_norms']),precision=2)}")

    # (3) scaling-tangent sensitivity vs sampling density
    sweep = {}
    for npts in [3, 5, 7, 9, 13, 25]:
        t = np.linspace(0, 120, npts)
        sf = diag(t, xi0, G0_NOM, False); sc = diag(t, xi0, G0_NOM, True)
        sweep[npts] = dict(full_Jt=sf['Jt'], const_Jt=sc['Jt'], full_sigmin=sf['sigma_min'], const_sigmin=sc['sigma_min'])
        print(f"  npts={npts:2d}: full ||Jt||={sf['Jt']:.2e} | const ||Jt||={sc['Jt']:.2e}")
    res['sampling_sweep'] = sweep

    res['symmetry_generator'] = dict(full=symmetry_generator(False), constant=symmetry_generator(True))
    print("symmetry generator: full=%.4g constant=%.4g" % (res['symmetry_generator']['full'], res['symmetry_generator']['constant']))

    # (4) finite-scaling invariance
    t25 = np.linspace(0, 120, 25); fs = {}
    for c in [1.25, 1.5, 2.0]:
        xic = np.array([I0_NOM * c, SI_NOM / c, SIG_NOM * c]); row = {}
        for tag, ch in [('full', False), ('constant', True)]:
            Gn = glucose(t25, xi0, G0_NOM, ch); Gc = glucose(t25, xic, G0_NOM, ch)
            row[tag] = float(np.max(np.abs(Gc - Gn)) / np.max(np.abs(Gn)))
        fs[f'c={c}'] = row
        print(f"  scaling c={c}: full={row['full']:.2e} const={row['constant']:.2e}")
    res['finite_scaling_invariance'] = fs

    # (6) multi-point statistics over prior draws
    rng = np.random.default_rng(0); s3 = []; kap = []; cvt = []
    for _ in range(40):
        g0 = float(st.lognorm.rvs(0.286, 60.07, 29.81, random_state=rng))
        i0 = abs(float(st.lognorm.rvs(0.596, -0.013, 6.495, random_state=rng))) + 1e-3
        si = abs(float(st.lognorm.rvs(0.527, -0.105, 0.506, random_state=rng))) + 1e-3
        sg = abs(float(st.lognorm.rvs(0.593, -0.053, 0.543, random_state=rng))) + 1e-3
        d = diag(t5, np.array([i0, si, sg]), g0, False)
        s3.append(d['sigma_min']); kap.append(d['cond']); cvt.append(d['vmin_cos_tangent'])
    res['prior_stats'] = dict(n=40,
        sigma3_median=float(np.median(s3)), sigma3_iqr=[float(np.percentile(s3, 25)), float(np.percentile(s3, 75))],
        cond_median=float(np.median(kap)), cond_iqr=[float(np.percentile(kap, 25)), float(np.percentile(kap, 75))])
    print("prior draws: sigma3 median=%.3g IQR=[%.3g,%.3g] cond median=%.3g IQR=[%.3g,%.3g]" %
          (res['prior_stats']['sigma3_median'], *res['prior_stats']['sigma3_iqr'],
           res['prior_stats']['cond_median'], *res['prior_stats']['cond_iqr']))

    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    json.dump(res, open(OUTJSON, 'w'), indent=2)

    # ---- figure (3 panels) ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    svf = np.array(res['full']['singular_values']); svc = np.array(res['constant']['singular_values'])
    idx = np.arange(1, 4)
    ax[0].semilogy(idx, svf, 'o-', color='#1f77b4', lw=2, ms=10, label='full HGP$(S_I,I)$')
    ax[0].semilogy(idx, np.maximum(svc, 1e-18), 's--', color='#d62728', lw=2, ms=10, label='constant HGP')
    ax[0].set_xlabel('index $i$'); ax[0].set_ylabel(r'singular value $\varsigma_i$')
    ax[0].set_title('(a) Restricted spectrum\n$5{\\times}3$ Jacobian, $\\xi=(I_0,S_I,\\sigma)$')
    ax[0].set_xticks(idx); ax[0].grid(True, which='both', alpha=0.3); ax[0].legend()

    npts = sorted(sweep.keys())
    fjt = [sweep[n]['full_Jt'] for n in npts]; cjt = [max(sweep[n]['const_Jt'], 1e-18) for n in npts]
    ax[1].semilogy(npts, fjt, 'o-', color='#1f77b4', lw=2, label=r'full HGP')
    ax[1].semilogy(npts, cjt, 's--', color='#d62728', lw=2, label=r'constant HGP')
    ax[1].axvline(5, color='0.5', lw=0.8, ls=':'); ax[1].text(5.2, min(cjt) * 2, '5-pt grid', fontsize=12, color='0.4')
    ax[1].set_xlabel('number of glucose samples'); ax[1].set_ylabel(r'sensitivity along scaling tangent $\|J\hat t\|$')
    ax[1].set_title('(b) Symmetry-breaking signal\nvs sampling density')
    ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(loc='center right')

    cs = np.array([1.25, 1.5, 2.0])
    ff = np.array([fs[f'c={c}']['full'] for c in cs]); cc = np.array([fs[f'c={c}']['constant'] for c in cs])
    ax[2].semilogy(cs, ff, 'o-', color='#1f77b4', lw=2, label='full HGP')
    ax[2].semilogy(cs, cc, 's--', color='#d62728', lw=2, label='constant HGP')
    ax[2].set_xlabel('scaling factor $c$'); ax[2].set_ylabel(r'$\max_t|\Delta G|/\max_t|G|$')
    ax[2].set_title('(c) Glucose invariance under the\nscaling symmetry')
    ax[2].grid(True, which='both', alpha=0.3); ax[2].legend(loc='center right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTFIG), exist_ok=True)
    plt.savefig(OUTFIG, dpi=200, bbox_inches='tight')
    print("saved", OUTFIG)
    set_variant(False)


if __name__ == '__main__':
    main()
