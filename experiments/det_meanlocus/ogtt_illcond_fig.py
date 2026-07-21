"""
Regenerate the OGTT ill-conditioning / sensitivity figure (paper Fig. 4) at PDF quality.
(Left)  singular-value spectrum of the observability Jacobian J = dG(t_i)/dz(0),
        z(0) = [G0, I0, N5, N6, S_I, sigma]  (relative/normalized sensitivities).
(Right) temporal sensitivities dG(t)/dS_I and dG(t)/dsigma.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from systems.ogtt_simul import OgttSimul, OGTTModel, SYS_PARAMS, ODE_PARAMS

sysm = OgttSimul()
si0, sig0 = 0.5, 0.5                       # nominal parameters (typical prior values)
G0, I0 = 91.0, 6.0                         # nominal fasting glucose / insulin
mdl = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': si0, 'sigma': sig0})
n5_0, n6_0 = mdl.find_steady_state_N(G0)
z0 = np.array([G0, I0, n5_0, n6_0])        # state initial conditions

def glucose(t_eval, y0, si, sig):
    sol = solve_ivp(lambda t, y: sysm.ode_func(t, y, (si, sig)), (0, 120), list(y0),
                    t_eval=t_eval, rtol=1e-8, atol=1e-8, method='LSODA')
    return sol.y[0]

# ---- Right panel: temporal sensitivities dG/dS_I, dG/dsigma ----
t_dense = np.linspace(0, 120, 241)
G_nom = glucose(t_dense, z0, si0, sig0)
eps = 1e-3
dG_dSI  = (glucose(t_dense, z0, si0 + eps, sig0) - G_nom) / eps
dG_dsig = (glucose(t_dense, z0, si0, sig0 + eps) - G_nom) / eps

# ---- Left panel: observability Jacobian SVD ----
t_obs = np.linspace(0, 120, 13)           # >= 6 sampling points
G_obs = glucose(t_obs, z0, si0, sig0)
cols = []
# perturb each augmented-state coordinate; use relative (normalized) sensitivity z_k * dG/dz_k
for k, base in enumerate([G0, I0, n5_0, n6_0]):
    dz = z0.copy(); h = abs(base) * 1e-3 + 1e-6; dz[k] += h
    cols.append((glucose(t_obs, dz, si0, sig0) - G_obs) / h * base)
for k, (pval, pert) in enumerate([(si0, (si0*1e-3+1e-6, 0.0)), (sig0, (0.0, sig0*1e-3+1e-6))]):
    hs, hg = pert
    cols.append((glucose(t_obs, z0, si0+hs, sig0+hg) - G_obs) / (hs+hg) * pval)
J = np.column_stack(cols)                 # (13 x 6)
svals = np.linalg.svd(J, compute_uv=False)
print("singular values:", np.array2string(svals, precision=4))

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogy(range(1, len(svals)+1), svals, 'o-', color='#333333', lw=2, ms=8)
ax[0].set_xlabel('index $i$'); ax[0].set_ylabel(r'singular value $\varsigma_i$ (log scale)')
ax[0].set_title('Observability Jacobian spectrum\n(decay $\\Rightarrow$ practical ill-conditioning)', fontsize=10)
ax[0].set_xticks(range(1, len(svals)+1)); ax[0].grid(True, which='both', alpha=0.3)
ax[1].plot(t_dense, dG_dSI,  color='#1f77b4', lw=2.2, label=r'$\partial G/\partial S_I$')
ax[1].plot(t_dense, dG_dsig, color='#d62728', lw=2.2, ls='--', label=r'$\partial G/\partial \sigma$')
ax[1].axhline(0, color='grey', lw=0.6)
ax[1].set_xlabel('time $t$ (min)'); ax[1].set_ylabel(r'glucose sensitivity')
ax[1].set_title('Temporal parameter sensitivity of glucose', fontsize=10)
ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
out = 'paper/simods/figures/ogtt_illcond.pdf'
plt.savefig(out, dpi=200); print("saved", out)
