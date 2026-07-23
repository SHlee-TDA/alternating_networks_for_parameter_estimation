"""
Regenerate the OGTT ill-conditioning / sensitivity figure (paper Fig. 4) at PDF quality.
(Left)  singular-value spectrum of the restricted sampled-sensitivity Jacobian at the 5-point grid,
        J = dG(t_i)/dxi with xi = (I0, S_I, sigma) in R^3 (G0 observed; N5,N6 slaved to (G0,sigma)).
        J is 5x3, so it has three singular values -- no six-dimensional ambient claim.
(Right) temporal sensitivities dG(t)/dS_I and dG(t)/dsigma over the OGTT window.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 13, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13})
from scipy.integrate import solve_ivp
from systems.ogtt_simul import OgttSimul, OGTTModel, SYS_PARAMS, ODE_PARAMS

sysm = OgttSimul()
si0, sig0 = 0.5, 0.5                       # nominal parameters (typical prior values)
G0, I0 = 91.0, 6.0                         # nominal fasting glucose / insulin


def glucose(t_eval, xi):
    """xi=(I0,S_I,sigma); G0 fixed; (N5,N6)=SS(G0,sigma)."""
    I0v, si, sig = float(xi[0]), float(xi[1]), float(xi[2])
    mdl = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': si, 'sigma': sig})
    n5, n6 = mdl.find_steady_state_N(G0)
    sol = solve_ivp(lambda t, y: sysm.ode_func(t, y, (si, sig)), (0, 120), [G0, I0v, n5, n6],
                    t_eval=t_eval, rtol=1e-9, atol=1e-9, method='LSODA')
    return sol.y[0]


xi0 = np.array([I0, si0, sig0])
col_scale = np.abs(xi0)

# ---- Right panel: temporal sensitivities dG/dS_I, dG/dsigma ----
t_dense = np.linspace(0, 120, 241)
G_nom = glucose(t_dense, xi0)
eps = 1e-3
dG_dSI = (glucose(t_dense, [I0, si0 + eps, sig0]) - G_nom) / eps
dG_dsig = (glucose(t_dense, [I0, si0, sig0 + eps]) - G_nom) / eps

# ---- Left panel: restricted 5x3 sampled-sensitivity SVD at the actual 5-point grid ----
t_obs = np.array([0, 30, 60, 90, 120.0])
G_obs = glucose(t_obs, xi0)
cols = []
for k in range(3):
    h = abs(xi0[k]) * 1e-4 + 1e-6
    xp = xi0.copy(); xp[k] += h
    xm = xi0.copy(); xm[k] -= h
    cols.append((glucose(t_obs, xp) - glucose(t_obs, xm)) / (2 * h) * col_scale[k])
J = np.column_stack(cols)                 # (5 x 3)
svals = np.linalg.svd(J, compute_uv=False)
print("restricted 5x3 singular values:", np.array2string(svals, precision=4), "cond=%.2e" % (svals[0] / svals[-1]))

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].semilogy(range(1, len(svals) + 1), svals, 'o-', color='#333333', lw=2, ms=10)
ax[0].set_xlabel('index $i$'); ax[0].set_ylabel(r'singular value $\varsigma_i$ (log scale)')
ax[0].set_title('Restricted sampled-sensitivity spectrum\n($5{\\times}3$ Jacobian, $\\xi=(I_0,S_I,\\sigma)$)')
ax[0].set_xticks(range(1, len(svals) + 1)); ax[0].grid(True, which='both', alpha=0.3)
ax[1].plot(t_dense, dG_dSI, color='#1f77b4', lw=2.4, label=r'$\partial G/\partial S_I$')
ax[1].plot(t_dense, dG_dsig, color='#d62728', lw=2.4, ls='--', label=r'$\partial G/\partial \sigma$')
ax[1].axhline(0, color='grey', lw=0.6)
ax[1].set_xlabel('time $t$ (min)'); ax[1].set_ylabel(r'glucose sensitivity')
ax[1].set_title('Temporal parameter sensitivity of glucose')
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
out = 'paper/simods/figures/ogtt_illcond.pdf'
plt.savefig(out, dpi=200); print("saved", out)
