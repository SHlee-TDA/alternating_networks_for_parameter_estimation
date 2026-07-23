"""
Figures + LaTeX table for the round-5 noise stress test.
Reads results/det_meanlocus/noise_stress_test.json and emits:
  paper/simods/figures/exp/noise_ogtt_2x2.pdf     (main: OGTT, 4 panels x 3 methods)
  paper/simods/figures/exp/noise_sir_lv_2x2.pdf   (appendix: SIR + LV rows)
  paper/simods/figures/exp/noise_stability_plane.pdf (clean error vs sensitivity)
  results/det_meanlocus/noise_table.tex           (main table fragment)

Robust choices (pre-registered): absolute error uses the median log-parameter error e_log (mean NRMSE is
outlier-driven for the unbounded direct head under exp-denorm); failure = out-of-support / non-finite /
positivity rate; displacement = median log-coordinate shift from the clean prediction.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14,
                     'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

FIGDIR = 'paper/simods/figures/exp'
J = json.load(open('results/det_meanlocus/noise_stress_test.json'))
SUM, META = J['summary'], J['meta']
ALPHAS = J['protocol']['alphas']
AX = np.array(ALPHAS)

STYLE = {'iterative': dict(color='#1f77b4', marker='o', label='Iterative'),
         'direct': dict(color='#d62728', marker='s', label='Direct'),
         'prior_mean': dict(color='#7f7f7f', marker='^', label='Prior mean', ls='--')}
METHS = ['iterative', 'direct', 'prior_mean']


def series(sysn, meth, key, sub=0):
    out = []
    for a in ALPHAS:
        e = SUM[sysn][meth][str(a)]
        v = e[key]
        out.append(v[sub] if isinstance(v, list) else v)
    return np.array(out, float)


def panel(ax, sysn, key, ylabel, logy=False, sub=0, delta=False):
    for m in METHS:
        y = series(sysn, m, key, sub)
        if delta:
            y = y - y[0]
        st = STYLE[m]
        ax.plot(AX, y, marker=st['marker'], color=st['color'], ls=st.get('ls', '-'),
                label=st['label'], ms=6, lw=1.8)
    ax.set_xlabel(r'noise level $\alpha$')
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.grid(alpha=0.3)


def fig_2x2(sysn, path, title):
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    panel(axs[0, 0], sysn, 'elog_median', r'absolute error  $\tilde e_{\log}$')
    panel(axs[0, 1], sysn, 'elog_median', r'degradation  $\Delta \tilde e_{\log}$', delta=True)
    panel(axs[1, 0], sysn, 'displacement_median', r'prediction displacement $D(\alpha)$')
    panel(axs[1, 1], sysn, 'fail_any', r'failure rate')
    axs[0, 0].legend(frameon=False)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGDIR, path))
    plt.close(fig)
    print('wrote', path)


def fig_two_rows(sysA, sysB, path, titles):
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for row, sysn in enumerate([sysA, sysB]):
        panel(axs[row, 0], sysn, 'elog_median', r'absolute error $\tilde e_{\log}$')
        panel(axs[row, 1], sysn, 'elog_median', r'degradation $\Delta\tilde e_{\log}$', delta=True)
        panel(axs[row, 2], sysn, 'displacement_median', r'displacement $D(\alpha)$')
        panel(axs[row, 3], sysn, 'fail_any', r'failure rate')
        axs[row, 0].set_ylabel(titles[row] + '\n' + r'absolute error $\tilde e_{\log}$')
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, path))
    plt.close(fig)
    print('wrote', path)


def fig_stability_plane(path):
    """Clean absolute error (x) vs noise sensitivity slope (y). Lower-left = ideal."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sysmark = {'sir': 'o', 'ogtt_simul': 's', 'lotka_volterra': 'D'}
    sysname = {'sir': 'SIR', 'ogtt_simul': 'OGTT', 'lotka_volterra': 'LV'}
    for sysn in SUM:
        for m in METHS:
            clean = SUM[sysn][m]['0.0']['elog_median'][0]
            disp = SUM[sysn][m]['0.1']['displacement_median'][0]   # sensitivity at high noise
            ax.scatter(clean, disp, marker=sysmark[sysn], color=STYLE[m]['color'], s=90,
                       edgecolor='k', linewidth=0.5, zorder=3)
    # legends
    from matplotlib.lines import Line2D
    meth_h = [Line2D([0], [0], marker='o', color=STYLE[m]['color'], ls='', ms=9, label=STYLE[m]['label'])
              for m in METHS]
    sys_h = [Line2D([0], [0], marker=sysmark[s], color='k', ls='', ms=9, mfc='none', label=sysname[s])
             for s in SUM]
    l1 = ax.legend(handles=meth_h, loc='upper right', frameon=False, title='estimator')
    ax.add_artist(l1)
    ax.legend(handles=sys_h, loc='lower right', frameon=False, title='system')
    ax.set_xlabel(r'clean absolute error $\tilde e_{\log}(\alpha{=}0)$')
    ax.set_ylabel(r'noise sensitivity $D(\alpha{=}0.1)$')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, path))
    plt.close(fig)
    print('wrote', path)


def fmt(x, nd=3):
    if not np.isfinite(x):
        return '--'
    ax = abs(x)
    if ax != 0 and (ax >= 1e4 or ax < 1e-3):
        return '$%.1f{\\times}10^{%d}$' % tuple(_mant_exp(x))
    return ('%.' + str(nd) + 'f') % x


def _mant_exp(x):
    e = int(np.floor(np.log10(abs(x))))
    return x / 10 ** e, e


def make_table():
    rows = []
    levels = ['0.0', '0.02', '0.1']
    lname = {'0.0': 'clean', '0.02': r'$\alpha{=}0.02$', '0.1': r'$\alpha{=}0.10$'}
    sysname = {'sir': 'SIR', 'ogtt_simul': 'OGTT', 'lotka_volterra': 'LV'}
    mname = {'iterative': 'Iterative', 'direct': 'Direct', 'prior_mean': 'Prior mean'}
    for sysn in ['sir', 'ogtt_simul', 'lotka_volterra']:
        for mi, m in enumerate(METHS):
            for li, lv in enumerate(levels):
                e = SUM[sysn][m][lv]
                nrmse = e['nrmse'][0]
                elog = e['elog_median'][0]
                disp = e['displacement_median'][0]
                if abs(disp) < 1e-6:            # clean vs clean is definitionally zero
                    disp = 0.0
                fail = e['fail_any'][0]
                key = e['prod_abserr'] if 'prod_abserr' in e else float('nan')
                sysc = sysname[sysn] if (mi == 0 and li == 0) else ''
                mc = mname[m] if li == 0 else ''
                rows.append('%s & %s & %s & %s & %s & %s & %s & %.3f \\\\' % (
                    sysc, mc, lname[lv], fmt(nrmse), fmt(elog), fmt(key), fmt(disp), fail))
            if not (sysn == 'lotka_volterra' and m == 'prior_mean'):
                rows.append('\\cmidrule(l){2-8}' if m != 'prior_mean' else '\\midrule')
    body = '\n'.join(rows)
    tex = r"""% Auto-generated by experiments/det_meanlocus/noise_report.py
\begin{table}[t]
\centering
\caption{Test-time observation-perturbation stress test (noiseless training, noisy test; single training
seed, $30$ noise replicates). Absolute error $\tilde e_{\log}$ is the median log-parameter $\ell_2$ error;
NRMSE is the mean per-coordinate RMSE normalized by the training-prior std (outlier-sensitive for the
unbounded direct head under $\exp$-denormalization, hence shown alongside the failure rate). ``Key error''
is the OGTT product $|\,\widehat{S_I\sigma}-S_I\sigma\,|$; displacement $D$ is the median log-coordinate shift
from the clean prediction; failure = out-of-support / non-finite / positivity rate.}
\label{tab:noise}
\small
\begin{tabular}{lllrrrrr}
\toprule
System & Estimator & Noise & NRMSE & $\tilde e_{\log}$ & Key err. & $D(\alpha)$ & Fail \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open('results/det_meanlocus/noise_table.tex', 'w').write(tex)
    print('wrote results/det_meanlocus/noise_table.tex')


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig_2x2('ogtt_simul', 'noise_ogtt_2x2.pdf', 'OGTT (weakly identifiable): test-time noise response')
    fig_two_rows('sir', 'lotka_volterra', 'noise_sir_lv_2x2.pdf', ['SIR', 'Lotka--Volterra'])
    fig_stability_plane('noise_stability_plane.pdf')
    make_table()


if __name__ == '__main__':
    main()
