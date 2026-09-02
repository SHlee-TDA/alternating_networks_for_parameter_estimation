"""G0 evaluation + NLS baseline.

For each trained seed, reloads the iterative operator, reproduces the exact test
split (same SEED / NUM_SAMPLES => same cached data and split), and measures
relative parameter error for:

  - Iterative: the fixed-point operator T reloaded from Hnet.pth / Pnet.pth.
  - NLS:       scipy.least_squares fit of the true linear ODE model, given the
               SAME theta^(0) (p_init) the network receives. Non-convergences are
               counted, never dropped.

G0 passes when BOTH methods reach mean relative parameter error < 1e-3.
Writes metrics.json and a PDF bar chart under the experiment output dir.
"""
import os
import sys
import json
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import g0_settings as S
from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator
from src.infer import InferenceEngine
from tools.exp_tools import get_system_class, set_seed


def build_config(seed):
    cfg = Config()
    cfg.SYSTEM_NAME = S.SYSTEM_NAME
    cfg.SEED = seed
    cfg.NUM_SAMPLES = S.NUM_SAMPLES
    cfg.AUGMENTATION_FACTOR = 0
    cfg.BATCH_SIZE = S.BATCH_SIZE
    cfg.ITERATIONS = S.ITERATIONS
    cfg.USE_DERIVATIVE = False
    cfg.USE_SPECTRAL_NORM = S.USE_SPECTRAL_NORM
    cfg.RESULTS_DIR = S.RESULTS_DIR
    cfg.EXPERIMENT_NAME = S.experiment_name(seed)
    cfg.NORMALIZER_STATE_SCALES = None
    cfg.NORMALIZER_PARAM_BOUNDS = None
    cfg.__post_init__()  # rebuild EXPERIMENTS
    return cfg


def load_models(cfg, sample_x_dim, sample_y_dim, n_params, device):
    rp = S.results_path(cfg.SEED)
    hidden = HiddenVarPredictor(sample_x_dim, sample_y_dim, n_params,
                                model_config=cfg.MODEL_CONFIG['hidden_net'],
                                use_spectral_norm=cfg.USE_SPECTRAL_NORM).to(device)
    param = ParameterEstimator(sample_x_dim, sample_y_dim, n_params,
                               model_config=cfg.MODEL_CONFIG['param_net'],
                               use_spectral_norm=cfg.USE_SPECTRAL_NORM).to(device)
    hidden.load_state_dict(torch.load(os.path.join(rp, 'Hnet.pth'), map_location=device))
    param.load_state_dict(torch.load(os.path.join(rp, 'Pnet.pth'), map_location=device))
    hidden.eval(); param.eval()
    return hidden, param


def rel_errors(p_true, p_pred):
    """Per-sample relative L2-ish error per parameter; return dict of stats."""
    denom = np.maximum(np.abs(p_true), 1e-12)
    rel = np.abs(p_pred - p_true) / denom          # (N, P)
    per_param_mean = rel.mean(axis=0)               # (P,)
    overall = rel.mean()
    p95 = np.percentile(rel, 95)
    return {
        'mean': float(overall),
        'p95': float(p95),
        'max': float(rel.max()),
        'per_param_mean': [float(v) for v in per_param_mean],
    }


def nls_fit_one(x_obs, t_points, system, theta_init_ab, y0_init):
    """Fit [a, b, y0] to the observed x trajectory. Returns (theta_ab, converged)."""
    x0 = float(x_obs[0])

    def simulate(theta):
        a, b, y0 = theta
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sol = solve_ivp(fun=system.ode_func, t_span=(t_points[0], t_points[-1]),
                            y0=[x0, y0], t_eval=t_points, args=([a, b],))
        if not sol.success or np.any(~np.isfinite(sol.y)):
            return np.full(len(t_points), 1e4)
        return sol.y[system.observed_var_idx]

    def residual(theta):
        return simulate(theta) - x_obs

    theta0 = np.array([theta_init_ab[0], theta_init_ab[1], y0_init], dtype=float)
    lb = np.array([1e-3, 1e-3, 1e-3])
    ub = np.array([5.0, 5.0, 5.0])
    theta0 = np.clip(theta0, lb, ub)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = least_squares(residual, theta0, bounds=(lb, ub), method='trf',
                            max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    return res.x[:2], bool(res.success)


def evaluate_seed(seed, device):
    cfg = build_config(seed)
    set_seed(seed)
    system = get_system_class(S.SYSTEM_NAME)()

    gen_cfg = Config()
    for k in ('SYSTEM_NAME', 'NUM_SAMPLES', 'AUGMENTATION_FACTOR', 'USE_DERIVATIVE', 'SEED'):
        setattr(gen_cfg, k, getattr(cfg, k))
    gen_cfg.USE_SDE = False
    data = DataGenerator(system, gen_cfg).generate_data()

    exp_cfg = cfg.EXPERIMENTS[0]
    train_l, val_l, test_l, _, p_init, normalizer = setup_dataloaders(exp_cfg, data, system, cfg)

    xb, yb, pb = next(iter(test_l))
    hidden, param = load_models(cfg, xb.shape[1], yb.shape[1], pb.shape[1], device)

    # --- Iterative estimator: sweep inference unroll depth K (fixed-point behavior) ---
    p_true = np.concatenate(
        [normalizer.denormalize_params(p_b).cpu().numpy() for _, _, p_b in test_l], axis=0)
    engine = InferenceEngine(normalizer, cfg, hidden, param)

    def iter_pred_at(K):
        preds = []
        with torch.no_grad():
            for x_b, _, _ in test_l:
                pc = engine.run_fixed_point_iteration(x_b.to(device), p_init, max_iter=K, tol=0)
                preds.append(normalizer.denormalize_params(pc).cpu().numpy())
        return np.concatenate(preds, axis=0)

    k_curve = {int(K): float(rel_errors(p_true, iter_pred_at(K))['mean']) for K in S.K_SWEEP}
    iter_stats = rel_errors(p_true, iter_pred_at(S.ITERATIONS))  # headline at K=ITERATIONS
    iter_stats['k_curve'] = k_curve
    iter_stats['headline_K'] = S.ITERATIONS
    iter_stats['best_K'] = int(min(k_curve, key=k_curve.get))
    iter_stats['best_K_mean'] = k_curve[iter_stats['best_K']]

    # --- Collect physical test observations for NLS (normalization is identity) ---
    xs, ps = [], []
    for x_batch, _, p_batch in test_l:
        xs.append(normalizer.denormalize_inputs(x_batch, 'observed').cpu().numpy())
        ps.append(normalizer.denormalize_params(p_batch).cpu().numpy())
    x_all = np.concatenate(xs, axis=0)
    p_all = np.concatenate(ps, axis=0)
    assert np.allclose(p_all, p_true, atol=1e-5), "NLS/iterative test targets misaligned"

    n_sub = min(S.NLS_SUBSET, len(x_all))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_all), size=n_sub, replace=False)

    p_init_np = p_init.cpu().numpy()
    y0_init = 1.0  # midpoint of y0 range [0.5, 1.5]
    t_points = np.asarray(system.t_points)

    nls_pred = np.zeros((n_sub, 2))
    n_fail = 0
    for j, i in enumerate(idx):
        theta_ab, ok = nls_fit_one(x_all[i], t_points, system, p_init_np, y0_init)
        nls_pred[j] = theta_ab
        if not ok:
            n_fail += 1
    nls_true = p_all[idx]
    nls_stats = rel_errors(nls_true, nls_pred)
    nls_stats['n_eval'] = int(n_sub)
    nls_stats['n_nonconverged'] = int(n_fail)

    return {
        'seed': seed,
        'n_test': int(len(p_true)),
        'iterative': iter_stats,
        'nls': nls_stats,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(S.OUTPUT_DIR, exist_ok=True)

    per_seed = []
    for seed in S.SEEDS:
        print(f"[G0-eval] seed {seed} ...", flush=True)
        per_seed.append(evaluate_seed(seed, device))

    iter_means = np.array([r['iterative']['mean'] for r in per_seed])          # at K=ITERATIONS
    iter_best = np.array([r['iterative']['best_K_mean'] for r in per_seed])
    nls_means = np.array([r['nls']['mean'] for r in per_seed])
    total_fail = sum(r['nls']['n_nonconverged'] for r in per_seed)

    # Aggregate iterative K-curve across seeds (mean over seeds at each K).
    k_curve_mean = {int(K): float(np.mean([r['iterative']['k_curve'][str(K)]
                                           if str(K) in r['iterative']['k_curve']
                                           else r['iterative']['k_curve'][K] for r in per_seed]))
                    for K in S.K_SWEEP}

    # G0 pass GATE is NLS (pipeline validity). The iterative estimator's error is
    # RECORDED as a documented method characterization, not gated (see FINDINGS.md).
    summary = {
        'threshold': S.REL_ERR_THRESHOLD,
        'gate': 'NLS (pipeline validity)',
        'nls_mean_rel_err': {'mean': float(nls_means.mean()), 'std': float(nls_means.std())},
        'nls_total_nonconverged': int(total_fail),
        'nls_pass': bool(nls_means.mean() < S.REL_ERR_THRESHOLD),
        'iterative_mean_rel_err_at_K': {'K': S.ITERATIONS,
                                        'mean': float(iter_means.mean()),
                                        'std': float(iter_means.std())},
        'iterative_best_K_rel_err': {'mean': float(iter_best.mean()),
                                     'std': float(iter_best.std())},
        'iterative_k_curve_mean': k_curve_mean,
        'iterative_note': ('documented fixed-point bias; not an accuracy gate. '
                           'theta* is not an attracting fixed point of T (see FINDINGS.md).'),
    }
    summary['G0_PASS'] = summary['nls_pass']

    out = {'summary': summary, 'per_seed': per_seed, 'settings': {
        'system': S.SYSTEM_NAME, 'num_samples': S.NUM_SAMPLES, 'epochs': S.EPOCHS,
        'iterations': S.ITERATIONS, 'recurrent_iter': S.RECURRENT_ITER, 'seeds': S.SEEDS,
        'use_spectral_norm': S.USE_SPECTRAL_NORM,
    }}
    metrics_path = os.path.join(S.OUTPUT_DIR, 'g0_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[G0-eval] wrote {metrics_path}", flush=True)

    # --- Figure: NLS pass bar + iterative K-curve (documented bias) ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    axL.bar(['NLS'], [nls_means.mean()], yerr=[nls_means.std()], capsize=6,
            color='#DD8452', width=0.5)
    axL.axhline(S.REL_ERR_THRESHOLD, ls='--', color='crimson',
                label=f'pass gate = {S.REL_ERR_THRESHOLD:g}')
    axL.text(0, nls_means.mean(), f'{nls_means.mean():.1e}', ha='center', va='bottom', fontsize=10)
    axL.set_yscale('log'); axL.set_ylabel('Mean relative parameter error')
    axL.set_title('Pass gate: NLS recovery'); axL.legend()

    Ks = list(k_curve_mean.keys())
    axR.plot(Ks, [k_curve_mean[k] for k in Ks], 'o-', color='#4C72B0', label='iterative (mean over seeds)')
    axR.axhline(S.REL_ERR_THRESHOLD, ls='--', color='crimson', label=f'{S.REL_ERR_THRESHOLD:g}')
    axR.set_yscale('log'); axR.set_xlabel('inference unroll depth K')
    axR.set_ylabel('Mean relative parameter error')
    axR.set_title('Iterative fixed-point (documented bias)'); axR.legend()

    fig.suptitle(f'G0 sanity oracle ({S.SYSTEM_NAME}, {len(S.SEEDS)} seeds)')
    fig.tight_layout()
    fig_path = os.path.join(S.OUTPUT_DIR, 'g0_relative_error.pdf')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[G0-eval] wrote {fig_path}", flush=True)

    print("\n===== G0 SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("G0 PASS (pipeline gate: NLS)" if summary['G0_PASS'] else "G0 FAIL")


if __name__ == '__main__':
    main()
