"""
Round-5 review: test-time observation-perturbation stress test (noiseless training / noisy test).

Pre-registered protocol (fixed BEFORE looking at any output):
  * Comparators (identical test samples & identical noise draws for all three):
      - iterative    : trained H/P alternating fixed-point operator (Tanh-bounded param output)
      - direct       : SingleNetworkBaseline direct regressor x -> theta (UNBOUNDED output)
      - prior_mean   : constant predictor = mean of the training parameters in the LOSS coordinate
                       (geometric mean for OGTT [log MSE], arithmetic mean for SIR/LV [physical MSE]).
  * Primary noise family: independent additive Gaussian on each observed channel j,
        x'_ij = x_ij + alpha * s_j * eps_ij ,  eps ~ N(0,1),
    with s_j = channel-wise std of the CLEAN TRAINING observations (physical units).
    alpha grid fixed: {0, 0.005, 0.01, 0.02, 0.05, 0.10}.
  * Noise injected in PHYSICAL observation space, then re-normalized with the clean-training
    normalizer (which is a pure linear scaling x/scale for observed states -> exact).
  * R noise replicates per clean test trajectory (same trajectories, resampled noise); a single
    training seed (42) is used because retraining is GPU-bound (stated honestly in the paper).
  * Metrics (per replicate, then aggregated mean +/- std over replicates):
      - per-coordinate physical RMSE and Pearson r on the finite&positive subset,
      - NRMSE (RMSE / training-prior std, per coord),
      - log-parameter error e_log = || log|pred| - log|true| ||_2 (finite&positive subset),
      - clean-to-noisy degradation Delta R(alpha) = R(alpha) - R(0),
      - prediction displacement D(alpha) = median_i || log pred(x') - log pred(x) ||_2,
      - failure rate, split into non-finite / positivity-violation / support-violation,
      - OGTT only: product S_I*sigma signed bias and abs error,
      - bias / noise-variance decomposition in log coordinates (over replicates).
  * Failed samples are NOT silently dropped: success-conditioned errors are always reported
    together with the failure rate. Support = physical [min,max] of the TRAINING parameters.

Outputs:
  results/det_meanlocus/noise_stress_test.json           (all aggregates)
  results/det_meanlocus/noise_stress_test_raw.csv        (one row per system/method/alpha/replicate)

Usage:
  python experiments/det_meanlocus/noise_stress_test.py [--systems sir,ogtt_simul,lotka_volterra]
                                                        [--reps 30] [--device cpu]
"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator, SingleNetworkBaseline
from tools.exp_tools import get_system_class, set_seed

ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
OUT_JSON = 'results/det_meanlocus/noise_stress_test.json'
OUT_CSV = 'results/det_meanlocus/noise_stress_test_raw.csv'

# (system, iterative run dir, iterative config, baseline run dir, baseline config)
RUNS = {
    'sir': ('results/det_paper/sir/main_sn0p9',
            'results/det_paper/sir/20260720_085103_main_sn0p9/experiment_config.json',
            'results/det_paper/sir/baseline',
            'results/det_paper/sir/20260720_085216_baseline/experiment_config.json'),
    'ogtt_simul': ('results/det_paper/ogtt_simul/main_sn0p9',
                   'results/det_paper/ogtt_simul/20260720_093028_main_sn0p9/experiment_config.json',
                   'results/det_paper/ogtt_simul/baseline',
                   'results/det_paper/ogtt_simul/20260720_093019_baseline/experiment_config.json'),
    'lotka_volterra': ('results/det_paper/lotka_volterra/main_sn0p9',
                       'results/det_paper/lotka_volterra/20260720_085119_main_sn0p9/experiment_config.json',
                       'results/det_paper/lotka_volterra/baseline',
                       'results/det_paper/lotka_volterra/20260720_085216_baseline/experiment_config.json'),
}

CFG_KEYS = ['SYSTEM_NAME', 'NUM_SAMPLES', 'USE_DERIVATIVE', 'USE_SPECTRAL_NORM', 'SEED', 'TEST_SPLIT',
            'BATCH_SIZE', 'ITERATIONS', 'NORMALIZER_STATE_SCALES', 'NORMALIZER_PARAM_BOUNDS',
            'RUN_BASELINE', 'MODEL_CONFIG', 'DEVICE']
SCEN_KEYS = ['NAME', 'USE_SDE', 'SCENARIO', 'VAL_SOURCE', 'OOD_SPLIT']


def build_config(cfg_path, device):
    cfgj = json.load(open(cfg_path))
    c = Config()
    for k in CFG_KEYS:
        if k in cfgj:
            setattr(c, k, cfgj[k])
    c.DEVICE = device
    c.__post_init__()
    scen = {k: cfgj[k] for k in SCEN_KEYS if k in cfgj}
    exp_config = scen or c.EXPERIMENTS[0]
    return c, exp_config


def load_split(cfg_path, device):
    """Run the exact (cached) data pipeline for one config; return normalizer + loaders."""
    c, exp_config = build_config(cfg_path, device)
    set_seed(c.SEED)
    system = get_system_class(c.SYSTEM_NAME)()
    gen = DataGenerator(system, c)
    sim = gen.generate_data()
    train_l, val_l, test_l, real_l, p_init, normalizer = setup_dataloaders(exp_config, sim, system, c)
    return c, system, train_l, test_l, p_init, normalizer


def stack_loader(loader):
    Xs, Ps = [], []
    for xb, _, pb in loader:
        Xs.append(xb)
        Ps.append(pb)
    return torch.cat(Xs), torch.cat(Ps)


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def eval_run(system_name, device, reps, seed0=12345):
    iter_dir, iter_cfg, base_dir, base_cfg = RUNS[system_name]

    # --- one shared split/normalizer (iter and baseline share it; verified identical stats) ---
    c, system, train_l, test_l, p_init, normalizer = load_split(iter_cfg, device)
    dev = torch.device(device)
    param_names = list(system.param_names)
    p_dim = len(param_names)
    ITER = c.ITERATIONS
    scale_obs = float(normalizer.state_scales[0].cpu())     # observed states: pure linear scale
    sx0, sy0, sp0 = next(iter(test_l))                      # hidden-variable flat dim from a batch
    sy_dim = sy0.shape[1]

    # test physical observations & truth
    Xn_test, Pn_test = stack_loader(test_l)
    Xphys_test = (Xn_test * scale_obs).numpy()              # (n_test, n_ch) physical
    theta_true = normalizer.denormalize_params(Pn_test.to(dev)).cpu().numpy()  # (n_test, p) physical
    n_test, n_ch = Xphys_test.shape

    # training statistics: channel std (physical) and parameter support
    Xn_tr, Pn_tr = stack_loader(train_l)
    Xphys_tr = (Xn_tr * scale_obs).numpy()
    s_j = Xphys_tr.std(axis=0)                               # (n_ch,) physical channel scale
    theta_tr = normalizer.denormalize_params(Pn_tr.to(dev)).cpu().numpy()
    supp_lo = theta_tr.min(axis=0)
    supp_hi = theta_tr.max(axis=0)
    train_std = theta_tr.std(axis=0)                         # for NRMSE

    # prior-mean predictor in the loss coordinate: mean of normalized train params -> denormalize
    prior_norm = Pn_tr.mean(dim=0, keepdim=True).to(dev)
    prior_phys = normalizer.denormalize_params(prior_norm).cpu().numpy().ravel()  # (p,)

    # --- models ---
    sx_dim = Xn_test.shape[1]
    H = HiddenVarPredictor(sx_dim, sy_dim, p_dim,
                           c.MODEL_CONFIG['hidden_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
    P = ParameterEstimator(sx_dim, sy_dim, p_dim,
                           c.MODEL_CONFIG['param_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
    H.load_state_dict(torch.load(os.path.join(iter_dir, 'Hnet.pth'), map_location=dev)); H.eval()
    P.load_state_dict(torch.load(os.path.join(iter_dir, 'Pnet.pth'), map_location=dev)); P.eval()
    base = SingleNetworkBaseline(sx_dim, p_dim, c.MODEL_CONFIG['param_net'], use_spectral_norm=False).to(dev)
    base.load_state_dict(torch.load(os.path.join(base_dir, 'baseline_net.pth'), map_location=dev)); base.eval()

    p_init_norm = normalizer.normalize_params(p_init).to(dev).view(1, -1)

    def predict_iter(Xn):
        with torch.no_grad():
            pc = p_init_norm.repeat(Xn.size(0), 1)
            for _ in range(ITER):
                pc = P(Xn, H(Xn, pc))
            return normalizer.denormalize_params(pc).cpu().numpy()

    def predict_base(Xn):
        with torch.no_grad():
            return normalizer.denormalize_params(base(Xn)).cpu().numpy()

    def predict_prior(n):
        return np.tile(prior_phys, (n, 1))

    predictors = {'iterative': predict_iter, 'direct': predict_base, 'prior_mean': predict_prior}

    # clean predictions (alpha = 0, deterministic)
    Xn_clean = torch.from_numpy(Xphys_test / scale_obs).float().to(dev)
    clean_pred = {'iterative': predict_iter(Xn_clean),
                  'direct': predict_base(Xn_clean),
                  'prior_mean': predict_prior(n_test)}

    rows = []
    # per-(method, alpha) store: list over reps of metric dicts, plus log-pred stacks for bias/var
    agg = {m: {a: {'metrics': [], 'logpreds': []} for a in ALPHAS} for m in predictors}
    R0_store = {m: {} for m in predictors}  # clean risk per coord for degradation

    for a in ALPHAS:
        this_reps = 1 if a == 0.0 else reps
        for r in range(this_reps):
            rng = np.random.default_rng(seed0 + int(a * 1e6) + r)
            if a == 0.0:
                Xphys_noisy = Xphys_test
            else:
                eps = rng.standard_normal(size=Xphys_test.shape)
                Xphys_noisy = Xphys_test + a * s_j[None, :] * eps
            Xn = torch.from_numpy((Xphys_noisy / scale_obs).astype(np.float32)).to(dev)

            for m, fn in predictors.items():
                pred = fn(Xn) if m != 'prior_mean' else predict_prior(n_test)
                met, logpred = compute_metrics(pred, theta_true, clean_pred[m], supp_lo, supp_hi,
                                               train_std, system_name, param_names)
                met.update(system=system_name, method=m, alpha=a, replicate=r)
                rows.append(met)
                agg[m][a]['metrics'].append(met)
                agg[m][a]['logpreds'].append(logpred)   # (n_test, p) with nan where invalid
                if a == 0.0:
                    R0_store[m]['rmse'] = met['rmse_mean']
                    R0_store[m]['nrmse'] = met['nrmse']
                    R0_store[m]['elog'] = met['elog_median']

    # aggregate
    summary = {}
    for m in predictors:
        summary[m] = {}
        for a in ALPHAS:
            mets = agg[m][a]['metrics']
            def ms(key):
                vals = np.array([x[key] for x in mets], float)
                vals = vals[np.isfinite(vals)]
                return (float(np.mean(vals)) if vals.size else float('nan'),
                        float(np.std(vals)) if vals.size else float('nan'))
            rmse_m, rmse_s = ms('rmse_mean')
            nrmse_m, nrmse_s = ms('nrmse')
            elog_m, elog_s = ms('elog_median')
            disp_m, disp_s = ms('disp_median')
            failany_m, failany_s = ms('fail_any')
            bias2, nvar = biasvar_logcoord(agg[m][a]['logpreds'], theta_true, a)
            entry = dict(
                rmse=[rmse_m, rmse_s], nrmse=[nrmse_m, nrmse_s],
                elog_median=[elog_m, elog_s], displacement_median=[disp_m, disp_s],
                fail_any=[failany_m, failany_s],
                fail_nonfinite=ms('fail_nonfinite')[0], fail_positivity=ms('fail_positivity')[0],
                fail_support=ms('fail_support')[0],
                r_per_coord=[ms('r_%d' % k)[0] for k in range(p_dim)],
                rmse_per_coord=[ms('rmse_%d' % k)[0] for k in range(p_dim)],
                delta_nrmse=(nrmse_m - R0_store[m]['nrmse']) if np.isfinite(nrmse_m) else float('nan'),
                delta_elog=(elog_m - R0_store[m]['elog']) if np.isfinite(elog_m) else float('nan'),
                logbias2=bias2, log_noise_var=nvar,
            )
            if system_name == 'ogtt_simul':
                entry['prod_signed_bias'] = ms('prod_signed_bias')[0]
                entry['prod_abserr'] = ms('prod_abserr')[0]
            summary[m][str(a)] = entry

    meta = dict(system=system_name, n_test=n_test, n_channels=n_ch, p_dim=p_dim,
                param_names=param_names, iterations=ITER, reps=reps, alphas=ALPHAS,
                channel_std_phys=s_j.tolist(), support_lo=supp_lo.tolist(), support_hi=supp_hi.tolist(),
                prior_mean_phys=prior_phys.tolist(), train_param_std=train_std.tolist(),
                iter_dir=iter_dir, base_dir=base_dir, scale_obs=scale_obs)
    return meta, summary, rows


def compute_metrics(pred, true, clean_pred, supp_lo, supp_hi, train_std, system_name, param_names):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    p_dim = pred.shape[1]
    finite = np.isfinite(pred).all(axis=1)
    positive = (pred > 0).all(axis=1)
    in_supp = ((pred >= supp_lo[None, :]) & (pred <= supp_hi[None, :])).all(axis=1)
    valid = finite & positive                                     # for log-based metrics
    fail_nonfinite = float(np.mean(~finite))
    fail_positivity = float(np.mean(finite & ~positive))
    fail_support = float(np.mean(finite & ~in_supp))
    fail_any = float(np.mean(~(finite & positive & in_supp)))

    met = dict(fail_nonfinite=fail_nonfinite, fail_positivity=fail_positivity,
               fail_support=fail_support, fail_any=fail_any, n_valid=int(valid.sum()))

    # success-conditioned per-coordinate RMSE / NRMSE / Pearson r
    v = valid
    rmses = []
    for k in range(p_dim):
        if v.sum() >= 2:
            err = pred[v, k] - true[v, k]
            rmse_k = float(np.sqrt(np.mean(err ** 2)))
            r_k = pearson(pred[v, k], true[v, k])
        else:
            rmse_k, r_k = float('nan'), float('nan')
        met['rmse_%d' % k] = rmse_k
        met['r_%d' % k] = r_k
        rmses.append(rmse_k)
    met['rmse_mean'] = float(np.nanmean(rmses))
    met['nrmse'] = float(np.nanmean([rmses[k] / (train_std[k] + 1e-12) for k in range(p_dim)]))

    # log-parameter error and displacement (finite&positive subset)
    logpred_full = np.full_like(pred, np.nan)
    if v.any():
        logpred_full[v] = np.log(pred[v])
        elog = np.linalg.norm(np.log(pred[v]) - np.log(np.abs(true[v]) + 1e-30), axis=1)
        met['elog_median'] = float(np.median(elog))
        # displacement vs clean pred (need clean also valid&positive)
        cv = np.isfinite(clean_pred).all(axis=1) & (clean_pred > 0).all(axis=1) & v
        if cv.any():
            disp = np.linalg.norm(np.log(pred[cv]) - np.log(clean_pred[cv]), axis=1)
            met['disp_median'] = float(np.median(disp))
        else:
            met['disp_median'] = float('nan')
    else:
        met['elog_median'] = float('nan')
        met['disp_median'] = float('nan')

    if system_name == 'ogtt_simul' and p_dim >= 2:
        vv = valid
        if vv.any():
            prod_pred = pred[vv, 0] * pred[vv, 1]
            prod_true = true[vv, 0] * true[vv, 1]
            met['prod_signed_bias'] = float(np.mean(prod_pred - prod_true))
            met['prod_abserr'] = float(np.mean(np.abs(prod_pred - prod_true)))
        else:
            met['prod_signed_bias'] = float('nan')
            met['prod_abserr'] = float('nan')

    return met, logpred_full


def biasvar_logcoord(logpreds, true, alpha):
    """logpreds: list over reps of (n_test, p) arrays (nan where invalid). Bias^2 and noise var."""
    if len(logpreds) == 0:
        return float('nan'), float('nan')
    stack = np.stack(logpreds, axis=0)                # (R, n, p)
    valid = np.isfinite(stack).all(axis=(0, 2))       # samples valid in every replicate
    if valid.sum() < 2:
        return float('nan'), float('nan')
    s = stack[:, valid, :]                            # (R, m, p)
    logtrue = np.log(np.abs(true[valid]) + 1e-30)     # (m, p)
    mean_log = s.mean(axis=0)                         # (m, p)
    bias2 = float(np.mean(np.sum((mean_log - logtrue) ** 2, axis=1)))
    if s.shape[0] < 2:
        return bias2, 0.0
    nvar = float(np.mean(np.sum(((s - mean_log[None]) ** 2).mean(axis=0), axis=1)))
    return bias2, nvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', default='sir,ogtt_simul,lotka_volterra')
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    os.makedirs('results/det_meanlocus', exist_ok=True)
    all_summary, all_meta, all_rows = {}, {}, []
    for sysname in args.systems.split(','):
        sysname = sysname.strip()
        print('\n' + '=' * 70 + f'\n[noise stress test] {sysname}\n' + '=' * 70)
        meta, summary, rows = eval_run(sysname, args.device, args.reps)
        all_meta[sysname] = meta
        all_summary[sysname] = summary
        all_rows.extend(rows)
        # brief console readout
        for m in ['iterative', 'direct', 'prior_mean']:
            print(f'  {m:10s}', end='')
            for a in ALPHAS:
                e = summary[m][str(a)]
                print(f'  a={a:<5g} nrmse={e["nrmse"][0]:.3f} disp={e["displacement_median"][0]:.3f} '
                      f'fail={e["fail_any"][0]:.3f}', end='')
            print()

    json.dump(dict(meta=all_meta, summary=all_summary, protocol=dict(alphas=ALPHAS, reps=args.reps,
              noise='iid additive alpha*s_j*N(0,1) in physical obs space; s_j=clean-train channel std',
              training_seeds=1, device=args.device)),
              open(OUT_JSON, 'w'), indent=2)

    # raw CSV
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        # put identifiers first
        head = ['system', 'method', 'alpha', 'replicate']
        keys = head + [k for k in keys if k not in head]
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
    print(f'\nwrote {OUT_JSON}\nwrote {OUT_CSV}')


if __name__ == '__main__':
    main()
