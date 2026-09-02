"""G0 iterative-estimator tuning sweep.

Searches network capacity, epochs/data, spectral margin, and recurrent unroll depth
to see whether the iterative fixed-point estimator can reach <1e-3 relative parameter
error on the linear_oracle. Drives config.py + main.py (backing up / restoring
config.py), then evaluates each trial with a K-sweep and reports the best rel error.

Run: python experiments/g0_sanity_oracle/sweep_g0.py
"""
import sys, os, re, subprocess, json, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_g0, evaluate_g0 as E
from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator
from src.infer import InferenceEngine
from tools.exp_tools import get_system_class, set_seed

CFG = os.path.join(ROOT, 'config.py')
RESULTS_DIR = 'experiments/g0_sanity_oracle/results'
KS = [1, 2, 3, 5, 8, 10, 15, 20, 30]

# Each trial: name, use_spectral_norm, spectral_scale, rec_iter, hidden_dims,
#             num_samples, epochs
TRIALS = [
    dict(name='sw_snoff_big', sn=False, sscale=0.99, rec=5, dims=[512, 512, 512], ns=40000, ep=1000),
    dict(name='sw_snon_big',  sn=True,  sscale=0.99, rec=5, dims=[512, 512, 512], ns=40000, ep=1000),
]


def model_config_block(dims, sscale):
    d = str(list(dims))
    return ("    MODEL_CONFIG: Dict[str, Any] = field(default_factory=lambda: {"
            f"'hidden_net': {{'hidden_dims': {d}, 'activation': 'SiLU', 'spectral_scale': {sscale}}}, "
            f"'param_net': {{'hidden_dims': {d}, 'activation': 'SiLU', 'spectral_scale': {sscale}}}"
            "})")


def patch_all(original, t, seed):
    ov = {
        'SYSTEM_NAME': "'linear_oracle'", 'EXPERIMENT_NAME': f"'{t['name']}'", 'SEED': str(seed),
        'NUM_SAMPLES': str(t['ns']), 'EPOCHS': str(t['ep']), 'ITERATIONS': '30',
        'USE_SPECTRAL_NORM': str(t['sn']), 'RECURRENT_ITER': str(t['rec']),
        'AUGMENTATION_FACTOR': '0', 'BATCH_SIZE': '256', 'LEARNING_RATE': '1e-3',
        'USE_EARLY_STOPPING': 'False', 'USE_DERIVATIVE': 'False', 'RUN_BASELINE': 'False',
        'RESULTS_DIR': f"'{RESULTS_DIR}'",
    }
    text = run_g0.patch_config(original, ov)
    # replace MODEL_CONFIG block (from its field line to the closing "    })" line)
    text, n = re.subn(r'(?ms)^    MODEL_CONFIG:.*?^    \}\)$',
                      model_config_block(t['dims'], t['sscale']), text)
    assert n == 1, f"MODEL_CONFIG match {n}"
    # enable recurrent loss
    text, n = re.subn(r'LOSS_CONFIG:[^\n]*=\s*field\(default_factory=lambda:\s*\[.*?\]\s*\)',
                      "LOSS_CONFIG: List[Tuple[str, float]] = field(default_factory=lambda: "
                      "[('supervised', 1.0), ('recurrent', 1.0)])", text, flags=re.DOTALL)
    assert n == 1, f"LOSS_CONFIG match {n}"
    return text


def build_cfg(t, seed):
    cfg = Config()
    cfg.SYSTEM_NAME = 'linear_oracle'; cfg.SEED = seed; cfg.NUM_SAMPLES = t['ns']
    cfg.AUGMENTATION_FACTOR = 0; cfg.USE_DERIVATIVE = False; cfg.USE_SPECTRAL_NORM = t['sn']
    cfg.RESULTS_DIR = RESULTS_DIR; cfg.EXPERIMENT_NAME = t['name']
    mc = {'hidden_dims': list(t['dims']), 'activation': 'SiLU', 'spectral_scale': t['sscale']}
    cfg.MODEL_CONFIG = {'hidden_net': dict(mc), 'param_net': dict(mc)}
    cfg.__post_init__()
    return cfg


def evaluate(t, seed=0):
    cfg = build_cfg(t, seed); set_seed(seed)
    system = get_system_class('linear_oracle')()
    gen = Config(); gen.SYSTEM_NAME='linear_oracle'; gen.NUM_SAMPLES=t['ns']; gen.AUGMENTATION_FACTOR=0
    gen.USE_DERIVATIVE=False; gen.SEED=seed; gen.USE_SDE=False
    data = DataGenerator(system, gen).generate_data()
    tr, va, test_l, _, p_init, norm = setup_dataloaders(cfg.EXPERIMENTS[0], data, system, cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xb, yb, pb = next(iter(test_l))
    rp = os.path.join(RESULTS_DIR, 'linear_oracle', t['name'])
    h = HiddenVarPredictor(xb.shape[1], yb.shape[1], pb.shape[1],
                           model_config=cfg.MODEL_CONFIG['hidden_net'], use_spectral_norm=t['sn']).to(device)
    p = ParameterEstimator(xb.shape[1], yb.shape[1], pb.shape[1],
                           model_config=cfg.MODEL_CONFIG['param_net'], use_spectral_norm=t['sn']).to(device)
    h.load_state_dict(torch.load(rp + '/Hnet.pth')); p.load_state_dict(torch.load(rp + '/Pnet.pth'))
    h.eval(); p.eval()
    Ps = [p_b for _, _, p_b in test_l]
    pt = norm.denormalize_params(torch.cat(Ps).to(device)).cpu().numpy()
    eng = InferenceEngine(norm, cfg, h, p)
    curve = {}
    for K in KS:
        ps = []
        with torch.no_grad():
            for x_b, _, _ in test_l:
                pc = eng.run_fixed_point_iteration(x_b.to(device), p_init, max_iter=K, tol=0)
                ps.append(norm.denormalize_params(pc).cpu().numpy())
        curve[K] = float(E.rel_errors(pt, np.concatenate(ps))['mean'])
    bestK = min(curve, key=curve.get)
    return curve, bestK, curve[bestK]


def main():
    original = open(CFG).read()
    results = {}
    try:
        for t in TRIALS:
            open(CFG, 'w').write(patch_all(original, t, seed=0))
            print(f"\n[sweep] training {t['name']} ...", flush=True)
            proc = subprocess.run([sys.executable, 'main.py'], capture_output=True, text=True)
            el = [l for l in proc.stdout.splitlines() if l.startswith('Epoch')]
            print(f"[sweep] {t['name']} RC{proc.returncode} {el[-1] if el else ''}", flush=True)
            if proc.returncode != 0:
                print(proc.stdout.splitlines()[-15:]); continue
            curve, bestK, best = evaluate(t)
            results[t['name']] = dict(trial=t, curve=curve, bestK=bestK, best_rel=best)
            print(f"[sweep] {t['name']}: bestK={bestK} best_rel={best:.6f}  curve="
                  + " ".join(f"{k}:{v:.4f}" for k, v in curve.items()), flush=True)
    finally:
        open(CFG, 'w').write(original)
        print("[sweep] restored config.py", flush=True)
    with open(os.path.join(HERE, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n===== SWEEP SUMMARY =====")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['best_rel']):
        print(f"  {name:20s} best_rel={r['best_rel']:.6f} @K={r['bestK']}")


if __name__ == '__main__':
    main()
