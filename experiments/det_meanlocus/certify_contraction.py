"""
Part 7 (SIMODS review): contraction certification for a trained OGTT operator.

For the saved run <run_dir> (expects Hnet.pth, Pnet.pth + the timestamped experiment_config.json):
  (A) T_x(Theta) subset Theta : verified architecturally (P_psi ends in Tanh -> output in (-1,1)^p)
      and empirically (max |T output| over the test set).
  (B) Certified Lipschitz upper bound Lip_T = Lip_H * Lip_P, each = product over layers of the EXACT
      spectral norm sigma_max(W) (SVD of the effective, spectral-normalized weight) times the
      ExcludeLambda scale and the activation Lipschitz constant (SiLU=1.0998, Tanh=1). This is a
      rigorous UPPER bound (loose): Lip_T<1 => certified contraction; Lip_T>1 does NOT prove
      non-contractivity.
  (C) Empirical Lipschitz q_emp = max over random theta pairs of ||T(a)-T(b)|| / ||a-b|| (a lower-ish
      probe of the true constant).
  (D) Over the whole test set, in normalized coordinates: one-step residual eps=||T(theta_true)-theta_true||,
      fixed-point error ||theta*-theta_true|| (K iterations from the dataset-mean init), and the
      Lemma bound eps/(1-q). Reports coverage = fraction with ||theta*-theta_true|| <= eps/(1-q).

Usage: python experiments/det_meanlocus/certify_contraction.py results/det_paper/ogtt_simul/main_sn0p9
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator
from src.infer import InferenceEngine
from tools.exp_tools import get_system_class, set_seed

run_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/det_paper/ogtt_simul/main_sn0p9'
OUTJSON = os.path.join('results/det_meanlocus', 'contraction_cert_%s.json' % os.path.basename(run_dir.rstrip('/')))

# ---- locate config + checkpoints (mirrors extract_preds.py) ----
hits = glob.glob(os.path.join(os.path.dirname(run_dir), '*', 'experiment_config.json'))
key = os.path.basename(run_dir.rstrip('/'))
cfg_path = next((h for h in hits if key in h), hits[0])
cfgj = json.load(open(cfg_path))
ckpt_dir = run_dir

c = Config()
for k in ['SYSTEM_NAME', 'NUM_SAMPLES', 'USE_DERIVATIVE', 'USE_SPECTRAL_NORM', 'SEED', 'TEST_SPLIT',
          'BATCH_SIZE', 'ITERATIONS', 'NORMALIZER_STATE_SCALES', 'NORMALIZER_PARAM_BOUNDS',
          'RUN_BASELINE', 'MODEL_CONFIG', 'DEVICE']:
    if k in cfgj: setattr(c, k, cfgj[k])
c.__post_init__()
scen = {kk: cfgj[kk] for kk in ['NAME', 'USE_SDE', 'SCENARIO', 'VAL_SOURCE', 'OOD_SPLIT'] if kk in cfgj}
exp_config = scen or c.EXPERIMENTS[0]
dev = c.DEVICE

set_seed(c.SEED)
system = get_system_class(c.SYSTEM_NAME)()
gen = DataGenerator(system, c); sim = gen.generate_data()
loaders = setup_dataloaders(exp_config, sim, system, c)
train_l, val_l, test_l, real_l, p_init, normalizer = loaders
sx, sy, sp = next(iter(train_l))

H = HiddenVarPredictor(sx.shape[1], sy.shape[1], sp.shape[1], c.MODEL_CONFIG['hidden_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
P = ParameterEstimator(sx.shape[1], sy.shape[1], sp.shape[1], c.MODEL_CONFIG['param_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
H.load_state_dict(torch.load(os.path.join(ckpt_dir, 'Hnet.pth'), map_location=dev)); H.eval()
P.load_state_dict(torch.load(os.path.join(ckpt_dir, 'Pnet.pth'), map_location=dev)); P.eval()

# trigger spectral_norm forward hooks so module.weight holds the effective (normalized) weight
with torch.no_grad():
    _ = P(sx[:2].to(dev), sy[:2].to(dev)); _ = H(sx[:2].to(dev), sp[:2].to(dev))

# ---- exact activation Lipschitz constants ----
xx = torch.linspace(-8, 8, 200001)
silu_lip = float(torch.max(torch.abs(torch.autograd.functional.jacobian(
    lambda t: torch.nn.functional.silu(t).sum(), xx))))  # max |SiLU'|
ACT_LIP = {'SiLU': silu_lip, 'Tanh': 1.0, 'ReLU': 1.0, 'Sigmoid': 0.25}


def certified_lip(net, act_name):
    """Product of exact sigma_max(W) * ExcludeLambda scale * activation Lipschitz over the Sequential."""
    import torch.nn as nn
    from src.models import ExcludeLambda
    lip = 1.0; sigmas = []
    for m in net.network:
        if isinstance(m, nn.Linear):
            s = float(torch.linalg.svdvals(m.weight.detach().float().cpu())[0]); sigmas.append(s); lip *= s
        elif isinstance(m, ExcludeLambda):
            lip *= float(m.scale)
        elif isinstance(m, nn.Tanh):
            lip *= ACT_LIP['Tanh']
        elif isinstance(m, nn.SiLU):
            lip *= ACT_LIP['SiLU']
    return lip, sigmas


act = c.MODEL_CONFIG['param_net']['activation']
lipH, sH = certified_lip(H, act)
lipP, sP = certified_lip(P, act)
lipT = lipH * lipP


def T(x_norm, th):
    return P(x_norm, H(x_norm, th))


# ---- (C) empirical Lipschitz over random theta pairs on the test set ----
q_emp = 0.0
with torch.no_grad():
    for xb, _, _ in test_l:
        xb = xb.to(dev); n = xb.size(0)
        a = (torch.rand(n, sp.shape[1], device=dev) * 2 - 1)
        b = (torch.rand(n, sp.shape[1], device=dev) * 2 - 1)
        ta, tb = T(xb, a), T(xb, b)
        r = (torch.norm(ta - tb, dim=1) / (torch.norm(a - b, dim=1) + 1e-9))
        q_emp = max(q_emp, float(r.max()))

# ---- (A)+(D) residual, fixed-point error, bound coverage over the FULL test set ----
q_bound = lipT if lipT < 1 else q_emp     # use certified if <1, else the empirical estimate
eps_list, err_list, out_absmax = [], [], 0.0
p_init_norm = normalizer.normalize_params(p_init).to(dev)
with torch.no_grad():
    for xb, _, pb in test_l:
        xb = xb.to(dev); th_true = pb.to(dev)
        # one-step residual at ground truth
        eps = torch.norm(T(xb, th_true) - th_true, dim=1)
        # fixed point from dataset-mean init
        pc = p_init_norm.repeat(xb.size(0), 1)
        for _ in range(c.ITERATIONS):
            pc = T(xb, pc)
        err = torch.norm(pc - th_true, dim=1)
        out_absmax = max(out_absmax, float(T(xb, th_true).abs().max()))
        eps_list.append(eps.cpu().numpy()); err_list.append(err.cpu().numpy())
eps = np.concatenate(eps_list); err = np.concatenate(err_list)
bound = eps / (1 - q_bound)
coverage = float(np.mean(err <= bound + 1e-9))
bound_emp = eps / (1 - q_emp)
coverage_emp = float(np.mean(err <= bound_emp + 1e-9))


def pct(a): return {k: float(np.percentile(a, p)) for k, p in [('p50', 50), ('p90', 90), ('p99', 99), ('max', 100)]}


res = dict(
    run_dir=run_dir, activation=act, silu_lipschitz=silu_lip,
    T_maps_into_Theta=dict(architectural='P ends in Tanh -> output in (-1,1)^p',
                           empirical_max_abs_output=out_absmax, holds=bool(out_absmax <= 1.0 + 1e-6)),
    certified_lipschitz=dict(Lip_H=lipH, Lip_P=lipP, Lip_T=lipT,
                             certified_contraction=bool(lipT < 1),
                             note='rigorous UPPER bound (exact SVD spectral norms x scale x act Lip); loose'),
    empirical_lipschitz=dict(q_emp=q_emp, note='max ||T(a)-T(b)||/||a-b|| over random test pairs (probe)'),
    sigma_H=sH, sigma_P=sP,
    q_used_for_bound=q_bound,
    test_set=dict(n=int(len(eps)), eps=pct(eps), fixedpt_err=pct(err),
                  bound_certified_q=dict(q=q_bound, pct=pct(bound), coverage=coverage),
                  bound_empirical_q=dict(q=q_emp, pct=pct(bound_emp), coverage=coverage_emp),
                  eps_mean=float(eps.mean()), err_mean=float(err.mean())),
)
os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
json.dump(res, open(OUTJSON, 'w'), indent=2)
print(json.dumps(res, indent=2))
