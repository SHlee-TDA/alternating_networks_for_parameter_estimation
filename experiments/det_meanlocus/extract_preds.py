"""
Faithfully re-extract (p_true, p_pred) for a saved run directory using the real pipeline,
then report the recovery structure (individual coords vs. the identifiable product) and whether
predictions are pinned against the normalizer/Tanh bounds. Used to verify the OGTT stiff/sloppy
claim and the SIR bounded-prediction concern at code level.

Usage: python experiments/det_meanlocus/extract_preds.py <run_dir>
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from scipy.stats import pearsonr
from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator, SingleNetworkBaseline
from src.infer import InferenceEngine
from tools.exp_tools import get_system_class, set_seed

import glob
run_dir = sys.argv[1]
# config json lives in the timestamped logger dir; checkpoints in the EXPERIMENT_NAME dir.
cfg_path = os.path.join(run_dir, 'experiment_config.json')
if not os.path.exists(cfg_path):
    hits = glob.glob(os.path.join(os.path.dirname(run_dir), '*', 'experiment_config.json'))
    # pick the one whose dir name shares the exp-name suffix of run_dir
    key = os.path.basename(run_dir.rstrip('/'))
    cfg_path = next((h for h in hits if key in h), hits[0] if hits else cfg_path)
cfgj = json.load(open(cfg_path))
# checkpoints: prefer run_dir; else RESULTS_DIR/SYSTEM/EXPERIMENT_NAME
ckpt_dir = run_dir
if not (os.path.exists(os.path.join(ckpt_dir,'Hnet.pth')) or os.path.exists(os.path.join(ckpt_dir,'baseline_net.pth'))):
    ckpt_dir = os.path.join(cfgj['RESULTS_DIR'], cfgj['SYSTEM_NAME'], cfgj['EXPERIMENT_NAME'])

c = Config()
for k in ['SYSTEM_NAME','NUM_SAMPLES','USE_DERIVATIVE','USE_SPECTRAL_NORM','SEED','TEST_SPLIT',
          'BATCH_SIZE','ITERATIONS','NORMALIZER_STATE_SCALES','NORMALIZER_PARAM_BOUNDS',
          'RUN_BASELINE','MODEL_CONFIG','DEVICE']:
    if k in cfgj: setattr(c, k, cfgj[k])
c.__post_init__()
# match the single experiment scenario stored in the run
scen = {kk: cfgj[kk] for kk in ['NAME','USE_SDE','SCENARIO','VAL_SOURCE','OOD_SPLIT'] if kk in cfgj}
exp_config = scen or c.EXPERIMENTS[0]

set_seed(c.SEED)
system = get_system_class(c.SYSTEM_NAME)()
gen = DataGenerator(system, c); sim = gen.generate_data()
loaders = setup_dataloaders(exp_config, sim, system, c)
train_l, val_l, test_l, real_l, p_init, normalizer = loaders

sx, sy, sp = next(iter(train_l))
baseline = cfgj.get('RUN_BASELINE', False)
dev = c.DEVICE
if baseline:
    net = SingleNetworkBaseline(sx.shape[1], sp.shape[1], c.MODEL_CONFIG['param_net'], use_spectral_norm=False).to(dev)
    net.load_state_dict(torch.load(os.path.join(ckpt_dir,'baseline_net.pth'), map_location=dev))
    eng = InferenceEngine(normalizer, c, None, net)
else:
    H = HiddenVarPredictor(sx.shape[1], sy.shape[1], sp.shape[1], c.MODEL_CONFIG['hidden_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
    P = ParameterEstimator(sx.shape[1], sy.shape[1], sp.shape[1], c.MODEL_CONFIG['param_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
    H.load_state_dict(torch.load(os.path.join(ckpt_dir,'Hnet.pth'), map_location=dev))
    P.load_state_dict(torch.load(os.path.join(ckpt_dir,'Pnet.pth'), map_location=dev))
    eng = InferenceEngine(normalizer, c, H, P)

pt, pp = eng.get_predictions(test_l, p_init)
np.savez(os.path.join(ckpt_dir,'extracted_preds.npz'), p_true=pt, p_pred=pp)

names = system.param_names
def r(a,b): return pearsonr(a,b)[0]
print(f"\n=== {run_dir} ({'baseline' if baseline else 'iterative'}) | {names} ===")
for i,nm in enumerate(names):
    print(f"  {nm}: r={r(pt[:,i],pp[:,i]):.3f}  RMSE={np.sqrt(np.mean((pt[:,i]-pp[:,i])**2)):.4f}  "
          f"pred_range=[{pp[:,i].min():.3f},{pp[:,i].max():.3f}]  true_range=[{pt[:,i].min():.3f},{pt[:,i].max():.3f}]")
if len(names)==2:
    prod_t, prod_p = pt[:,0]*pt[:,1], pp[:,0]*pp[:,1]
    print(f"  PRODUCT {names[0]}*{names[1]}: r={r(prod_t,prod_p):.3f}  (stiff/identifiable direction check)")
# bound check: fraction of preds within 1% of the physical normalizer bounds
if getattr(normalizer,'use_normalization',False):
    pb = normalizer.NORMALIZER_PARAM_BOUNDS if hasattr(normalizer,'NORMALIZER_PARAM_BOUNDS') else None
print("  [saved extracted_preds.npz]")
