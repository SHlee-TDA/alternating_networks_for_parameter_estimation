"""
Regenerate the pipeline figures (Fig 7 SIR scatter, Fig 8 LV scatter, Fig 9 OGTT collapse,
Fig 11 OGTT phase) reusing the (label-corrected) analyzer plotting, and copy the outputs into
paper/simods/figures/exp/ under the names the manuscript references.

Label fixes now baked into src/analyzer.py: "both sloppy" -> "both weakly recovered"; phase-portrait
title/axes use LaTeX names ($S_I$,$\sigma$,...); scatter metrics box omits absent (N/A) models.

Usage: python experiments/det_meanlocus/regen_pipeline_figs.py
"""
import os, sys, json, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator, SingleNetworkBaseline
from src.infer import InferenceEngine
from src.analyzer import get_analyzer_class
from tools.exp_tools import get_system_class, set_seed

FIGDIR = 'paper/simods/figures/exp'


def setup(run_dir, baseline=False):
    hits = glob.glob(os.path.join(os.path.dirname(run_dir), '*', 'experiment_config.json'))
    key = os.path.basename(run_dir.rstrip('/'))
    cfg_path = next((h for h in hits if key in h), hits[0])
    cfgj = json.load(open(cfg_path))
    c = Config()
    for k in ['SYSTEM_NAME', 'NUM_SAMPLES', 'USE_DERIVATIVE', 'USE_SPECTRAL_NORM', 'SEED', 'TEST_SPLIT',
              'BATCH_SIZE', 'ITERATIONS', 'NORMALIZER_STATE_SCALES', 'NORMALIZER_PARAM_BOUNDS',
              'RUN_BASELINE', 'MODEL_CONFIG', 'DEVICE']:
        if k in cfgj: setattr(c, k, cfgj[k])
    c.RUN_BASELINE = baseline
    c.RESULTS_DIR = 'results/_figregen/'          # scratch output dir for analyzer
    c.EXPERIMENT_NAME = key
    c.__post_init__()
    scen = {kk: cfgj[kk] for kk in ['NAME', 'USE_SDE', 'SCENARIO', 'VAL_SOURCE', 'OOD_SPLIT'] if kk in cfgj}
    exp_config = scen or c.EXPERIMENTS[0]
    dev = c.DEVICE
    set_seed(c.SEED)
    system = get_system_class(c.SYSTEM_NAME)()
    gen = DataGenerator(system, c); sim = gen.generate_data()
    train_l, val_l, test_l, real_l, p_init, normalizer = setup_dataloaders(exp_config, sim, system, c)
    sx, sy, sp = next(iter(train_l))
    if baseline:
        net = SingleNetworkBaseline(sx.shape[1], sp.shape[1], c.MODEL_CONFIG['param_net'], use_spectral_norm=False).to(dev)
        net.load_state_dict(torch.load(os.path.join(run_dir, 'baseline_net.pth'), map_location=dev))
        eng = InferenceEngine(normalizer, c, None, net)
        H = P = None
    else:
        H = HiddenVarPredictor(sx.shape[1], sy.shape[1], sp.shape[1], c.MODEL_CONFIG['hidden_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
        P = ParameterEstimator(sx.shape[1], sy.shape[1], sp.shape[1], c.MODEL_CONFIG['param_net'], use_spectral_norm=c.USE_SPECTRAL_NORM).to(dev)
        H.load_state_dict(torch.load(os.path.join(run_dir, 'Hnet.pth'), map_location=dev))
        P.load_state_dict(torch.load(os.path.join(run_dir, 'Pnet.pth'), map_location=dev))
        eng = InferenceEngine(normalizer, c, H, P)
    pt, pp = eng.get_predictions(test_l, p_init)
    Analyzer = get_analyzer_class(c.SYSTEM_NAME)
    az = Analyzer(hidden_net=H, param_net=P, normalizer=normalizer, config=c, system=system, history={})
    return c, system, normalizer, az, pt, pp, test_l


def copy_out(c, src_name, dst_name):
    src = os.path.join(c.RESULTS_DIR, c.SYSTEM_NAME, c.EXPERIMENT_NAME, src_name)
    dst = os.path.join(FIGDIR, dst_name)
    shutil.copyfile(src, dst)
    print(f"  {src}  ->  {dst}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    # ---- Fig 7: SIR wide, iterative -> sir_scatter_wide_iter.pdf ----
    print("[Fig 7] SIR wide iterative")
    c, sysm, nrm, az, pt, pp, test_l = setup('results/det_paper/sir/fixedIC_wide', baseline=False)
    xb = next(iter(test_l))[0].cpu().numpy()
    az.evaluate_simulation(pt, pp, np.zeros_like(pt), x_obs=xb)      # ours=pp, base=dummy
    copy_out(c, 'sir_scatter_comparison.pdf', 'sir_scatter_wide_iter.pdf')

    # ---- Fig 8: LV baseline -> lv_scatter_base.pdf ----
    print("[Fig 8] LV baseline")
    c, sysm, nrm, az, pt, pp, test_l = setup('results/det_paper/lotka_volterra/baseline', baseline=True)
    az.evaluate_simulation(pt, np.zeros_like(pt), pp)               # ours=dummy, base=pp
    copy_out(c, 'lv_scatter_comparison.pdf', 'lv_scatter_base.pdf')

    # ---- Fig 9 + Fig 11: OGTT iterative ----
    print("[Fig 9/11] OGTT iterative")
    c, sysm, nrm, az, pt, pp, test_l = setup('results/det_paper/ogtt_simul/main_sn0p9', baseline=False)
    az.evaluate_simulation(pt, pp, np.zeros_like(pt))               # -> sim_symmetric_collapse.pdf
    copy_out(c, 'sim_symmetric_collapse.pdf', 'ogtt_collapse_iter.pdf')
    # phase portrait from one test sample
    xb = next(iter(test_l))[0]
    x_sample = xb[0:1].to(c.DEVICE)                                 # already normalized by loader
    az.plot_phase_portraits(x_sample, pt[0])
    copy_out(c, 'phase_portraits.pdf', 'ogtt_phase.pdf')
    print("done.")


if __name__ == '__main__':
    main()
