"""
Reproducible runner for the deterministic (SIMODS) paper experiments.
Wraps config.py + main.py with the corrected, good-performance settings:
  - normalization enabled for all systems (data_loader fix; resolves LV Tanh-cap)
  - spectral normalization on/off (contraction story vs ablation)
  - early stopping on (saves best_model.pth; avoids overfitting)

Usage:
  python experiments/det_meanlocus/run_paper_exp.py --system sir --sn true --samples 20000 --epochs 300
"""
import argparse
import os
import sys
# Ensure the project root (3 levels up) is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config
from main import run_experiment_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--system', required=True, choices=['sir', 'lotka_volterra', 'ogtt_simul'])
ap.add_argument('--sn', default='true')
ap.add_argument('--epochs', type=int, default=1000)
ap.add_argument('--samples', type=int, default=50000)
ap.add_argument('--patience', type=int, default=100)
ap.add_argument('--exp_name', default=None)
ap.add_argument('--results_dir', default='results/det_paper/')
ap.add_argument('--spectral_scale', type=float, default=0.99, help='per-layer SN target (lower => stronger contraction, smaller 1/(1-Lip_T))')
ap.add_argument('--ood', default='true', help='sir only: OOD (extrapolation) test split. Set false for in-distribution accuracy runs.')
ap.add_argument('--baseline', default='false', help='Run the direct SingleNetworkBaseline (x_obs->theta) instead of the iterative operator.')
args = ap.parse_args()

sn = args.sn.lower() in ('true', '1', 'yes', 't', 'y')
ood = args.ood.lower() in ('true', '1', 'yes', 't', 'y')
baseline = args.baseline.lower() in ('true', '1', 'yes', 't', 'y')
c = Config()
c.SYSTEM_NAME = args.system
c.USE_SPECTRAL_NORM = sn
c.RUN_BASELINE = baseline
c.USE_EARLY_STOPPING = True
c.EARLY_STOPPING_PATIENCE = args.patience
c.EPOCHS = args.epochs
c.NUM_SAMPLES = args.samples
default_name = 'baseline' if baseline else ('sn_on' if sn else 'sn_off')
c.EXPERIMENT_NAME = args.exp_name or default_name
c.RESULTS_DIR = args.results_dir
c.PLOT_PHASE_EVOLUTION = False  # eval-phase phase portrait is still produced; skip per-epoch for speed
# inject per-layer spectral-norm target into both networks
for net in ('hidden_net', 'param_net'):
    c.MODEL_CONFIG[net]['spectral_scale'] = args.spectral_scale
# toggle the SIR OOD/extrapolation split
for scen in c.SCENARIOS:
    scen['OOD_SPLIT'] = ood
c.__post_init__()  # rebuild EXPERIMENTS from SCENARIOS with the updated fields

print(f"[run_paper_exp] system={c.SYSTEM_NAME} SN={c.USE_SPECTRAL_NORM} samples={c.NUM_SAMPLES} "
      f"epochs={c.EPOCHS} patience={c.EARLY_STOPPING_PATIENCE} exp={c.EXPERIMENT_NAME}")
run_experiment_pipeline(c)
