"""
Shared, non-interactive experiment context for the A-DCVAE (Paper 2) experiment track.

Unlike ``figure3.py`` / ``analysis.py`` (which rely on an interactive file
selector), this module reproduces the exact training context programmatically so
that scripts (B3 reference posterior, B7 ablation, B4 baselines, figures) can run
head-less with ``seed>=5`` loops.

The normalizer is reproduced deterministically: given the same ``SEED``,
``NUM_SAMPLES`` and cached simulation data, ``setup_dataloaders`` recomputes the
identical train split and hence identical normalizer scales/param-bounds that the
trained checkpoints used.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

# project root = three levels up from this file: prob_models/paper/experiments/_context.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from prob_models.config import ProbConfig
from prob_models.models import SingleCVAE, HiddenStateCVAE, ParameterCVAE
from src.data_loader import DataGenerator, setup_dataloaders
from tools.exp_tools import get_system_class, set_seed

# Canonical checkpoint locations (main-tree results, symlinked into the worktree).
CKPT = {
    "iter_cvae": "results/master_train/probabilistic/ogtt_simul/iter_cvae/ogtt_simul/probabilistic_ogtt_v1_prob",
    "single_cvae": "results/master_train/probabilistic/ogtt_simul/single_cvae/ogtt_simul/probabilistic_ogtt_v1_prob",
    "iter_det": "results/master_train/deterministic/ogtt_simul/iter_det/ogtt_simul/baseline_comparison",
    "single_det": "results/master_train/deterministic/ogtt_simul/single_det/ogtt_simul/baseline_comparison",
}


def make_config(seed=42, num_samples=50000, device=None, **overrides):
    cfg = ProbConfig()
    cfg.SEED = seed
    cfg.NUM_SAMPLES = num_samples
    if device is not None:
        cfg.DEVICE = device
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def load_data_context(cfg):
    """Reproduce the deterministic data split + normalizer for ``cfg``.

    Returns dict with train/val/test loaders, normalizer, and dims.
    """
    set_seed(cfg.SEED)
    system = get_system_class(cfg.SYSTEM_NAME)()
    gen = DataGenerator(system, cfg)
    sim_data = gen.generate_data()
    loaders = setup_dataloaders(vars(cfg), sim_data, system, cfg)
    train_l, val_l, test_l, real_test_l, p_init, normalizer = loaders
    sample_x, sample_y, sample_p = next(iter(test_l))
    return {
        "system": system,
        "train_l": train_l, "val_l": val_l, "test_l": test_l,
        "real_test_l": real_test_l, "normalizer": normalizer,
        "x_dim": sample_x.shape[1], "y_dim": sample_y.shape[1],
        "theta_dim": sample_p.shape[1],
    }


def _load_weight_safe(model, path, keys):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in keys:
            if key in ckpt:
                model.load_state_dict(ckpt[key]); return
        for k, v in ckpt.items():
            if "state_dict" in k and "optimizer" not in k:
                model.load_state_dict(v); return
    model.load_state_dict(ckpt)


def load_dual_cvae(cfg, ctx, ckpt_dir=None, infer_noise_y=None, infer_noise_p=None):
    """Load the trained Hidden/Parameter CVAEs (proposed A-DCVAE)."""
    ckpt_dir = ckpt_dir or CKPT["iter_cvae"]
    device = cfg.DEVICE
    lh = getattr(cfg, "LATENT_DIM_HIDDEN", 4)
    lp = getattr(cfg, "LATENT_DIM_PARAM", 2)
    hd = getattr(cfg, "HIDDEN_DIMS", [128, 128, 128, 128])
    Hnet = HiddenStateCVAE(ctx["x_dim"], ctx["theta_dim"], ctx["y_dim"], latent_dim=lh, hidden_dims=hd).to(device)
    Pnet = ParameterCVAE(ctx["x_dim"], ctx["y_dim"], ctx["theta_dim"], latent_dim=lp, hidden_dims=hd).to(device)
    Pnet.theta_dim = ctx["theta_dim"]
    _load_weight_safe(Hnet, os.path.join(ckpt_dir, "hidden_cvae.pth"), ["hidden_cvae_state_dict", "Hnet_state_dict"])
    _load_weight_safe(Pnet, os.path.join(ckpt_dir, "param_cvae.pth"), ["param_cvae_state_dict", "Pnet_state_dict"])
    Hnet.infer_noise_y = getattr(cfg, "INFER_NOISE_Y", 0.05) if infer_noise_y is None else infer_noise_y
    Pnet.infer_noise_p = getattr(cfg, "INFER_NOISE_P", 0.05) if infer_noise_p is None else infer_noise_p
    Hnet.eval(); Pnet.eval()
    return Hnet, Pnet


def load_single_cvae(cfg, ctx, ckpt_dir=None):
    ckpt_dir = ckpt_dir or CKPT["single_cvae"]
    device = cfg.DEVICE
    lp = getattr(cfg, "LATENT_DIM_PARAM", 2)
    hd = getattr(cfg, "HIDDEN_DIMS", [128, 128, 128, 128])
    net = SingleCVAE(ctx["x_dim"], ctx["theta_dim"], latent_dim=lp, hidden_dims=hd).to(device)
    _load_weight_safe(net, os.path.join(ckpt_dir, "baseline_cvae.pth"), ["baseline_cvae_state_dict"])
    net.eval()
    return net
