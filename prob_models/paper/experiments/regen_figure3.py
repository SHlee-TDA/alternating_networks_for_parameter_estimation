"""Regenerate figure3_B/C (global joint + local posterior) non-interactively.

Reuses the canonical, non-interactive context loaders from ``_context.py`` (the
same checkpoints/data split as B3/B4/B7) and the plotting functions from the
repo-root ``figure3.py`` (whose legend labels are now "A-DCVAE", not "Iter").
Baseline kept as the architecture-matched Single CVAE (see paper Table 1 / Fig 3).

Run from the MAIN tree (which holds results/ + data/), importing this worktree's
code:  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> conda run -n vision_task \
       python <worktree>/prob_models/paper/experiments/regen_figure3.py
Output PDFs go to this worktree's prob_models/paper/figures/.
"""
import sys
from pathlib import Path

import torch

import _context as C                       # inserts worktree root on sys.path
sys.path.insert(0, str(C.PROJECT_ROOT))
import figure3 as F                         # plot_figure3_B/C, samplers, tail selector

SEED = 42
FIG_DIR = C.PROJECT_ROOT / "prob_models" / "paper" / "figures"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = C.make_config(seed=SEED, num_samples=50000, device=device)
    ctx = C.load_data_context(cfg)          # calls set_seed(SEED) internally
    normalizer = ctx["normalizer"]
    Hnet, Pnet = C.load_dual_cvae(cfg, ctx)
    single = C.load_single_cvae(cfg, ctx)

    # ---- global test population (mirror figure3.py main: up to 1000 samples) ----
    xs, ps, n = [], [], 0
    for xb, _, pb in ctx["test_l"]:
        xs.append(xb); ps.append(pb); n += xb.size(0)
        if n >= 1000:
            break
    x_global = torch.cat(xs, dim=0).to(device)
    p_true_global = torch.cat(ps, dim=0).to(device)

    torch.manual_seed(SEED)
    with torch.no_grad():
        p_single_g = F.single_cvae_sampling(single, x_global, num_samples=1).squeeze(1)
        _, p_iter_g, _ = F.pseudo_gibbs_sampling(Hnet, Pnet, x_global,
                                                 num_chains=1, num_steps=30, burn_in=20)
    p_iter_g = p_iter_g.squeeze(1)

    p_true_phys = normalizer.denormalize_params(p_true_global).cpu().numpy()
    p_single_g_phys = normalizer.denormalize_params(p_single_g).cpu().numpy()
    p_iter_g_phys = normalizer.denormalize_params(p_iter_g).cpu().numpy()

    # ---- pick the representative "mean-trap" observation for the local panel ----
    tail = F.find_best_mean_trap_sample(p_true_phys, p_single_g_phys)
    x_local = x_global[tail:tail + 1]
    p_true_local = p_true_phys[tail]
    print(f"[*] tail idx {tail} | true S_I={p_true_local[0]:.3f} sigma={p_true_local[1]:.3f}")

    torch.manual_seed(SEED)
    with torch.no_grad():
        p_single_l = F.single_cvae_sampling(single, x_local, num_samples=1000).squeeze(0)
        _, p_iter_l, _ = F.pseudo_gibbs_sampling(Hnet, Pnet, x_local,
                                                 num_chains=1000, num_steps=50, burn_in=30)
    p_iter_l = p_iter_l.squeeze(0)
    p_single_l_phys = normalizer.denormalize_params(p_single_l).cpu().numpy()
    p_iter_l_phys = normalizer.denormalize_params(p_iter_l).cpu().numpy()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    F.plot_figure3_B_global_2D(p_true_phys, p_single_g_phys, p_iter_g_phys, FIG_DIR)
    F.plot_figure3_C_local_2D(p_true_local, p_single_l_phys, p_iter_l_phys, FIG_DIR)
    print(f"[*] wrote figure3_B/C to {FIG_DIR}")


if __name__ == "__main__":
    main()
