"""
Probabilistic Generative Experiment Pipeline (CVAE Framework)

Orchestrates the entire workflow for Paper 2:
1. Setup: Config loading, Seed setting, Logger initialization.
2. Phase 1 (Data Generation): Generates or loads synthetic data.
3. Phase 2 (Training): Initializes SingleCVAE or Dual CVAE, and trains via ELBO.
4. Phase 3 (Evaluation): Evaluates posterior distributions and runs Pseudo Gibbs Sampling.
"""
import os
import gc
import sys
import copy
import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

# [Paper 2 전용 모듈 임포트]
from prob_models.config import ProbConfig 
from prob_models.models import SingleCVAE, HiddenStateCVAE, ParameterCVAE
from prob_models.trainer import ProbTrainer
from prob_models.analysis import run_prob_evaluation_phase

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataGenerator, setup_dataloaders
from tools.exp_tools import ExperimentLogger, set_seed, get_system_class

def run_experiment_pipeline(global_config):
    # Setup
    set_seed(global_config.SEED)
    device = torch.device(global_config.DEVICE)
    print(f"System: {global_config.SYSTEM_NAME} | Device: {device}")
    
    SystemClass = get_system_class(global_config.SYSTEM_NAME)
    system = SystemClass()
    
    # Phase 1: Data Generation (Caching)
    data_cache = {}
    required_configs = set()
    
    for exp in global_config.EXPERIMENTS:
        sde_mode = exp.get('USE_SDE', False)
        if sde_mode == 'mixed':
            required_configs.update([False, True])
        else:
            required_configs.add(sde_mode)
            
    print(f"\n[Phase 1] Ensuring data availability for SDE modes: {required_configs}")
    for use_sde in required_configs:
        gen_config = copy.deepcopy(global_config)
        gen_config.USE_SDE = use_sde
        generator = DataGenerator(system, gen_config)
        data_cache[use_sde] = generator.generate_data()

    # Phase 2 & 3: Experiment Loop
    total_exps = len(global_config.EXPERIMENTS)
    for idx, exp_config in enumerate(global_config.EXPERIMENTS):
        exp_name = exp_config['NAME']
        print(f"\n{'='*60}")
        print(f"Experiment {idx+1}/{total_exps}: {exp_name}")
        
        # Configure Current Run
        run_config = copy.deepcopy(global_config)
        for key, value in exp_config.items():
            setattr(run_config, key, value)
            
        logger = ExperimentLogger(run_config)
        print(f"  -> Log Dir: {logger.results_dir}")
        
        # 2. Data Preparation
        req_sde = exp_config.get('USE_SDE', False)
        if req_sde == 'mixed':
            sim_data_tuple = tuple(np.concatenate([d1, d2], axis=0) if i < 3 else d1 
                                   for i, (d1, d2) in enumerate(zip(data_cache[False], data_cache[True])))
        else:
            sim_data_tuple = data_cache[req_sde]
            
        loaders = setup_dataloaders(exp_config, sim_data_tuple, system, run_config)
        train_l, val_l, test_l, real_test_loader, p_init, normalizer = loaders

        # JSON Config 저장 (Reconstruction 시 사용)
        config_save_path = os.path.join(logger.results_dir, 'experiment_config.json')
        if hasattr(run_config, 'save_to_json'):
            run_config.save_to_json(config_save_path)
        else:
            with open(config_save_path, 'w') as f:
                json.dump(asdict(run_config), f, indent=4)

        # 3. Model Initialization (Probabilistic)
        sample_x, sample_y, sample_p = next(iter(train_l))
        x_dim, y_dim, theta_dim = sample_x.shape[1], sample_y.shape[1], sample_p.shape[1]
        
        latent_dim_h = getattr(run_config, 'LATENT_DIM_HIDDEN', 4)
        latent_dim_p = getattr(run_config, 'LATENT_DIM_PARAM', 2)
        hidden_dims = getattr(run_config, 'HIDDEN_DIMS', [256, 256, 256, 256])
                
        
        if getattr(run_config, 'RUN_BASELINE', False):
            baseline_net = SingleCVAE(
                x_dim=x_dim, 
                theta_dim=theta_dim, 
                latent_dim=latent_dim_p,
                hidden_dims=hidden_dims
            ).to(device)
            state_estimator, param_estimator = None, None
            print("  -> Initialized: [Single CVAE] Baseline Model")
        else:
            baseline_net = None
            state_estimator = HiddenStateCVAE(
                x_dim=x_dim, 
                theta_dim=theta_dim, 
                y_dim=y_dim, 
                latent_dim=latent_dim_h,
                hidden_dims=hidden_dims
            ).to(device)
            param_estimator = ParameterCVAE(
                x_dim=x_dim, 
                y_dim=y_dim, 
                theta_dim=theta_dim, 
                latent_dim=latent_dim_p,
                hidden_dims=hidden_dims
            ).to(device)
            
            # [안전장치] infer.py에서 속성을 조회할 수 있도록 할당
            param_estimator.theta_dim = theta_dim
            # C4 fix (2026-07-13): wire config INFER_NOISE_* onto the models so inference
            # (N4 injection noise) actually reflects the configured value instead of the
            # hard-coded 0.05 fallback in infer.pseudo_gibbs_sampling.
            state_estimator.infer_noise_y = getattr(run_config, 'INFER_NOISE_Y', 0.05)
            param_estimator.infer_noise_p = getattr(run_config, 'INFER_NOISE_P', 0.05)
            print("  -> Initialized: [Dual CVAE] Proposed Iterative Framework")
        
        # Phase 2: Training
        trainer = ProbTrainer(
            train_loader=train_l, val_loader=val_l, config=run_config,
            hidden_cvae=state_estimator, param_cvae=param_estimator, baseline_cvae=baseline_net
        )
        
        if getattr(run_config, 'RUN_BASELINE', False):
            baseline_net, history = trainer.train()
            param_estimator = baseline_net  # 평가 함수 인터페이스 통일용 매핑
        else:
            state_estimator, param_estimator, history = trainer.train()
        
        # Phase 3: Evaluation (리팩토링된 analysis.py의 평가 모듈 호출)
        run_prob_evaluation_phase(
            run_config=run_config, 
            state_estimator=state_estimator, 
            param_estimator=param_estimator, 
            test_l=test_l, 
            normalizer=normalizer, 
            device=device
        )
            
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = ProbConfig()
    parser = argparse.ArgumentParser(description="Training Pipeline for Probabilistic Parameter Estimation")
    parser.add_argument('--system', type=str, default=None, help="Dataset system name (e.g., sir, lotka_volterra, ogtt_simul)")
    parser.add_argument('--run_baseline', type=str, default=None, help="'true' for Single CVAE, 'false' for Dual CVAE")
    parser.add_argument('--epochs', type=int, default=None, help="Override default epochs")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to save checkpoints and evaluation plots")
    args = parser.parse_args()
    
    if args.system is not None:
        config.SYSTEM_NAME = args.system
    if args.run_baseline is not None:
        config.RUN_BASELINE = (args.run_baseline.lower() in ['true', '1', 't', 'y', 'yes'])
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.results_dir is not None:
        config.RESULTS_DIR = args.results_dir
        
    print(f"\n🚀 Probabilistic Pipeline Start")
    print(f" - System: {config.SYSTEM_NAME}")
    print(f" - Baseline Mode: {config.RUN_BASELINE}")
    print(f" - Results Dir: {config.RESULTS_DIR}")
        
    run_experiment_pipeline(config)