# main.py
"""
Main Experiment Pipeline

Orchestrates the entire workflow:
1. Setup: Config loading, Seed setting, Logger initialization.
2. Phase 1 (Data Generation): Generates or loads synthetic data via DataGenerator.
3. Phase 2 (Training): Prepares DataLoaders, Initializes Networks, and executes Training Loop.
4. Phase 3 (Evaluation): Analyzes performance and validates against real clinical data.
"""
import os
import gc
import copy
import argparse

import numpy as np
import torch

from config import Config
from src.data_loader import DataGenerator, setup_dataloaders
from src.models import HiddenVarPredictor, ParameterEstimator, SingleNetworkBaseline
from src.trainer import Trainer
from src.infer import run_evaluation_phase  
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
            required_configs.update([False, True]) # ODE and SDE
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
            print(f"  -> [Data] Mixed ODE+SDE ({len(sim_data_tuple[0])} samples)")
        else:
            sim_data_tuple = data_cache[req_sde]
            
        # Extracted to data_loader.py for cleaner main file
        loaders = setup_dataloaders(exp_config, sim_data_tuple, system, run_config)
        train_l, val_l, test_l, real_test_loader, p_init, normalizer = loaders

        # Model Initialization
        sample_x, sample_y, sample_p = next(iter(train_l))
        
        if getattr(run_config, 'RUN_BASELINE', False):
            baseline_net = SingleNetworkBaseline(
                flat_x_dim=sample_x.shape[1], flat_y_dim=sample_p.shape[1],
                model_config=run_config.MODEL_CONFIG['param_net'],
                use_spectral_norm=None
            ).to(device)
            hidden_net, param_net = None, None
        else:
            baseline_net = None
            hidden_net = HiddenVarPredictor(
                sample_x.shape[1], sample_y.shape[1], sample_p.shape[1],
                model_config=run_config.MODEL_CONFIG['hidden_net'],
                use_spectral_norm=run_config.USE_SPECTRAL_NORM
            ).to(device)
            param_net = ParameterEstimator(
                sample_x.shape[1], sample_y.shape[1], sample_p.shape[1],
                model_config=run_config.MODEL_CONFIG['param_net'],
                use_spectral_norm=run_config.USE_SPECTRAL_NORM
            ).to(device)
        
        # Phase 2:Training
        trainer = Trainer(
            train_l, val_l, run_config,
            hidden_net=hidden_net, param_net=param_net, baseline_net=baseline_net
        )
        
        if getattr(run_config, 'RUN_BASELINE', False):
            baseline_net, history = trainer.train()
        else:
            hidden_net, param_net, history = trainer.train()
        
        # Phase 3: Evaluation & Analysis Phase
        # Abstracted heavy plotting/loading logic to an external function
        run_evaluation_phase(
            run_config, logger, system, history, 
            hidden_net, param_net, 
            test_l, real_test_loader, p_init, normalizer, 
            device
        )
            
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser(description="Training and Evaluation for Iterative Parameter Estimation")
    parser.add_argument('--system', type=str, default=None, help="Dataset system name (e.g., sir, lotka_volterra, ogtt_simul)")
    parser.add_argument('--run_baseline', type=str, default=None, help="'true' for Single Network, 'false' for Iterative")
    parser.add_argument('--epochs', type=int, default=None, help="Override default epochs")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to save checkpoints and evaluation plots")
    args = parser.parse_args()
    
    if args.system is not None:
        config.SYSTEM_NAME = args.system
        
    if args.run_baseline is not None:
        # Transform string coming from bash to boolean
        config.RUN_BASELINE = (args.run_baseline.lower() in ['true', '1', 't', 'y', 'yes'])
        
    if args.epochs is not None:
        config.EPOCHS = args.epochs
        
    if args.results_dir is not None:
        config.RESULTS_DIR = args.results_dir
        
    print(f"\n🚀 Pipeline Start")
    print(f" - System: {config.SYSTEM_NAME}")
    print(f" - Baseline (Single Net): {config.RUN_BASELINE}")
    print(f" - Results Directory: {config.RESULTS_DIR}")
        
    run_experiment_pipeline(config)