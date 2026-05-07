# main.py
"""
Main Experiment Pipeline

Orchestrates the entire workflow:
1. Setup: Config loading, Seed setting, Logger initialization.
2. Phase 1 (Data): Generates or loads synthetic data (ODE/SDE) via DataGenerator.
3. Phase 2 (Training):
   - Prepares DataLoaders (Sim only or Hybrid with Real data).
   - Initializes Networks (f_theta, g_phi).
   - Executes Training Loop via Trainer.
4. Phase 3 (Evaluation):
   - Analyzes performance on Test set.
   - Validates against Real Clinical Data using the Analyzer.

Usage:
    python main.py --epochs 2000
"""
import os
import random
import copy
import gc
import json
import importlib
import argparse
from collections import defaultdict
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset, WeightedRandomSampler, Subset

from config import Config
from data_loader import DataGenerator, RealOGTTDataLoader
from models import HiddenVarPredictor, ParameterEstimator, SingleNetworkBaseline
from trainer import Trainer
from analyzer import get_analyzer_class
from utils import Normalizer, ExperimentLogger


def set_seed(seed):
    """Fixes random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_system_class(system_name):
    """Dynamically imports the system class (e.g., systems.ogtt_simul.OgttSimul)."""
    module_name = f"systems.{system_name}"
    # Convention: snake_case file -> PascalCase class (e.g., ogtt_simul -> OgttSimul)
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def prepare_dataloaders(exp_config, sim_data_tuple, system, global_config):
    """
    Constructs DataLoaders for Train, Val, and Test.
    Handles Normalization and Hybrid Mixing (Sim + Real).
    """
    # 1. Unpack Simulation Data
    obs_sim, hid_sim, params_sim, t_points = sim_data_tuple
    
    # 2. Initialize Normalizer (Data-Driven)
    # Uses 99.9th percentile of Sim data to define robust scaling bounds
    scale_obs = np.percentile(np.abs(obs_sim), 99.9)
    scale_hid = np.percentile(np.abs(hid_sim), 99.9)
    calc_scales = [scale_obs * 1.2, scale_hid * 1.2] # Add 20% margin
    
    p_min = np.min(params_sim, axis=0)
    p_max = np.max(params_sim, axis=0)
    p_bounds = (p_min / 1.2, p_max * 1.2) # Log-space friendly margin
    
    use_log = (global_config.SYSTEM_NAME == 'ogtt_simul')
    use_normalization = (global_config.SYSTEM_NAME == 'ogtt_simul')
    normalizer = Normalizer(
        system, 
        global_config.DEVICE, 
        state_scales=calc_scales, 
        param_bounds=p_bounds,
        use_log_params=use_log,
        use_normalization=use_normalization
    )
    
    # 3. Create Sim Dataset
    X_sim = torch.FloatTensor(obs_sim).view(len(obs_sim), -1).to(global_config.DEVICE)
    Y_sim = torch.FloatTensor(hid_sim).view(len(hid_sim), -1).to(global_config.DEVICE)
    P_sim = torch.FloatTensor(params_sim).to(global_config.DEVICE)
    
    # Normalize
    dataset_sim = TensorDataset(
        normalizer.normalize_inputs(X_sim, 'observed'),
        normalizer.normalize_inputs(Y_sim, 'hidden'),
        normalizer.normalize_params(P_sim)
    )
    
    # Split Sim Data
    total_len = len(dataset_sim)
    test_len = int(total_len * global_config.TEST_SPLIT)
    val_len = int(total_len * global_config.TEST_SPLIT)
    train_len = total_len - val_len - test_len
    
    sim_train, sim_val, sim_test = random_split(
        dataset_sim, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(global_config.SEED)
    )

    # 4. Prepare Real Data (for Hybrid Training)
    if global_config.SYSTEM_NAME == 'ogtt_simul':
        real_loader = RealOGTTDataLoader(
            file_path='data/clean_sumner_n_612.xlsx', 
            config=global_config,
            split_file='data/data_split_indices.json'
        )
        obs_real, hid_real, params_real, _ = real_loader.load_data()
        
        # Normalize Real Data using Sim-derived Normalizer
        X_real = torch.FloatTensor(obs_real).view(len(obs_real), -1).to(global_config.DEVICE)
        Y_real = torch.FloatTensor(hid_real).view(len(hid_real), -1).to(global_config.DEVICE)
        P_real = torch.FloatTensor(params_real).to(global_config.DEVICE)
        
        dataset_real = TensorDataset(
            normalizer.normalize_inputs(X_real, 'observed'),
            normalizer.normalize_inputs(Y_real, 'hidden'),
            normalizer.normalize_params(P_real)
        )
        
        # Load Split Indices for Real Data
        with open('data/data_split_indices.json', 'r') as f:
            split_data = json.load(f)
        real_train = Subset(dataset_real, split_data['train_indices'])
        real_test = Subset(dataset_real, split_data['test_indices']) # For final evaluation
    else:
        real_train = None
        real_test = None

    # 5. Construct Train Loader (Scenario-based)
    scenario = exp_config.get('SCENARIO', 'sim_only')
    
    if scenario == 'hybrid':
        print(f"  -> [Hybrid] Mixing Sim ({len(sim_train)}) + Real ({len(real_train)})")
        final_train_set = ConcatDataset([sim_train, real_train])
        
        # Weighted Sampling to balance Sim/Real ratio
        real_ratio = exp_config.get('REAL_RATIO', 0.3)
        w_sim = (1 - real_ratio) / len(sim_train)
        w_real = real_ratio / len(real_train)
        weights = [w_sim] * len(sim_train) + [w_real] * len(real_train)
        
        sampler = WeightedRandomSampler(weights, num_samples=len(final_train_set), replacement=True)
        train_loader = DataLoader(final_train_set, batch_size=global_config.BATCH_SIZE, sampler=sampler)
    else:
        print(f"  -> [Sim Only] Using Sim Data ({len(sim_train)})")
        train_loader = DataLoader(sim_train, batch_size=global_config.BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(sim_val, batch_size=global_config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(sim_test, batch_size=global_config.BATCH_SIZE, shuffle=False)
    
    # Initial Parameter Guess (Geometric Mean of Sim Params)
    P_sim_safe = torch.maximum(P_sim, torch.tensor(1e-6).to(global_config.DEVICE))
    p_initial_guess = torch.exp(torch.log(P_sim_safe).mean(dim=0))
    
    return train_loader, val_loader, test_loader, real_test, p_initial_guess, normalizer

def run_experiment_pipeline(global_config):
    # Setup
    set_seed(global_config.SEED)
    device = torch.device(global_config.DEVICE)
    print(f"System: {global_config.SYSTEM_NAME} | Device: {device}")
    
    SystemClass = get_system_class(global_config.SYSTEM_NAME)
    system = SystemClass()
    
    # --- Phase 1: Data Generation (Caching) ---
    # Generates ODE and SDE data once, to be reused across experiments.
    data_cache = {}
    required_configs = set()
    
    for exp in global_config.EXPERIMENTS:
        sde_mode = exp.get('USE_SDE', False)
        # If 'mixed', we need both ODE and SDE datasets
        if sde_mode == 'mixed':
            required_configs.add(False) # ODE
            required_configs.add(True)  # SDE
        else:
            required_configs.add(sde_mode)
            
    print(f"\n[Phase 1] Ensuring data availability for modes: {required_configs}")
    for use_sde in required_configs:
        gen_config = copy.deepcopy(global_config)
        gen_config.USE_SDE = use_sde
        generator = DataGenerator(system, gen_config)
        data_cache[use_sde] = generator.generate_data()

    # --- Phase 2: Experiment Loop ---
    total_exps = len(global_config.EXPERIMENTS)
    for idx, exp_config in enumerate(global_config.EXPERIMENTS):
        exp_name = exp_config['NAME']
        print(f"\n{'='*60}")
        print(f"Experiment {idx+1}/{total_exps}: {exp_name}")
        
        # 1. Configure Current Run
        run_config = copy.deepcopy(global_config)
        run_config.EXPERIMENT_NAME = exp_name
        # run_config.USE_SPECTRAL_NORM = exp_config['use_spectral_norm']
        # run_config.USE_CONSISTENCY_LOSS = exp_config['use_consistency_loss']
        # run_config.USE_SDE = exp_config.get('use_sde', False)
        
        for key, value in exp_config.items():
            if hasattr(run_config, key):
                setattr(run_config, key, value)
            else:
                setattr(run_config, key, value)
        
    
        # Logger Setup
        logger = ExperimentLogger(run_config)
        print(f"  -> Log Dir: {logger.results_dir}")
        
        # Inject Logger Path into Config for Trainer
        trainer_config = copy.deepcopy(run_config)
        trainer_config.RESULTS_DIR = logger.results_dir
        # trainer_config.SYSTEM_NAME = ""     # Already in path
        # trainer_config.EXPERIMENT_NAME = "" # Already in path
        
        # 2. Data Preparation
        req_sde = exp_config.get('USE_SDE', False)
        
        if req_sde == 'mixed':
            # Merge ODE and SDE data
            ode_data = data_cache[False]
            sde_data = data_cache[True]
            # Concatenate arrays: (Obs, Hid, Params, T)
            sim_data_tuple = tuple(np.concatenate([d1, d2], axis=0) if i < 3 else d1 
                                   for i, (d1, d2) in enumerate(zip(ode_data, sde_data)))
            print(f"  -> [Data] Mixed ODE+SDE ({len(sim_data_tuple[0])} samples)")
        else:
            sim_data_tuple = data_cache[req_sde]
            
        train_l, val_l, test_l, real_test_set, p_init, normalizer = prepare_dataloaders(
            exp_config, sim_data_tuple, system, global_config
        )
        
        # Create Real Test Loader for Analyzer
        real_test_loader = DataLoader(real_test_set, batch_size=global_config.BATCH_SIZE, shuffle=False)

        # 3. Model Initialization
        sample_x, sample_y, sample_p = next(iter(train_l))
        
        if getattr(run_config, 'RUN_BASELINE', False):
            baseline_net = SingleNetworkBaseline(
                flat_x_dim=sample_x.shape[1],
                flat_y_dim=sample_p.shape[1],
                model_config=run_config.MODEL_CONFIG['param_net'], # 기존 모델의 파라미터 구조 차용
                use_spectral_norm=run_config.USE_SPECTRAL_NORM
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
        
        # 4. Training
        trainer = Trainer(
            train_l, val_l, 
            config,
            hidden_net=hidden_net, param_net=param_net,
            baseline_net=baseline_net
        )
        
        if getattr(run_config, 'RUN_BASELINE', False):
            baseline_net, history = trainer.train()
        else:
            hidden_net, param_net, history = trainer.train()
        
        # --- Phase 3: Analysis & Evaluation ---
        if getattr(run_config, 'RUN_BASELINE', False):
            print("  -> [Baseline Mode] Skipping Analyzer for now. (Will be implemented in Step 4)")
            metrics = {
                'train_loss': history['train_total_loss'][-1],
                'val_loss': history['val_total_loss'][-1],
                'test_mse': -1, # 평가는 4단계에서 수행할 예정이므로 임시값 할당
                'epoch': len(history['train_total_loss'])
            }
            logger.log_result_to_csv(metrics)
            print(f"  -> Baseline Experiment Completed. Metrics: {metrics}")
            
            del baseline_net, trainer, train_l, val_l, test_l
            
        else:
            print("  -> Starting Analysis...")
            
            # [수정된 부분] 동적으로 시스템에 맞는 Analyzer 할당
            AnalyzerClass = get_analyzer_class(global_config.SYSTEM_NAME)
            analyzer = AnalyzerClass(
                f_theta=hidden_net, 
                g_phi=param_net, 
                test_loader=test_l, 
                config=trainer_config,
                system=system, 
                p_initial_guess=p_init, 
                normalizer=normalizer, 
                history=history
            )
            
            try:
                print("  -> Loading Baseline Model for comparison...")
                baseline_model = SingleNetworkBaseline(
                    flat_x_dim=sample_x.shape[1],
                    flat_y_dim=sample_p.shape[1],
                    model_config=run_config.MODEL_CONFIG['param_net'],
                    use_spectral_norm=None
                ).to(device)
                
                base_dir = os.path.dirname(logger.results_dir)
                baseline_paths = glob.glob(os.path.join(base_dir, '*', 'baseline_net.pth'))
                
                if baseline_paths:
                    latest_baseline_path = max(baseline_paths, key=os.path.getmtime)
                    print(f"  -> Found baseline at: {latest_baseline_path}")
                    baseline_model.load_state_dict(torch.load(latest_baseline_path))
                    
                    # 비교 분석 실행
                    analyzer.run_comparison(baseline_model)
                else:
                    print("  -> [Warning] baseline_net.pth not found in previous run folders.")
                    
            except Exception as e:
                import traceback
                print(f"  -> [Warning] Baseline comparison skipped: {e}")
                traceback.print_exc() # 숨겨진 진짜 에러 원인을 출력합니다.
                
            # A. Simulation Metrics
            analyzer.plot_loss_curves()
            p_true, p_pred = analyzer.evaluate_predictions()
            analyzer.plot_scatter(p_true, p_pred)
            analyzer.plot_phase_portraits()
            analyzer.plot_spectral_norms_by_layer()

            # B. Real Data Validation
            print("  -> Evaluating on Real Clinical Data...")
            if hasattr(analyzer, 'evaluate_real_data') and real_test_loader is not None:
                print("  -> Evaluating on Real Clinical Data...")
                analyzer.evaluate_real_data(real_test_loader, baseline_model=baseline_model)
            
            # C. Logging Metrics
            metrics = {
                'train_loss': history['train_total_loss'][-1],
                'val_loss': history['val_total_loss'][-1],
                'test_mse': np.mean((p_true - p_pred)**2),
                'epoch': len(history['train_total_loss'])
            }
            logger.log_result_to_csv(metrics)
            print(f"  -> Experiment Completed. Metrics: {metrics}")
            
            # Cleanup
            del hidden_net, param_net, trainer, analyzer, train_l, val_l, test_l
            
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None, help="Override default epochs")
    args = parser.parse_args()
    
    if args.epochs: 
        config.EPOCHS = args.epochs
        
    run_experiment_pipeline(config)