import os
import random
import numpy as np
import torch
import importlib
import copy
import argparse
import sys
import json

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from config import Config
from data_loader import DataGenerator, RealOGTTDataLoader
from models import HiddenVarPredictor, ParameterEstimator
from trainer import Trainer
from analyzer import Analyzer
from utils import Normalizer, ExperimentLogger

# ==============================================================================
# 1. Setup & Utility Functions
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_system_class(system_name):
    """시스템 클래스 동적 로딩"""
    module_name = f"systems.{system_name}"
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load system '{system_name}'. Error: {e}")

def parse_args():
    """Phase 5 실험 설정을 위한 커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Alternating Networks Experiment Runner")
    parser.add_argument('--name', type=str, default=None, help='Specific experiment name to run')
    
    # Config Overrides
    parser.add_argument('--use_sde', type=str, default=None)
    parser.add_argument('--scenario', type=str, default=None)
    parser.add_argument('--real_ratio', type=float, default=None)
    parser.add_argument('--val_source', type=str, default=None)
    parser.add_argument('--use_spectral_norm', type=str, default=None)
    parser.add_argument('--use_consistency_loss', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    
    return parser.parse_args()

def str2bool(v):
    if v is None: return None
    if isinstance(v, bool): return v
    return v.lower() in ('yes', 'true', 't', 'y', '1')

# ==============================================================================
# 2. Data Loading Strategy
# ==============================================================================

def create_dataloaders(data_tuple, config, real_loader=None):
    """
    [Phase 5] Hybrid Learning 및 Validation 전략에 따른 DataLoader 생성
    """
    (train_set, val_set, test_set), _ = data_tuple
    
    # A. Train Loader 구성 (Hybrid Logic)
    if config.SCENARIO == 'hybrid' and real_loader is not None:
        print(f"  -> Creating Hybrid Train Loader (Real Ratio: {config.REAL_RATIO})")
        real_dataset = real_loader.get_dataset()
        
        # Concat
        hybrid_train_set = ConcatDataset([train_set, real_dataset])
        
        n_sim = len(train_set)
        n_real = len(real_dataset)
        
        if n_real > 0 and config.REAL_RATIO > 0:
            # Target Ratio r = (w_real * n_real) / (n_sim + w_real * n_real)
            r = config.REAL_RATIO
            w_real = (r * n_sim) / ((1 - r) * n_real)
            
            weights = [1.0] * n_sim + [w_real] * n_real
            sampler = WeightedRandomSampler(weights, num_samples=n_sim + n_real, replacement=True)
            train_loader = DataLoader(hybrid_train_set, batch_size=config.BATCH_SIZE, sampler=sampler)
        else:
            train_loader = DataLoader(hybrid_train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    else:
        # Simulation Only
        train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    # B. Validation Loader 구성
    if config.VAL_SOURCE == 'real' and real_loader is not None:
        print("  -> Using REAL Data for Validation")
        real_dataset = real_loader.get_dataset()
        val_loader = DataLoader(real_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

    # C. Test Loader (이론적 성능 검증용 Sim Data)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==============================================================================
# 3. Experiment Execution (with Logger)
# ==============================================================================

def run_experiment(exp_config, system, data_tuple, normalizer, base_config, real_loader=None):
    # 1. Config 적용 (Global State Update)
    base_config.EXPERIMENT_NAME = exp_config['name']
    base_config.USE_SDE = exp_config['use_sde']
    base_config.SCENARIO = exp_config['scenario']
    base_config.REAL_RATIO = exp_config.get('real_ratio', 0.0)
    base_config.VAL_SOURCE = exp_config.get('val_source', 'sim')
    
    base_config.USE_SPECTRAL_PENALTY = exp_config.get('use_spectral_norm', False)
    base_config.USE_CONSISTENCY_LOSS = exp_config.get('use_consistency_loss', False)
    
    # 2. Logger 초기화
    logger = ExperimentLogger(base_config)
    print(f"\n>>> [Start] {base_config.EXPERIMENT_NAME} (Log: {logger.results_dir}) <<<")

    # 3. Data Loader 준비
    train_loader, val_loader, test_loader = create_dataloaders(
        data_tuple, base_config, real_loader
    )

    # 4. Model 초기화
    obs_idx = system.observed_var_idx
    obs_dim = len(system.t_points) if isinstance(obs_idx, int) else len(system.t_points) * len(obs_idx)
    input_dim = obs_dim * 2 if base_config.USE_LAGRANGIAN else obs_dim 
    
    f_theta = HiddenVarPredictor(
        input_dim=input_dim, 
        param_dim=len(system.param_names),
        output_dim=len(system.t_points),
        config=base_config.MODEL_CONFIG['f_theta']
    )
    g_phi = ParameterEstimator(
        input_dim=input_dim + len(system.t_points),
        output_dim=len(system.param_names),
        config=base_config.MODEL_CONFIG['g_phi']
    )
    
    # 5. 학습 (Training)
    trainer = Trainer(f_theta, g_phi, train_loader, val_loader, base_config, normalizer)
    f_theta, g_phi, history = trainer.train()
    
    # [Logger] 학습 History 저장 (직접 JSON 저장)
    with open(os.path.join(logger.results_dir, 'loss_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # 6. 평가 및 분석 (Analysis)
    print("Analyzing...")
    
    # [Fix] 초기 추측값 설정 (Parameter Initial Guess)
    # Normalizer(Min-Max)의 물리적 중간값을 초기값으로 사용
    p_mins = normalizer.min
    p_maxs = normalizer.max
    p_mid = (p_mins + p_maxs) / 2.0
    p_initial_guess = p_mid.to(base_config.DEVICE)
        
    analyzer = Analyzer(f_theta, g_phi, test_loader, base_config, system, p_initial_guess, normalizer, history)
    
    # Plot 저장 경로를 Logger 경로로 덮어쓰기
    analyzer.results_path = logger.results_dir 
    
    analyzer.plot_loss_curves()
    if base_config.USE_SPECTRAL_PENALTY:
        analyzer.plot_spectral_norms_by_layer()
    
    p_true, p_pred = analyzer.evaluate_predictions()
    analyzer.plot_scatter(p_true, p_pred)
    
    try:
        analyzer.plot_phase_portraits()
    except Exception as e:
        print(f"[Warning] Phase Portrait Failed: {e}")

    # [Phase 5] Real Data Evaluation
    if real_loader is not None:
        split_path = os.path.join('data', 'data_split_indices.json')
        if not os.path.exists(split_path): split_path = 'data_split_indices.json'
        
        if os.path.exists(split_path):
            analyzer.evaluate_real_data(real_loader, split_path)

    # [Logger] 최종 Metrics 계산 및 저장
    metrics = analyzer.compute_summary_metrics(p_true, p_pred, real_data_loader=real_loader)
    
    # ExperimentLogger 활용: CSV Registry 등록
    logger.log_result_to_csv(metrics)
    
    # Metrics JSON 저장
    with open(os.path.join(logger.results_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f">>> [Finished] {base_config.EXPERIMENT_NAME}")


# ==============================================================================
# 4. Main Entry Point
# ==============================================================================

def main():
    args = parse_args()
    if args.epochs: 
        Config.EPOCHS = args.epochs
    set_seed(Config.SEED)
    
    print(f"Initializing System: {Config.SYSTEM_NAME}")
    SystemClass = get_system_class(Config.SYSTEM_NAME)
    system = SystemClass()
    
    # SDE Scaling Injection
    if hasattr(Config, 'SDE_SCALE_FACTORS'):
        if hasattr(system, 'bias_scale'):
            system.bias_scale = Config.SDE_SCALE_FACTORS.get('bias_scale', 1.0)
        if hasattr(system, 'diffusion_scale'):
            system.diffusion_scale = Config.SDE_SCALE_FACTORS.get('diffusion_scale', 1.0)
        print(f"SDE Scaling Applied: Bias={system.bias_scale}, Diff={system.diffusion_scale}")

    # 1. Real Data Loader (Lazy Loading)
    real_loader = None
    need_real = any(e['scenario'] == 'hybrid' or e.get('val_source') == 'real' for e in Config.EXPERIMENTS)
    if args.name:
        target_exp = next((e for e in Config.EXPERIMENTS if e['name'] == args.name), None)
        if target_exp:
            need_real = target_exp['scenario'] == 'hybrid' or target_exp.get('val_source') == 'real'

    if need_real:
        data_path = os.path.join('data', 'clean_sumner_n_612.xlsx')
        if not os.path.exists(data_path): data_path = 'clean_sumner_n_612.xlsx'
        
        if os.path.exists(data_path):
            print(f"Loading Real Data from {data_path}...")
            real_loader = RealOGTTDataLoader(data_path, Config)
        else:
            print("[Warning] Real Data File Not Found. Hybrid/Real-Val will fail.")

    # 2. 실행 대상 실험 리스트 확정
    target_experiments = []
    if args.name:
        base_exp = next((e for e in Config.EXPERIMENTS if e['name'] == args.name), None)
        if not base_exp:
            print(f"[Info] Creating custom experiment config for '{args.name}'")
            custom_exp = {
                'name': args.name,
                'use_sde': str2bool(args.use_sde) if args.use_sde is not None else False,
                'scenario': args.scenario or 'sim_only',
                'real_ratio': args.real_ratio or 0.0,
                'val_source': args.val_source or 'sim',
                'use_spectral_norm': str2bool(args.use_spectral_norm) or False,
                'use_consistency_loss': str2bool(args.use_consistency_loss) or False,
                'use_lagrangian': Config.USE_LAGRANGIAN
            }
            target_experiments.append(custom_exp)
        else:
            exp = base_exp.copy()
            # Override if args provided
            if args.use_sde: exp['use_sde'] = str2bool(args.use_sde)
            if args.scenario: exp['scenario'] = args.scenario
            if args.real_ratio: exp['real_ratio'] = args.real_ratio
            if args.val_source: exp['val_source'] = args.val_source
            if args.use_spectral_norm: exp['use_spectral_norm'] = str2bool(args.use_spectral_norm)
            if args.use_consistency_loss: exp['use_consistency_loss'] = str2bool(args.use_consistency_loss)
            target_experiments.append(exp)
    else:
        target_experiments = Config.EXPERIMENTS

    # 3. Data Caching & Experiment Loop
    data_cache = {}
    
    for exp_config in target_experiments:
        use_sde = exp_config['use_sde']
        use_lag = exp_config.get('use_lagrangian', Config.USE_LAGRANGIAN)
        data_key = (use_sde, use_lag)
        
        if data_key not in data_cache:
            print(f"\n--- [Data Generation] Key: SDE={use_sde}, Lag={use_lag} ---")
            gen_config = copy.deepcopy(Config)
            gen_config.USE_SDE = use_sde
            gen_config.USE_LAGRANGIAN = use_lag
            
            if use_sde:
                print(f"  -> Augmentation Factor: {gen_config.AUGMENTATION_FACTOR}")
            
            data_gen = DataGenerator(system, gen_config)
            data_tuple_norm = data_gen.generate_data() 
            # data_tuple consists of ...
            # (observed_data, hidden_data, params_data, t_points)
            
            # Normalizer 초기화 (Min-Max 방식)
            normalizer = Normalizer(system, Config.DEVICE)
            
            data_cache[data_key] = (data_tuple_norm, normalizer)
            
        data_tuple_norm, normalizer = data_cache[data_key]
        
        try:
            # data_tuple_norm[0]는 (train, val, test) 데이터셋
            run_experiment(exp_config, system, data_tuple_norm[0], normalizer, Config, real_loader)
        except Exception as e:
            print(f"[Error] Experiment {exp_config['name']} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()