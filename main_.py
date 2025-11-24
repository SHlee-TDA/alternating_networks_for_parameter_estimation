# main.py
import os
import random
import numpy as np
import torch
import importlib
import copy
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import Config
from data_loader import DataGenerator, create_dataloaders, RealOGTTDataLoader
from models import HiddenVarPredictor, ParameterEstimator
from trainer import Trainer
from analyzer import Analyzer
from utils import Normalizer, ExperimentLogger  

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
    module_name = f"systems.{system_name}"
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# --- [Phase 5 신규] 데이터셋 병합 및 샘플링 전략 함수 ---
def get_experiment_dataloaders(exp_config, sim_data_tuple, system, global_config):
    # (이전 단계에서 작성한 코드와 동일 - 생략 없이 전체 포함)
    from torch.utils.data import TensorDataset, ConcatDataset, WeightedRandomSampler, Subset
    
    X_sim, Y_sim, P_sim, _ = sim_data_tuple
    
    X_sim_flat = X_sim.reshape(X_sim.shape[0], -1)
    Y_sim_flat = Y_sim.reshape(Y_sim.shape[0], -1)
    
    sim_dataset = TensorDataset(
        torch.tensor(X_sim_flat, dtype=torch.float32),
        torch.tensor(Y_sim_flat, dtype=torch.float32),
        torch.tensor(P_sim, dtype=torch.float32)
    )
    
    real_dataset = None
    scenario = exp_config.get('scenario', 'sim_only')
    val_source = exp_config.get('val_source', 'sim')
    
    if scenario == 'hybrid' or val_source == 'real':
        data_path = 'data/clean_sumner_n_612.xlsx'
        if not os.path.exists(data_path):
             data_path = os.path.join('data', 'clean_sumner_n_612.xlsx')
             
        real_loader = RealOGTTDataLoader(data_path, global_config)
        obs, hid, params, _ = real_loader.load_data()
        
        obs_flat = obs.reshape(obs.shape[0], -1)
        hid_flat = hid.reshape(hid.shape[0], -1)
        
        real_dataset = TensorDataset(
            torch.tensor(obs_flat, dtype=torch.float32),
            torch.tensor(hid_flat, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )
        print(f"  -> Loaded {len(real_dataset)} Real Samples.")

    # Train / Validation / Test Split
    if val_source == 'real' and real_dataset is not None:
        N_real = len(real_dataset)
        val_size = int(0.2 * N_real)
        train_real_size = N_real - val_size
        train_real_ds, val_ds = torch.utils.data.random_split(
            real_dataset, [train_real_size, val_size], 
            generator=torch.Generator().manual_seed(global_config.SEED)
        )
        print(f"  -> Validation Source: Real Data ({len(val_ds)} samples)")
    else:
        N_sim = len(sim_dataset)
        test_size = int(global_config.TEST_SPLIT * N_sim)
        rest_size = N_sim - test_size
        val_size = int(0.1 * rest_size)
        
        indices = torch.randperm(N_sim, generator=torch.Generator().manual_seed(global_config.SEED))
        
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size+val_size]
        train_sim_idx = indices[test_size+val_size:]
        
        val_ds = Subset(sim_dataset, val_idx)
        train_sim_ds = Subset(sim_dataset, train_sim_idx)
        test_ds = Subset(sim_dataset, test_idx) 
        
        train_real_ds = real_dataset if real_dataset is not None else None
        print(f"  -> Validation Source: Simulation Data ({len(val_ds)} samples)")

    if scenario == 'hybrid' and train_real_ds is not None:
        combined_train_ds = ConcatDataset([train_sim_ds, train_real_ds])
        
        real_ratio = exp_config.get('real_ratio', 0.3)
        n_sim = len(train_sim_ds)
        n_real = len(train_real_ds)
        
        weight_real = real_ratio / n_real
        weight_sim = (1 - real_ratio) / n_sim
        
        weights = [weight_sim] * n_sim + [weight_real] * n_real
        sampler = WeightedRandomSampler(weights, num_samples=n_sim + n_real, replacement=True)
        
        train_loader = DataLoader(combined_train_ds, batch_size=global_config.BATCH_SIZE, sampler=sampler)
        print(f"  -> Training Scenario: Hybrid (Sim={n_sim} + Real={n_real}) with Weighted Sampling (r={real_ratio})")
        
    else:
        train_loader = DataLoader(train_sim_ds, batch_size=global_config.BATCH_SIZE, shuffle=True)
        print(f"  -> Training Scenario: Simulation Only ({len(train_sim_ds)} samples)")

    val_loader = DataLoader(val_ds, batch_size=global_config.BATCH_SIZE)
    
    if 'test_ds' not in locals():
         N_sim = len(sim_dataset)
         test_size = int(global_config.TEST_SPLIT * N_sim)
         indices = torch.randperm(N_sim, generator=torch.Generator().manual_seed(global_config.SEED))
         test_ds = Subset(sim_dataset, indices[:test_size])
         
    test_loader = DataLoader(test_ds, batch_size=global_config.BATCH_SIZE)
    
    p_initial_guess = sim_dataset.tensors[2].mean(dim=0).unsqueeze(0).to(global_config.DEVICE)

    return train_loader, val_loader, test_loader, p_initial_guess


def run_experiment(exp_config, system, data_tuple, normalizer, base_config):
    # 1. 설정 준비
    config = copy.deepcopy(base_config)
    config.EXPERIMENT_NAME = exp_config['name']
    config.USE_SPECTRAL_NORM = exp_config['use_spectral_norm']
    config.USE_CONSISTENCY_LOSS = exp_config['use_consistency_loss']
    
    # [수정] Logger 초기화 및 경로 동기화
    # Logger가 timestamp가 포함된 고유 디렉토리를 생성합니다.
    logger = ExperimentLogger(config)
    
    # Trainer와 Analyzer가 Logger가 만든 폴더에 저장하도록 경로를 업데이트합니다.
    # ExperimentLogger.exp_dir_name은 "20251121_..._uuid" 형태입니다.
    config.EXPERIMENT_NAME = logger.exp_dir_name
    
    print(f"\n===== Running Experiment: {exp_config['name']} =====")
    print(f"    Results will be saved to: {logger.results_dir}")
    print(f"    Scenario: {exp_config.get('scenario', 'sim_only')}, Use SDE: {exp_config.get('use_sde', False)}")

    # 2. 데이터 로더 생성
    train_loader, val_loader, test_loader, p_initial_guess = get_experiment_dataloaders(
        exp_config, data_tuple, system, base_config
    )

    # 모델 차원 확인
    try:
        x_sample, y_sample, p_sample = next(iter(train_loader))
    except StopIteration:
        raise ValueError("DataLoader is empty.")

    FLAT_X_DIM = x_sample.shape[1]
    FLAT_Y_DIM = y_sample.shape[1]
    NUM_PARAMS = p_sample.shape[1]

    # 3. 모델 초기화
    model_conf = config.MODEL_CONFIG
    f_theta = HiddenVarPredictor(
        flat_x_dim=FLAT_X_DIM,
        flat_y_dim=FLAT_Y_DIM,
        num_params=NUM_PARAMS,
        model_config=model_conf['f_theta'],
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        initialization_config=model_conf.get('initialization')
    ).to(config.DEVICE)
    
    g_phi = ParameterEstimator(
        flat_x_dim=FLAT_X_DIM,
        flat_y_dim=FLAT_Y_DIM,
        num_params=NUM_PARAMS,
        model_config=model_conf['g_phi'],
        use_spectral_norm=config.USE_SPECTRAL_NORM,
        initialization_config=model_conf.get('initialization')
    ).to(config.DEVICE)
    
    # 4. 학습
    trainer = Trainer(f_theta, g_phi, train_loader, val_loader, config, normalizer)
    f_theta_trained, g_phi_trained, history = trainer.train()
    
    # 5. 분석
    analyzer = Analyzer(
        f_theta_trained, g_phi_trained, test_loader, 
        config, system, p_initial_guess, normalizer, history
    )
    
    analyzer.plot_loss_curves()
    analyzer.analyze_spectral_norms()
    p_true, p_pred = analyzer.evaluate_predictions()
    analyzer.plot_scatter(p_true, p_pred)
    analyzer.plot_phase_portraits()
    
    # 6. [수정] 결과 로깅
    best_val_loss = min(history['val_total_loss']) if 'val_total_loss' in history else -1
    test_mse = np.mean((p_true - p_pred)**2)
    
    metrics = {
        'best_val_loss': best_val_loss,
        'test_param_mse': test_mse,
        'scenario': exp_config.get('scenario', 'sim_only'),
        'real_ratio': exp_config.get('real_ratio', 0.0)
    }
    
    logger.log_result_to_csv(metrics)
    print(f"Experiment finished. Metrics logged: {metrics}")
    
    return history

def main():
    global_config = Config()
    set_seed(global_config.SEED)
    
    system_class = get_system_class(global_config.SYSTEM_NAME)
    system = system_class()
    
    # Normalizer를 위한 임시 시스템 인스턴스 (범위 확인용)
    normalizer = Normalizer(system, global_config.DEVICE)

    # --- Data Caching Logic ---
    required_data_versions = set()
    for exp in global_config.EXPERIMENTS:
        use_sde = exp.get('use_sde', False)
        use_lagrangian = exp.get('use_lagrangian', getattr(global_config, 'USE_LAGRANGIAN', False))
        required_data_versions.add((use_sde, use_lagrangian))
    
    data_cache = {}
    print(f"Required data versions (SDE, Lagrangian): {required_data_versions}")
    
    for (use_sde, use_lagrangian) in required_data_versions:
        print(f"\n--- Generating Data: SDE={use_sde}, Lagrangian={use_lagrangian} ---")
        
        gen_config = copy.deepcopy(global_config)
        gen_config.USE_SDE = use_sde
        gen_config.USE_LAGRANGIAN = use_lagrangian
        
        data_gen = DataGenerator(system, gen_config)
        data_tuple = data_gen.generate_data()
        
        data_cache[(use_sde, use_lagrangian)] = (data_tuple, gen_config)
        print("--- Data Generation Complete ---")

    print(f"\n--- Starting {len(global_config.EXPERIMENTS)} experiments ---")
    
    for exp_config in global_config.EXPERIMENTS:
        req_sde = exp_config.get('use_sde', False)
        req_lagrangian = exp_config.get('use_lagrangian', getattr(global_config, 'USE_LAGRANGIAN', False))
        
        target_key = (req_sde, req_lagrangian)
        
        if target_key not in data_cache:
            print(f"Error: Data version {target_key} not found in cache.")
            continue
            
        data_tuple, base_config = data_cache[target_key]
        
        run_experiment(exp_config, system, data_tuple, normalizer, base_config)

if __name__ == '__main__':
    main()