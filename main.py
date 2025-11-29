import os
import random
import copy
import gc
import json
import importlib
import argparse
import numpy as np
import torch
from collections import defaultdict

# 사용자 모듈 Import
from config import Config
from data_loader import DataGenerator, RealOGTTDataLoader
from models import HiddenVarPredictor, ParameterEstimator
from trainer import Trainer
from analyzer import Analyzer
from utils import Normalizer, ExperimentLogger  # [핵심] Logger Import
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset, WeightedRandomSampler

# ==============================================================================
# 1. 유틸리티 함수
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
    module_name = f"systems.{system_name}"
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def get_experiment_dataloaders(exp_config, sim_data_tuple, system, global_config):
    """
    데이터 로더 생성 함수 (수정됨)
    - Train: Sim (Hybrid일 경우 Real Mix)
    - Val: Sim
    - Test: Sim (Default Analyzer용) -> Real Test는 Analyzer.evaluate_real_data에서 별도 수행
    """
    
    # --- 1. Simulation Data 준비 ---
    obs_sim, hid_sim, params_sim, t_points = sim_data_tuple
    
    # Tensor 변환 & Flatten
    X_sim = torch.FloatTensor(obs_sim).view(len(obs_sim), -1)
    Y_sim = torch.FloatTensor(hid_sim).view(len(hid_sim), -1)
    P_sim = torch.FloatTensor(params_sim)
    
    sim_dataset = TensorDataset(X_sim, Y_sim, P_sim)
    
    # [수정] 시뮬레이션 데이터 3분할 (Train / Val / Test)
    total_len = len(sim_dataset)
    test_len = int(total_len * global_config.TEST_SPLIT) # 예: 20%
    val_len = int(total_len * global_config.TEST_SPLIT)  # 예: 20%
    train_len = total_len - val_len - test_len            # 나머지 60%
    
    sim_train, sim_val, sim_test = random_split(
        sim_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(global_config.SEED)
    )

    # --- 2. Real Data 준비 (Hybrid 학습용) ---
    real_data_path = 'data/clean_sumner_n_612.xlsx' 
    split_index_path = 'data/data_split_indices.json'
    
    real_loader = RealOGTTDataLoader(
        file_path=real_data_path, 
        config=global_config,
        split_file=split_index_path 
    )
    # 학습에 섞을 Real Train 데이터만 가져옴
    real_train_dataset, _ = real_loader.get_train_test_datasets()


    # --- 3. 시나리오별 데이터 조합 ---
    scenario = exp_config.get('scenario', 'sim_only')
    
    # A. Train Loader 구성
    if scenario == 'hybrid':
        print(f"  -> [Hybrid] Merging Sim Train ({len(sim_train)}) + Real Train ({len(real_train_dataset)})")
        
        final_train_dataset = ConcatDataset([sim_train, real_train_dataset])
        
        # Weighted Sampling
        real_ratio = exp_config.get('real_ratio', 0.3)
        n_sim = len(sim_train)
        n_real = len(real_train_dataset)
        
        weight_sim = (1 - real_ratio) / n_sim
        weight_real = real_ratio / n_real
        
        weights = [weight_sim] * n_sim + [weight_real] * n_real
        sampler = WeightedRandomSampler(weights, num_samples=len(final_train_dataset), replacement=True)
        
        train_loader = DataLoader(final_train_dataset, batch_size=global_config.BATCH_SIZE, sampler=sampler, drop_last=True)
        
    else: # sim_only
        print(f"  -> [Sim Only] Using Sim Train ({len(sim_train)})")
        train_loader = DataLoader(sim_train, batch_size=global_config.BATCH_SIZE, shuffle=True, drop_last=True)

    # B. Validation Loader (항상 Sim Val)
    val_loader = DataLoader(sim_val, batch_size=global_config.BATCH_SIZE, shuffle=False)
    
    # C. [수정] Test Loader (Sim Test로 복구)
    # Real Test는 main.py 하단에서 evaluate_real_data()를 통해 별도로 평가됨
    print(f"  -> [Test] Standard Evaluation on Sim Test Set ({len(sim_test)})")
    test_loader = DataLoader(sim_test, batch_size=global_config.BATCH_SIZE, shuffle=False)
    
    # 초기값 추정용 (Sim 데이터 평균)
    p_initial_guess = P_sim.mean(dim=0)
        
    return train_loader, val_loader, test_loader, p_initial_guess


def merge_sim_data(tuple_ode, tuple_sde):
    """ODE 데이터와 SDE 데이터를 하나로 합칩니다."""
    obs_o, hid_o, par_o, t_o = tuple_ode
    obs_s, hid_s, par_s, t_s = tuple_sde
    
    # 시간축이 같은지 안전장치 확인
    # assert np.allclose(t_o, t_s), "Time points mismatch between ODE and SDE data!"
    
    # 배치 차원(axis 0)을 기준으로 병합
    obs_mix = np.concatenate([obs_o, obs_s], axis=0)
    hid_mix = np.concatenate([hid_o, hid_s], axis=0)
    par_mix = np.concatenate([par_o, par_s], axis=0)
    
    return obs_mix, hid_mix, par_mix, t_o
# ==============================================================================
# 2. 메인 파이프라인
# ==============================================================================
def run_experiment_pipeline(global_config):
    set_seed(global_config.SEED)
    device = torch.device(global_config.DEVICE)
    print(f"System: {global_config.SYSTEM_NAME} | Device: {device}")
    
    SystemClass = get_system_class(global_config.SYSTEM_NAME)
    system = SystemClass()
    normalizer = Normalizer(system, global_config.DEVICE)
    
    # [Phase 1] Data Caching
    required_data_configs = set()
    for exp in global_config.EXPERIMENTS:
        use_sde_setting = exp.get('use_sde', False)
        use_lag = exp.get('use_lagrangian', getattr(global_config, 'USE_LAGRANGIAN', False))
        
        # [수정] 'mixed' 모드면 ODE(False)와 SDE(True) 둘 다 필요함
        if use_sde_setting == 'mixed':
            required_data_configs.add((False, use_lag))
            required_data_configs.add((True, use_lag))
        else:
            required_data_configs.add((use_sde_setting, use_lag))
    
    data_cache = {}
    print(f"\n[Phase 1] Pre-generating data for: {required_data_configs}")
    for (use_sde, use_lag) in required_data_configs:
        gen_config = copy.deepcopy(global_config)
        gen_config.USE_SDE = use_sde
        gen_config.USE_LAGRANGIAN = use_lag
        generator = DataGenerator(system, gen_config)
        data_cache[(use_sde, use_lag)] = generator.generate_data()

    # [Phase 2] Experiment Loop
    total_exps = len(global_config.EXPERIMENTS)
    print(f"\n[Phase 2] Starting {total_exps} Experiments...")
    
    for idx, exp_config in enumerate(global_config.EXPERIMENTS):
        exp_name = exp_config['name']
        print(f"\n{'='*60}")
        print(f"Experiment {idx+1}/{total_exps}: {exp_name}")
        
        # 1. Config 준비
        current_run_config = copy.deepcopy(global_config)
        current_run_config.EXPERIMENT_NAME = exp_name
        current_run_config.USE_SPECTRAL_NORM = exp_config['use_spectral_norm']
        current_run_config.USE_CONSISTENCY_LOSS = exp_config['use_consistency_loss']
        current_run_config.MODEL_CONFIG = getattr(global_config, 'MODEL_CONFIG', {})
        # Config에 현재 실험의 특수 속성 추가 (Log용)
        current_run_config.USE_SDE = exp_config.get('use_sde', False)
        
        # 2. [핵심] ExperimentLogger 초기화
        # 여기서 timestamp가 포함된 고유 폴더(results_dir)가 생성되고 config.json이 저장됩니다.
        logger = ExperimentLogger(current_run_config)
        print(f"  -> Log Directory: {logger.results_dir}")
        
        # 3. [트릭] Trainer가 Logger의 경로를 사용하도록 Config 조작
        # Trainer.py를 수정하지 않고 경로를 주입하기 위해, 
        # RESULTS_DIR을 logger.results_dir로 바꾸고 하위 경로 생성을 무력화합니다.
        trainer_config = copy.deepcopy(current_run_config)
        trainer_config.RESULTS_DIR = logger.results_dir
        trainer_config.SYSTEM_NAME = ""     # 이미 경로에 포함됨
        trainer_config.EXPERIMENT_NAME = "" # 이미 경로에 포함됨
        
        # 4. 데이터 로드
        req_sde_setting = exp_config.get('use_sde', False)
        req_lag = exp_config.get('use_lagrangian', getattr(global_config, 'USE_LAGRANGIAN', False))
    
        if req_sde_setting == 'mixed':
            # [핵심] ODE와 SDE 데이터를 모두 꺼내서 합침
            ode_tuple = data_cache[(False, req_lag)]
            sde_tuple = data_cache[(True, req_lag)]
            sim_data_tuple = merge_sim_data(ode_tuple, sde_tuple)
            print(f"  -> [Data] Mixed ODE + SDE Data Loaded (Total: {len(sim_data_tuple[0])} samples)")
            
            # Config에 기록용으로 남길 때는 SDE를 썼다는 흔적을 남김 (또는 별도 표기)
            current_run_config.USE_SDE = True 
        else:
            sim_data_tuple = data_cache[(req_sde_setting, req_lag)]
            current_run_config.USE_SDE = req_sde_setting
            
        train_loader, val_loader, test_loader, p_initial_guess = get_experiment_dataloaders(
            exp_config, sim_data_tuple, system, global_config
        )
        
        # Normalizer setup
        # sim_data_tuple = (obs, hid, params, t)
        obs_all = sim_data_tuple[0] # (N, T, n_obs)
        hid_all = sim_data_tuple[1] # (N, T, n_hid)
        params_all = sim_data_tuple[2] # (N, n_params)
        
        # 99.9% Percentile로 최대 범위 계산
        scale_obs = np.percentile(np.abs(obs_all), 99.9, axis=(0, 1))
        scale_hid = np.percentile(np.abs(hid_all), 99.9, axis=(0, 1))
        
        # [obs_scale, hid_scale] 순서로 연결 (Flatten 대비)
        # obs_all이 (N, T, 1)이면 scale_obs는 스칼라일 수 있으므로 배열로 변환 확인
        if np.ndim(scale_obs) == 0: scale_obs = [scale_obs]
        if np.ndim(scale_hid) == 0: scale_hid = [scale_hid]
            
        calculated_state_scales = np.concatenate([scale_obs, scale_hid]).tolist()
        calculated_state_scales = [s * 1.2 for s in calculated_state_scales] # 20% 여유
        
        print(f"  -> Data-Driven State Scales: {calculated_state_scales}")
        
        # [추가] 2. Parameter Bounds 계산 (Min/Max)
        # 실제 데이터의 최소/최대를 구해서 타이트한 범위를 만듭니다.
        p_mins = np.min(params_all, axis=0)
        p_maxs = np.max(params_all, axis=0)
        
        # 너무 딱 맞으면 경계값 문제가 생길 수 있으므로 10% 정도 여유를 둡니다.
        # (범위 = max - min)
        p_ranges = p_maxs - p_mins
        p_mins_safe = p_mins - 0.1 * p_ranges
        p_maxs_safe = p_maxs + 0.1 * p_ranges
        
        # 음수가 될 수 없는 파라미터(si, sigma 등)라면 0으로 클리핑 (선택 사항)
        p_mins_safe = np.maximum(p_mins_safe, 0.0)
        
        calculated_param_bounds = (p_mins_safe.tolist(), p_maxs_safe.tolist())
        
        print(f"  -> Data-Driven State Scales: {calculated_state_scales}")
        print(f"  -> Data-Driven Param Bounds: {calculated_param_bounds}")
        
        # Normalizer에 스케일 주입
        normalizer = Normalizer(
            system, 
            global_config.DEVICE, 
            state_scales=calculated_state_scales,
            param_bounds=calculated_param_bounds
        )
        
        # 5. 모델 초기화
        sample_x, sample_y, sample_p = next(iter(train_loader))
        f_theta = HiddenVarPredictor(
            sample_x.shape[1], sample_y.shape[1], sample_p.shape[1],
            model_config=current_run_config.MODEL_CONFIG['f_theta'],
            use_spectral_norm=current_run_config.USE_SPECTRAL_NORM,
            initialization_config=current_run_config.MODEL_CONFIG.get('initialization')
        ).to(device)
        
        g_phi = ParameterEstimator(
            sample_x.shape[1], sample_y.shape[1], sample_p.shape[1],
            model_config=current_run_config.MODEL_CONFIG['g_phi'],
            use_spectral_norm=current_run_config.USE_SPECTRAL_NORM,
            initialization_config=current_run_config.MODEL_CONFIG.get('initialization')
        ).to(device)
        
        # 6. 학습 (Trainer)
        # trainer_config를 넘겨주어 결과가 logger.results_dir에 저장되게 함
        trainer = Trainer(f_theta, g_phi, train_loader, val_loader, trainer_config, normalizer)
        f_theta, g_phi, history = trainer.train()
        
        # Loss History 추가 저장 (Trainer가 best_model 등을 저장했지만 history는 json으로 명시적 저장)
        with open(os.path.join(logger.results_dir, 'loss_history.json'), 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=4)

        # 7. 분석 (Analyzer)
        # Analyzer는 내부적으로 savefig를 하므로, trainer_config(경로 수정된 것)를 넘겨줍니다.
        p_initial_guess = p_initial_guess.to(device) 

        analyzer = Analyzer(
            f_theta, g_phi, test_loader, trainer_config, 
            system, p_initial_guess, normalizer, history
        )
        
        analyzer.plot_loss_curves()
        analyzer.plot_phase_portraits()
        p_true, p_pred = analyzer.evaluate_predictions()
        analyzer.plot_scatter(p_true, p_pred)
        
        # 예측값 저장
        np.savez(os.path.join(logger.results_dir, 'predictions.npz'), p_true=p_true, p_pred=p_pred)
        # [추가] Real Data 전용 심층 평가 실행 (Scatter + Reconstruction)
        print("  -> Running specialized evaluation on Real Data...")
        
        # Real Data Loader 재생성 (Analyzer에 넘겨주기 위함)
        real_eval_loader = RealOGTTDataLoader(
            file_path='data/clean_sumner_n_612.xlsx', 
            config=global_config,
            split_file='data/data_split_indices.json'
        )
        
        # split_file 경로도 함께 전달
        analyzer.evaluate_real_data(
            real_data_loader=real_eval_loader,
            split_file_path='data/data_split_indices.json',
            num_vis=5 # 재구성 시각화할 환자 수
        )
        if current_run_config.USE_SPECTRAL_NORM:
            analyzer.analyze_spectral_norms()
            analyzer.plot_spectral_norms_by_layer()
            
        # 8. [핵심] 결과 요약 및 CSV 등록
        # 최종 성능 지표 계산
        final_train_loss = history['train_total_loss'][-1]
        final_val_loss = history['val_total_loss'][-1]
        
        # Test Set 성능 (MSE)
        mse_loss = np.mean((p_true - p_pred)**2)
        
        metrics = {
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'test_mse': mse_loss,
            'epoch': len(history['train_total_loss'])
        }
        
        # CSV 레지스트리에 한 줄 추가
        logger.log_result_to_csv(metrics)
        print(f"  -> Metrics logged to registry: {metrics}")

        # Cleanup
        del f_theta, g_phi, trainer, analyzer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    if args.epochs: config.EPOCHS = args.epochs
        
    run_experiment_pipeline(config)