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
    - Sim과 Real 데이터 모두 동일한 Normalizer로 전처리 수행
    - Train: Sim (Hybrid일 경우 Real Mix)
    - Val: Sim
    - Test: Sim (Default)
    """
    import json
    from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset, WeightedRandomSampler, Subset

    # --- 1. Simulation Data 준비 및 Normalizer 초기화 ---
    obs_sim, hid_sim, params_sim, t_points = sim_data_tuple
    
    # [Data-Driven Normalizer 초기화]
    # 학습 데이터(Sim)의 실제 분포를 측정하여 정규화 기준으로 삼습니다.
    
    # A. 상태 변수 스케일 (99.9% Percentile)
    scale_obs = np.percentile(np.abs(obs_sim), 99.9)
    scale_hid = np.percentile(np.abs(hid_sim), 99.9)
    calc_scales = [scale_obs * 1.2, scale_hid * 1.2] # 20% 여유
    
    # B. 파라미터 범위 (Min/Max)
    p_min = np.min(params_sim, axis=0)
    p_max = np.max(params_sim, axis=0)
    #p_range = p_max - p_min
    #p_bounds = (p_min - 0.1 * p_range, p_max + 0.1 * p_range) # 10% 여유
    
    p_bounds_min = p_min / 1.2
    p_bounds_max = p_max * 1.2
    
    # Multiplicative Margin (로그 스케일에 적합)
    p_bounds = (p_bounds_min, p_bounds_max)
    
    print(f"  [Normalizer Init] Scale: {calc_scales}")
    print(f"  [Normalizer Init] P-Bounds: \n    Min: {p_bounds[0]}\n    Max: {p_bounds[1]}")
    
    # Normalizer 생성 (이 객체는 나중에 Analyzer에 전달됨)
    normalizer = Normalizer(
        system, 
        global_config.DEVICE, 
        state_scales=calc_scales, 
        param_bounds=p_bounds,
        use_log_params=True  
    )
    
    # --- 2. Simulation Data 정규화 및 데이터셋 생성 ---
    # Numpy -> Tensor 변환 (Flatten)
    X_sim = torch.FloatTensor(obs_sim).view(len(obs_sim), -1).to(global_config.DEVICE)
    Y_sim = torch.FloatTensor(hid_sim).view(len(hid_sim), -1).to(global_config.DEVICE)
    P_sim = torch.FloatTensor(params_sim).to(global_config.DEVICE)
    
    # 정규화 수행
    X_sim_norm = normalizer.normalize_inputs(X_sim, variable_type='observed')
    Y_sim_norm = normalizer.normalize_inputs(Y_sim, variable_type='hidden')
    P_sim_norm = normalizer.normalize_params(P_sim)
    
    sim_dataset = TensorDataset(X_sim_norm, Y_sim_norm, P_sim_norm)    
    
    # Sim Data Split (Train / Val / Test)
    total_len = len(sim_dataset)
    test_len = int(total_len * global_config.TEST_SPLIT)
    val_len = int(total_len * global_config.TEST_SPLIT)
    train_len = total_len - val_len - test_len
    
    sim_train, sim_val, sim_test = random_split(
        sim_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(global_config.SEED)
    )

    # --- 3. Real Data 로드 및 정규화 (Hybrid 학습용) ---
    real_data_path = 'data/clean_sumner_n_612.xlsx' 
    split_index_path = 'data/data_split_indices.json'
    
    # Real Data Loader 생성 (Raw Data 로드용)
    real_loader = RealOGTTDataLoader(
        file_path=real_data_path, 
        config=global_config,
        split_file=split_index_path 
    )
    
    # Raw Data 로드 (Numpy)
    X_real, Y_real, P_real, _ = real_loader.load_data()
    
    # Tensor 변환 & Device 이동
    X_real_t = torch.FloatTensor(X_real).view(len(X_real), -1).to(global_config.DEVICE)
    Y_real_t = torch.FloatTensor(Y_real).view(len(Y_real), -1).to(global_config.DEVICE)
    P_real_t = torch.FloatTensor(P_real).to(global_config.DEVICE)
    
    # [핵심] Sim 데이터로 만든 Normalizer를 사용해 Real 데이터 정규화
    X_real_norm = normalizer.normalize_inputs(X_real_t, variable_type='observed')
    Y_real_norm = normalizer.normalize_inputs(Y_real_t, variable_type='hidden')
    P_real_norm = normalizer.normalize_params(P_real_t)
    
    # Normalized Real Dataset 생성
    real_dataset_norm = TensorDataset(X_real_norm, Y_real_norm, P_real_norm)
    
    # Split Indices 적용 (파일에서 로드)
    with open(split_index_path, 'r') as f:
        split_data = json.load(f)
        train_indices = split_data['train_indices']
        # test_indices = split_data['test_indices'] # 학습에는 Train만 사용
        
    real_train_dataset = Subset(real_dataset_norm, train_indices)
    

    # --- 4. 시나리오별 DataLoader 조합 ---
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
    
    # C. Test Loader (Sim Test)
    # Real Test는 main.py 하단에서 evaluate_real_data()를 통해 별도로 평가됨
    print(f"  -> [Test] Standard Evaluation on Sim Test Set ({len(sim_test)})")
    test_loader = DataLoader(sim_test, batch_size=global_config.BATCH_SIZE, shuffle=False)
    
    # 초기값 추정용 (Sim 데이터 평균 - 정규화 전의 P_sim 사용 권장하나, 여기서는 편의상 P_sim 사용)
    # Trainer 내부에서 정규화하여 사용할 것이므로 Raw Value 전달
    #p_initial_guess = P_sim.mean(dim=0)
    # 로그 공간에서의 평균 사용 (기하 평균)
    P_sim_safe = torch.maximum(P_sim, torch.tensor(1e-6).to(global_config.DEVICE))
    p_initial_guess = torch.exp(torch.log(P_sim_safe).mean(dim=0))
        
    # [중요] 생성된 normalizer 객체를 반환해야 함
    return train_loader, val_loader, test_loader, p_initial_guess, normalizer


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
            
        train_loader, val_loader, test_loader, p_initial_guess, normalizer = get_experiment_dataloaders(
            exp_config, sim_data_tuple, system, global_config
        )
        
        real_raw_loader = RealOGTTDataLoader(
            file_path='data/clean_sumner_n_612.xlsx', 
            config=global_config,
            split_file='data/data_split_indices.json'
        )
        
        X_real_np, Y_real_np, P_real_np, t_points = real_raw_loader.load_data()
        import json
        with open('data/data_split_indices.json', 'r') as f:
            split_data = json.load(f)
            test_indices = split_data['test_indices']
            
        # Numpy Slicing (Test Set 추출)
        X_test_np = X_real_np[test_indices]
        Y_test_np = Y_real_np[test_indices]
        P_test_np = P_real_np[test_indices]
        
        # Tensor 변환 & Flatten (Batch, Time * Dim)
        # 모델 입력 차원에 맞게 (N, -1)로 펼쳐줍니다.
        N_test = len(X_test_np)
        X_real_t = torch.tensor(X_test_np, dtype=torch.float32).view(N_test, -1).to(global_config.DEVICE)
        Y_real_t = torch.tensor(Y_test_np, dtype=torch.float32).view(N_test, -1).to(global_config.DEVICE)
        P_real_t = torch.tensor(P_test_np, dtype=torch.float32).to(global_config.DEVICE)
        
        # [핵심] Normalizer를 사용해 전처리 수행 (Glucose, Insulin, Params 모두 적용)
        # Sim 데이터와 동일한 스케일(예: /100)로 변환됩니다.
        X_real_norm = normalizer.normalize_inputs(X_real_t, variable_type='observed')
        Y_real_norm = normalizer.normalize_inputs(Y_real_t, variable_type='hidden')
        P_real_norm = normalizer.normalize_params(P_real_t)
        
        # 정규화된 데이터셋 및 로더 생성
        real_test_dataset = TensorDataset(X_real_norm, Y_real_norm, P_real_norm)
        real_test_loader = DataLoader(real_test_dataset, batch_size=global_config.BATCH_SIZE, shuffle=False)
        
        # -------------------------------------------------------------------------
        # 3. Analyzer 생성 및 실행
        # -------------------------------------------------------------------------
        # Analyzer 초기화 (Sim Test Loader 사용)
        
        # # Normalizer setup
        # # sim_data_tuple = (obs, hid, params, t)
        # obs_all = sim_data_tuple[0] # (N, T, n_obs)
        # hid_all = sim_data_tuple[1] # (N, T, n_hid)
        # params_all = sim_data_tuple[2] # (N, n_params)
        
        # # 99.9% Percentile로 최대 범위 계산
        # scale_obs = np.percentile(np.abs(obs_all), 99.9, axis=(0, 1))
        # scale_hid = np.percentile(np.abs(hid_all), 99.9, axis=(0, 1))
        
        # # [obs_scale, hid_scale] 순서로 연결 (Flatten 대비)
        # # obs_all이 (N, T, 1)이면 scale_obs는 스칼라일 수 있으므로 배열로 변환 확인
        # if np.ndim(scale_obs) == 0: scale_obs = [scale_obs]
        # if np.ndim(scale_hid) == 0: scale_hid = [scale_hid]
            
        # calculated_state_scales = np.concatenate([scale_obs, scale_hid]).tolist()
        # calculated_state_scales = [s * 1.2 for s in calculated_state_scales] # 20% 여유
        
        # print(f"  -> Data-Driven State Scales: {calculated_state_scales}")
        
        # # [추가] 2. Parameter Bounds 계산 (Min/Max)
        # # 실제 데이터의 최소/최대를 구해서 타이트한 범위를 만듭니다.
        # p_mins = np.min(params_all, axis=0)
        # p_maxs = np.max(params_all, axis=0)
        
        # # 너무 딱 맞으면 경계값 문제가 생길 수 있으므로 10% 정도 여유를 둡니다.
        # # (범위 = max - min)
        # p_ranges = p_maxs - p_mins
        # p_mins_safe = p_mins - 0.1 * p_ranges
        # p_maxs_safe = p_maxs + 0.1 * p_ranges
        
        # # 음수가 될 수 없는 파라미터(si, sigma 등)라면 0으로 클리핑 (선택 사항)
        # p_mins_safe = np.maximum(p_mins_safe, 0.0)
        
        # calculated_param_bounds = (p_mins_safe.tolist(), p_maxs_safe.tolist())
        
        # print(f"  -> Data-Driven State Scales: {calculated_state_scales}")
        # print(f"  -> Data-Driven Param Bounds: {calculated_param_bounds}")
        
        # # Normalizer에 스케일 주입
        # normalizer = Normalizer(
        #     system, 
        #     global_config.DEVICE, 
        #     state_scales=calculated_state_scales,
        #     param_bounds=calculated_param_bounds
        # )
        
        # 5. 모델 초기화
        sample_x, sample_y, sample_p = next(iter(train_loader))
        f_theta = HiddenVarPredictor(
            sample_x.shape[1], sample_y.shape[1], sample_p.shape[1],
            model_config=current_run_config.MODEL_CONFIG['f_theta'],
            use_spectral_norm=current_run_config.USE_SPECTRAL_NORM,
            #initialization_config=current_run_config.MODEL_CONFIG.get('initialization')
        ).to(device)
        
        g_phi = ParameterEstimator(
            sample_x.shape[1], sample_y.shape[1], sample_p.shape[1],
            model_config=current_run_config.MODEL_CONFIG['g_phi'],
            use_spectral_norm=current_run_config.USE_SPECTRAL_NORM,
            #initialization_config=current_run_config.MODEL_CONFIG.get('initialization')
        ).to(device)
        
        # 6. 학습 (Trainer)
        # 2. 파라미터별 분포 통계 출력
        param_names = system.param_names
        for i, name in enumerate(param_names):
            p_vals = sample_p[:, i]
            print(f"Param '{name}' (Normalized):")
            print(f"  - Mean: {p_vals.mean().item():.4f}")
            print(f"  - Median: {p_vals.median().item():.4f}")
            print(f"  - Min / Max: {p_vals.min().item():.4f} / {p_vals.max().item():.4f}")
            
            # 평균이 -1에 가깝다면(예: -0.8 이하) 분포 왜곡 문제입니다.
            if p_vals.mean() < -0.5:
                print(f"  ⚠️ WARNING: Distribution is highly skewed towards -1.0!")
        # trainer_config를 넘겨주어 결과가 logger.results_dir에 저장되게 함
        trainer = Trainer(f_theta, g_phi, train_loader, val_loader, trainer_config)
        f_theta, g_phi, history = trainer.train()
        
        # Loss History 추가 저장 (Trainer가 best_model 등을 저장했지만 history는 json으로 명시적 저장)
        with open(os.path.join(logger.results_dir, 'loss_history.json'), 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=4)


        analyzer = Analyzer(
            f_theta, g_phi, test_loader, trainer_config, 
            system, p_initial_guess, normalizer, history
        )

        # 기본 분석 (Sim Data)
        analyzer.plot_loss_curves()
        analyzer.plot_phase_portraits()
        p_true, p_pred = analyzer.evaluate_predictions()
        analyzer.plot_scatter(p_true, p_pred)
        
        # [수정] Real Data 분석 (위에서 만든 정규화된 로더 전달)
        print("  -> Running specialized evaluation on Real Data...")
        analyzer.evaluate_real_data(
            real_test_loader=real_test_loader,
            num_vis=5 
        )
        # 예측값 저장
        np.savez(os.path.join(logger.results_dir, 'predictions.npz'), p_true=p_true, p_pred=p_pred)
        # [추가] Real Data 전용 심층 평가 실행 (Scatter + Reconstruction)
        print("  -> Running specialized evaluation on Real Data...")
        
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