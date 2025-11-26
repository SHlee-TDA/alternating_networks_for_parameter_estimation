# data_loader.py
from concurrent.futures import ProcessPoolExecutor
import os
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

from utils import euler_maruyama, get_derivative_estimator
from systems.ogtt_simul import OgttSimul


# --- Helper Functions ---

# External function for parallel data generation
def _generate_one_sample(args):
    # [수정] 인자에 aug_factor 추가
    system, config, seed, dist_params, bias_scale, diffusion_scale, aug_factor = args 
    
    np.random.seed(seed)
    
    # 1. Sampling (한 번만 수행 -> 파라미터 고정)
    if dist_params is not None:
        si = sample_from_lognorm(dist_params['si'])
        sigma_p = sample_from_lognorm(dist_params['sigma'])
        params_list = [si, sigma_p]
        
        g0 = sample_from_lognorm(dist_params['G0'])
        i0 = sample_from_lognorm(dist_params['I0'])
        
        from systems.ogtt_simul import OGTTModel, ode_params, sys_params
        temp_model = OGTTModel(ode_params, sys_params, {'si': si, 'sigma': sigma_p})
        n5, n6 = temp_model.find_steady_state_N(g0)
        y0 = [g0, i0, n5, n6]
    else:
        params_dict = {k: np.random.uniform(*v) for k, v in system.param_ranges.items()}
        params_list = [params_dict[p] for p in system.param_names]
        y0 = system.sample_initial_conditions(params_dict)

    n_vars = len(y0)
    
    # 2. Simulation Loop (Augmentation)
    results_obs = []
    results_hid = []
    
    # SDE 모드가 아니면 증강은 의미가 없으므로 1회만 수행
    actual_aug_factor = aug_factor if getattr(config, 'USE_SDE', False) else 1
    
    sys_instance = system() if isinstance(system, type) else system
    sys_instance.bias_scale = bias_scale
    sys_instance.diffusion_scale = diffusion_scale # Scale 주입
    
    for k in range(actual_aug_factor):
        # 각 반복마다 다른 노이즈가 생성되도록 seed 관리 (Global seed는 위에서 설정됨)
        # euler_maruyama 내부에서 np.random을 쓰므로, 루프만 돌리면 다른 궤적이 나옴
        
        if getattr(config, 'USE_SDE', False):
            y_full = euler_maruyama(
                sys_instance.drift_func,
                sys_instance.diffusion_func,
                sys_instance.t_span,
                y0,
                sys_instance.t_points,
                params_list,
                dt_sim=0.01, # [중요] 고해상도 유지
                system=sys_instance
            )
        else:
            sol = solve_ivp(
                fun=lambda t, y: system.ode_func(t, y, params_list),
                t_span=system.t_span,
                y0=y0,
                t_eval=system.t_points 
            )
            y_full = sol.y if sol.success else np.tile(sol.y[:, -1][:, None], (1, len(system.t_points)))

        # Lagrangian Feature
        if getattr(config, 'USE_LAGRANGIAN', False):
            t_points = sys_instance.t_points
            T = len(t_points)
            y_dot_full = np.zeros_like(y_full)
            for i in range(T):
                t_i = t_points[i]
                y_i = y_full[:, i]
                y_dot_full[:, i] = sys_instance.drift_func(t_i, y_i, params_list)
            y_full = np.concatenate([y_full, y_dot_full], axis=0)

        # Formatting
        y_full_T = y_full.T
        obs_idx = [sys_instance.observed_var_idx]
        hid_idx = [sys_instance.hidden_var_idx]

        if getattr(config, 'USE_LAGRANGIAN', False):
            obs_deriv_idx = [idx + n_vars for idx in obs_idx]
            obs_idx += obs_deriv_idx
        
        X_obs = y_full_T[:, obs_idx]
        Y_hid = y_full_T[:, hid_idx]
        
        results_obs.append(X_obs)
        results_hid.append(Y_hid)
    
    # 리스트 반환 (DataGenerator에서 풀어서 저장)
    # params_list는 고정이므로 하나만 반환해도 되지만, 데이터 짝을 맞추기 위해 복제해서 반환
    return results_obs, results_hid, [params_list] * actual_aug_factor
    
def sample_from_lognorm(dist_params, size=1, max_retries=100):
    """
    scipy.stats.lognorm에서 샘플링하되, 0 이하의 값이 나오면 Rejection Sampling 수행.
    dist_params: {'s': ..., 'loc': ..., 'scale': ...}
    """
    s = dist_params['s']
    loc = dist_params['loc']
    scale = dist_params['scale']

    samples = np.zeros(size)
    remaining_indices = np.arange(size)

    for _ in range(max_retries):
        if len(remaining_indices) == 0:
            break

        current_n = len(remaining_indices)
        # Scipy rvs returns samples in ORIGINAL scale (already shifted by loc)
        new_samples = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=current_n)
        
        # Check positivity
        valid_mask = new_samples > 1e-6 # 0보다 커야 함 (안전장치 1e-6)
        
        # Assign valid samples
        valid_indices = remaining_indices[valid_mask]
        samples[valid_indices] = new_samples[valid_mask]
        
        # Update remaining
        remaining_indices = remaining_indices[~valid_mask]
        
    if len(remaining_indices) > 0:
        # Fallback for extremely rare cases: force absolute value or mean
        # print(f"Warning: {len(remaining_indices)} samples failed rejection sampling. Using absolute values.")
        # Force positive by taking absolute or using mean (scale * exp(s^2/2) is roughly mean for loc=0)
        samples[remaining_indices] = np.abs(stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=len(remaining_indices))) + 1e-6
        
    return samples if size > 1 else samples[0]

class DataGenerator:
    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.dist_params = None

        # Data-Driven Sampling용 분포 파라미터 로드
        dist_file = Path('data/parameters/distribution_params.json')
        if dist_file.exists():
            with open(dist_file, 'r') as f:
                self.dist_params = json.load(f)
            print("Loaded data-driven distribution parameters for sampling.")
        else:
            print("Data-driven distribution parameters file not found. Using uniform sampling.")

    def generate_data(self):
        #scale_val = getattr(self.config, 'DIFFUSION_SCALE', 'Not Found')
        #print(f"[DEBUG] Config DIFFUSION_SCALE: {scale_val}")
        # Data Save
        data_root = Path('data')
        save_dir = data_root / self.config.SYSTEM_NAME
        os.makedirs(save_dir, exist_ok=True)

        suffix = "sde" if getattr(self.config, 'USE_SDE', False) else "ode"
        filename = f"augmented_data_{suffix}_{self.config.NUM_SAMPLES}.npz"
        save_path = save_dir / filename

        if save_path.exists():
            print(f"Loading existing data from {save_path}...")
            try:
                with np.load(save_path) as data:
                    observed_data = data['observed_data']
                    hidden_data = data['hidden']
                    params_data = data['params']
                    t_points = data['t_points']
                print(f"Loaded {len(observed_data)} samples")
                return observed_data, hidden_data, params_data, t_points
            except Exception as e:
                print(f"Failed to load existing data: {e}. Regenerating data...")

        # Generating Data
        print(f"Generating {self.config.NUM_SAMPLES} samples using {suffix.upper()} model...")
        num_samples = self.config.NUM_SAMPLES
        t_points = np.asarray(self.system.t_points)
        
        # SDE scaling 
        scale_factor = getattr(self.config, 'SDE_SCALE_FACTORS', {'bias_scale': 1.0, 'diffusion_scale': 1.0})
        
        bias_scale = getattr(scale_factor, 'bias_scale', 1.0)
        diffusion_scale = getattr(scale_factor, 'diffusion_scale', 1.0)
        
        self.system.bias_scale = bias_scale
        self.system.diffusion_scale = diffusion_scale
        print(f"Applied SDE Scaling: {scale_factor}")
        
        
        # Augmentation Factor
        aug_factor = getattr(self.config, 'AUGMENTATION_FACTOR', 1)        # 각 작업에 전달할 고유한 시드 생성
        print(f"Generating samples... (N={self.config.NUM_SAMPLES}, Aug={aug_factor})")
        print(f"Total # = {self.config.NUM_SAMPLES * aug_factor} ")
        
        seeds = np.random.randint(0, 100000, size=num_samples)
        
        # 각 작업에 (system, config, seed, dist_params) 튜플 전달
        args_list = [(self.system, self.config, seeds[i], self.dist_params, bias_scale, diffusion_scale, aug_factor) for i in range(num_samples)]
        
        observed_data = []
        hidden_data = []
        params_data = []

        # 병렬 처리 실행
        # 사용할 CPU 코어 수 (None이면 가능한 모든 코어 사용)
        num_workers = min(8, os.cpu_count() or 1)
        print(f"Starting data generation with {num_workers} workers...")
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # results는 (X_obs, Y_hid, params_list) 튜플의 리스트가 됨
            results = list(executor.map(_generate_one_sample, args_list))
        
        print(f"Generation complete in {time.time() - start_time:.2f} seconds.")   

        # 결과 재조립
        for res in results:
            # res[0], res[1], res[2]는 각각 길이가 aug_factor인 리스트임
            observed_data.extend(res[0])
            hidden_data.extend(res[1])
            params_data.extend(res[2])

        observed_data = np.array(observed_data)
        hidden_data = np.array(hidden_data)
        params_data = np.array(params_data)

        print(f"Total Generated Samples: {len(observed_data)}")

        # Save
        print(f"Saving data to {save_path}...")
        np.savez_compressed(
            save_path,
            observed_data=observed_data,
            hidden=hidden_data,
            params=params_data,
            t_points=t_points
        )
            
        return observed_data, hidden_data, params_data, t_points


def create_dataloaders_(data_tuple, config):
    """
    data_tuple: (X_obs, Y_hidden, P_true, t_points)
      X_obs: ndarray shape (N, T, n_obs)
      Y_hidden: ndarray shape (N, T, n_hidden) or (N, n_hidden) depending on task
      P_true: ndarray shape (N, n_params)
      t_points: 1d ndarray length T
    """
    X_obs, Y_hidden, P_true, t_points = data_tuple
    N, T, n_features = X_obs.shape

    # (N, T, n_features) -> (N, T * n_features)
    X_flat = X_obs.reshape(N, T * n_features)

    # 텐서 변환
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_hidden, dtype=torch.float32)  # (N, T, n_hidden)
    
    # [유지] Y_hidden도 MLP 타깃을 위해 평탄화
    # (N, T, n_hidden) -> (N, T * n_hidden)
    if Y_tensor.dim() == 3:
        N_y, T_y, n_hidden = Y_tensor.shape
        Y_tensor = Y_tensor.reshape(N_y, T_y * n_hidden)
    elif Y_tensor.dim() == 2:
        pass # 이미 (N, T) 또는 (N, n_hidden) 등 2D 형태인 경우
    else:
        raise ValueError(f"Unexpected Y_tensor.dim(): {Y_tensor.dim()}")

    P_tensor = torch.tensor(P_true, dtype=torch.float32)

    # 데이터셋 및 로더 생성
    dataset = TensorDataset(X_tensor, Y_tensor, P_tensor)
    
    # Split Train/Test
    test_size = int(config.TEST_SPLIT * N)
    train_val_size = N - test_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
    
    # Split Train/Validation
    val_split = 0.1 # valid set 비율
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # 초기 파라미터 추측치 계산
    try:
        indices = train_dataset.indices
        p_initial_guess = P_tensor[indices].mean(dim=0).unsqueeze(0).to(config.DEVICE)
    except AttributeError:
        p_initial_guess = P_tensor[:train_size].mean(dim=0).unsqueeze(0).to(config.DEVICE)

    return train_loader, val_loader, test_loader, p_initial_guess


class RealOGTTDataLoader:
    """
    NIH OGTT 데이터를 로드하고 전처리하는 클래스.
    
    정책:
    1. Time Points: 15분 시점은 표준적이지 않으므로 제외하고 [0, 30, 60, 90, 120]만 사용함.
    2. Missing Values: 결측치가 하나라도 있는 환자 데이터는 품질 보장을 위해 삭제함.
    3. Parameters: 개별 환자의 BV 등은 계산하지 않고, 시스템 기본 파라미터를 따름.
    """
    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config = config
        self.t_points = np.array([0, 30, 60, 90, 120])  # 고정된 시간 지점

        # Derivative estimation logic
        root_path = Path(__file__).resolve().parent
        sigma_path = root_path / 'data' / 'parameters' / 'calibrated_sde_params.json'
        
        try:
            with open(sigma_path, 'r') as f:
                calib_data = json.load(f)
                self.s_glucose = np.sum(np.array(calib_data['sigma_G'])**2)
                self.s_insulin = np.sum(np.array(calib_data['sigma_I'])**2)
                print(f"[RealLoader] Loaded sigma for smoothing: s_G={self.s_glucose:.2f}, s_I={self.s_insulin:.2f}")
        except FileNotFoundError:
            print("[RealLoader] Warning: 'calibrated_sde_params.json' not found. Using default smoothing (s=None).")
            self.s_glucose = None
            self.s_insulin = None
            
        method = getattr(config, 'DERIVATIVE_METHOD', 'spline')
        
        # 메서드별 파라미터 설정
        kwargs = {}
        if method == 'spline':
            pass
        elif method == 'poly':
            kwargs['order'] = 3
            
        self.derivative_method = method
        self.derivative_kwargs = kwargs
        print(f"[RealLoader] Derivative Method: {self.derivative_method}")
            
            
    def _add_derivative_feature(self, t, y, s_val):
        N, T = y.shape
        y_aug = np.zeros((N, T, 2))
        
        if self.derivative_method == 'spline':
            estimator = get_derivative_estimator('spline', s=s_val)
        else:
            estimator = get_derivative_estimator(self.derivative_method, **self.derivative_kwargs)
        
        for i in range(N):
            y_aug[i, :, 0] = y[i, :]
            y_aug[i, :, 1] = estimator.estimate(t, y[i, :])
        
        return y_aug
        
    def load_data(self):
        print(f"Loading real OGTT data from {self.file_path}...")
        try:
            df = pd.read_excel(self.file_path)
        except:
            df = pd.read_csv(self.file_path)

        # column def
        glu_cols = ['oglu0', 'oglu30', 'oglu60', 'oglu90', 'oglu120']
        ins_cols = ['oins0', 'oins30', 'oins60', 'oins90', 'oins120']
        param_cols = ['si', 'sigma']

        required_cols = glu_cols + ins_cols + param_cols

        # Drop NA
        original_len = len(df)
        df_clean = df[required_cols].dropna()
        print(f"Data cleaning: Dropped {original_len - len(df_clean)} rows with missing values.")
        print(f"Remaining samples: {len(df_clean)}")

        glucose_raw = df_clean[glu_cols].values  # (N, 5)
        insulin_raw = df_clean[ins_cols].values  # (N, 5)
        params_data = df_clean[param_cols].values  # (N, 2)
        
        if getattr(self.config, 'USE_LAGRANGIAN', False):
            # (N, 5) -> (N, 5, 2)
            observed_data = self._add_derivative_feature(self.t_points, glucose_raw, self.s_glucose)
            # Hidden 변수(Insulin)는 DataGenerator와의 호환성을 위해 미분 미포함 (N, 5, 1)
            hidden_data = insulin_raw[:, :, np.newaxis]
        else:
            observed_data = glucose_raw[:, :, np.newaxis]  # (N, 5, 1)
            hidden_data = insulin_raw[:, :, np.newaxis]  # (N, 5, 1)
        
        return observed_data, hidden_data, params_data, self.t_points
    
def create_real_data_loaders(data_tuple, config):
    X_obs, Y_hid, P_true, t_points = data_tuple

    # 텐서 변환
    X_tensor = torch.tensor(X_obs, dtype=torch.float32)  # (N, T, 1)
    Y_tensor = torch.tensor(Y_hid, dtype=torch.float32)  # (N, T, 1)
    P_tensor = torch.tensor(P_true, dtype=torch.float32) # (N, n_params)
    
    # MLP 입력을 위한 평탄화 (N, T, 1) -> (N, T*1)
    N, T, _ = X_tensor.shape
    X_flat = X_tensor.reshape(N, -1)
    Y_flat = Y_tensor.reshape(N, -1) # Y도 필요시 평탄화
    
    # 데이터셋 생성
    dataset = TensorDataset(X_flat, Y_flat, P_tensor)
    
    # Train/Test Split
    train_size = int(0.8 * N)
    test_size = N - train_size
    train_d, test_d = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_d, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 초기 추측값 계산
    indices = train_d.indices
    p_initial_guess = P_tensor[indices].mean(dim=0).unsqueeze(0).to(config.DEVICE)
    
    return train_loader, test_loader, p_initial_guess

