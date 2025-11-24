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
    system, config, seed, dist_params = args
    
    np.random.seed(seed)
    
    # 1. Samping (Data-Driven or Uniform)
    # dist_params가 있으면 Data-Driven Log-Normal sampling 사용 (SDE 모드 권장)
    if dist_params is not None:
        si = sample_from_lognorm(dist_params['si'])
        sigma_p = sample_from_lognorm(dist_params['sigma'])
        params_list = [si, sigma_p]

        g0 = sample_from_lognorm(dist_params['G0'])
        i0 = sample_from_lognorm(dist_params['I0'])

        # 간소화: OgttSimul의 find_steady_state_N 로직을 호출하기 위해 임시 모델 생성
        from systems.ogtt_simul import OGTTModel, ode_params, sys_params
        temp_model = OGTTModel(ode_params, sys_params, {'si': si, 'sigma': sigma_p})
        n5, n6 = temp_model.find_steady_state_N(g0)
        
        y0 = [g0, i0, n5, n6]

    else:
        # Parameter sampling
        params_dict = {
            k: np.random.uniform(*v) 
            for k, v in system.param_ranges.items()
            }
        
        params_list = [
            params_dict[p] 
            for p in system.param_names
            ]
        # Initial state sampling
        y0 = system.sample_initial_conditions(params_dict)

    n_vars = len(y0)
    
    # 2. Simulation (SDE vs ODE)
    sys_instance = system() if isinstance(system, type) else system
    if getattr(config, 'USE_SDE', False):
        # SDE Solver: Euler-Maruyama
        y_full = euler_maruyama(
            sys_instance.drift_func,
            sys_instance.diffusion_func,
            sys_instance.t_span,
            y0,
            sys_instance.t_points,
            params_list,
            dt_sim=1.0,
            system=sys_instance # Clamping bounds 접근용
        )   # (n_vars, T)
    # Solve ODE
    else:
        sol = solve_ivp(
            fun=lambda t, y: system.ode_func(t, y, params_list),
            t_span=system.t_span,
            y0=y0,
            t_eval=system.t_points 
        )
        
        if not sol.success:
            # Fallback: use last y value repeated
            y_full = np.tile(sol.y[:, -1][:, None], (1, len(system.t_points)))
        else:
            y_full = sol.y  # shape (n_vars, T)
        
    # If Lagrangian method is used, compute derivatives at each time point
    if getattr(config, 'USE_LAGRANGIAN', False):
        t_points = system.t_points
        T = len(t_points)
        y_dot_full = np.zeros_like(y_full)

        for i in range(T):
            t_i = t_points[i]
            y_i = y_full[:, i]
            # ode_func=drift_func를 직접 호출하여 해당 시점의 도함수를 계산
            y_dot_full[:, i] = sys_instance.drift_func(t_i, y_i, params_list)
        y_full = np.concatenate([y_full, y_dot_full], axis=0)  # (n_vars*2, T)
    
    # Variable splitting
    y_full_T = y_full.T  # (T, n_features)
    
    obs_idx = [sys_instance.observed_var_idx]
    hid_idx = [sys_instance.hidden_var_idx]

    if getattr(config, 'USE_LAGRANGIAN', False):
        # 관측 변수의 도함수 인덱스 추가
        obs_deriv_idx = [idx + n_vars for idx in obs_idx]
        obs_idx += obs_deriv_idx
    
    X_obs = y_full_T[:, obs_idx] # (T, n_obs)
    Y_hid = y_full_T[:, hid_idx] # (T, n_hidden)
    
    return X_obs, Y_hid, params_list
    
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
            scale_val = getattr(self.config, 'DIFFUSION_SCALE', 'Not Found')
            print(f"[DEBUG] Config DIFFUSION_SCALE: {scale_val}")
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
            
            scale_factor = getattr(self.config, 'DIFFUSION_SCALE', 1.0)
            
            self.system.diffusion_scale = scale_factor
            print(f"Applied Diffusion Scale: {scale_factor}")
            
            # 각 작업에 전달할 고유한 시드 생성
            seeds = np.random.randint(0, 100000, size=num_samples)
            
            # 각 작업에 (system, config, seed, dist_params) 튜플 전달
            args_list = [(self.system, self.config, seeds[i], self.dist_params) for i in range(num_samples)]
            
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
                observed_data.append(res[0])
                hidden_data.append(res[1])
                params_data.append(res[2])

            observed_data = np.array(observed_data)
            hidden_data = np.array(hidden_data)
            params_data = np.array(params_data)

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


def create_dataloaders(data_tuple, config):
    # ... (기존 로직 유지) ...
    # 단, DataGenerator가 반환하는 shape이 (N, T, 1)이므로 reshape 로직이 호환되어야 함.
    # 기존 create_dataloaders는 (N, T, n_features)를 받아 (N, T*n_features)로 flattening 함.
    # 따라서 호환됨.
    X_obs, Y_hidden, P_true, t_points = data_tuple
    N, T, n_features = X_obs.shape

    # (N, T, n_features) -> (N, T * n_features)
    X_flat = X_obs.reshape(N, -1)
    
    # Y_hidden 처리 (N, T, 1) -> (N, T*1)
    if Y_hidden.ndim == 3:
        Y_flat = Y_hidden.reshape(N, -1)
    else:
        Y_flat = Y_hidden # 이미 평탄화된 경우

    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)
    P_tensor = torch.tensor(P_true, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor, P_tensor)
    
    # Split Train/Val/Test
    test_size = int(config.TEST_SPLIT * N)
    train_val_size = N - test_size
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])
    
    val_split = 0.1 
    val_size = int(val_split * train_val_size)
    train_size = train_val_size - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # 초기 추측값 (Train Mean)
    try:
        indices = train_dataset.indices
        p_initial_guess = P_tensor[indices].mean(dim=0).unsqueeze(0).to(config.DEVICE)
    except AttributeError:
        p_initial_guess = P_tensor[:train_size].mean(dim=0).unsqueeze(0).to(config.DEVICE)

    return train_loader, val_loader, test_loader, p_initial_guess


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



def lagrange_derivative(t_vals, y_vals):
    """
    Compute derivative of the Lagrange interpolation polynomial
    evaluated at each t in t_vals.
    Inputs:
        t_vals: list or 1d-array of length n
        y_vals: list or 1d-array of length n
    Returns:
        list of derivative values length n (dP/dt evaluated at each t_i)
    """
    n = len(t_vals)
    if n < 2:
        return [0.0] * n
    derivative_vals = []
    for t0 in t_vals:
        P_prime = 0.0
        for j in range(n):
            Lj_prime = 0.0
            for k in range(n):
                if k == j:
                    continue
                denom_jk = t_vals[j] - t_vals[k]
                prod = 1.0
                for m in range(n):
                    if m != j and m != k:
                        prod *= (t0 - t_vals[m]) / (t_vals[j] - t_vals[m])
                Lj_prime += prod / denom_jk
            P_prime += y_vals[j] * Lj_prime
        derivative_vals.append(P_prime)
    return derivative_vals

def compute_lagrange_derivative_matrix(t_arr, y_matrix, fd_threshold=50):
    """
    t_arr: 1d-array length T
    y_matrix: array shape (n_vars, T)
    returns: array shape (n_vars, T) with derivative for each variable
    """
    T = len(t_arr)
    n_vars = y_matrix.shape[0]
    deriv = np.zeros_like(y_matrix, dtype=float)

    # 성능 폴백: 시간 포인트가 많으면 중앙차분으로 대체
    if T > fd_threshold:
        dt = np.gradient(t_arr)
        for i in range(n_vars):
            # y_matrix[i] shape (T,)
            deriv[i, :] = np.gradient(y_matrix[i, :], t_arr)
        return deriv

    # 기본 Lagrange 기반 계산 (각 변수에 대해)
    for i in range(n_vars):
        deriv[i, :] = np.array(lagrange_derivative(list(map(float, t_arr)), list(map(float, y_matrix[i, :]))))
    return deriv

def augment_observed_with_lagrangian(t_arr, observed_sample, use_lagrangian: bool):
    """
    observed_sample: ndarray shape (T, n_obs)
    returns: ndarray shape (T, n_features) where n_features = n_obs (+ n_obs if use_lagrangian)
    """
    base = np.asarray(observed_sample)  # (T, n_obs)
    if not use_lagrangian:
        return base
    # compute derivatives for observed vars: compute_lagrange_derivative_matrix expects (n_vars, T)
    y_matrix = base.T  # (n_obs, T)
    deriv = compute_lagrange_derivative_matrix(t_arr, y_matrix)  # (n_obs, T)
    deriv_T = deriv.T  # (T, n_obs)
    X = np.concatenate([base, deriv_T], axis=1)  # (T, 2*n_obs)
    return X

def check_lagrangian_applied(data_tuple, config, n_samples=3):
    """
    data_tuple: (X_obs, Y_hidden, P_true, t_points)
    Checks if Lagrangian augmentation is applied correctly.
    """
    X_obs, _, _, t_points = data_tuple
    N, T, n_obs = X_obs.shape
    use = bool(getattr(config, "Use_LAGRANGIAN", False))
    
    sample_shapes = []
    for i in range(min(n_samples, N)):
        X_aug = augment_observed_with_lagrangian(t_points, X_obs[i], use)
        sample_shapes.append(X_aug.shape)
        expected_n_features = n_obs * 2 if use else n_obs
        assert X_aug.shape == (T, expected_n_features), f"Sample {i} shape mismatch: got {X_aug.shape}, expected {(T, expected_n_features)}"


class OGTTModel:
    """
    OGTT 모델의 기본 클래스입니다.
    
    이 클래스는 포도당-인슐린 동역학을 4개의 미분방정식으로 모델링합니다:
    - 포도당 농도 (G)
    - 인슐린 농도 (I)
    - 인슐린 분비 관련 변수 (N5)
    - 인슐린 분비 관련 변수 (N6)
    
    Attributes:
        ode_params (dict): ODE 시스템 파라미터
        sys_params (dict): 시스템 파라미터
        theta (dict): 모델 파라미터 (si, sigma)
    """
    def __init__(self, ode_params, sys_params, theta):
        self.ode_params = ode_params
        self.sys_params = sys_params
        self.theta = theta

    def GI_ode_universal(self, t, y):
        """
        Defines the system of ODEs for the glucose-insulin model.

        Parameters:
        t : float
            Current time point.
        y : array_like
            Current state vector [G, I, N5, N6], where:
            G : Glucose concentration
            I : Insulin concentration
            N5, N6 : Variables related to insulin secretion dynamics

        Returns:
        dydt : tuple
            Derivatives [dG/dt, dI/dt, dN5/dt, dN6/dt]
        """
        # 상태 변수 언패킹
        G, I, N5, N6 = y

        # 시스템 파라미터 접근
        p_sys = self.sys_params
        p_ode = self.ode_params
        
        # 시스템 파라미터 설정
        Eg0 = p_sys['Eg0']
        k = p_sys['k']
        BV = p_sys['BV']
        b = p_sys['b']

        # 대사율 M 계산
        M = self.calculate_metabolic_rate(G)

        # OGTT 투여율 계산
        OGTT_rate = self.calculate_ogtt_flux(t)

        # 간 포도당 생성(HGP) 계산
        HGP = self.calculate_HGP(I)

        # Glucose Amplifying Factor (GF) 계산
        GF = self.calculate_GF(G)
        

        # Microdomain Ca2+ (cmd) 계산
        ci = self.calculate_ci(M)
        cmd = self.calculate_cmd(ci)

        # 인슐린 분비 관련 변수 계산
        r2 = self.calculate_r2(ci)
        r3 = self.calculate_r3(ci, GF)
        CN = self.calculate_CN(cmd)
        CN1 = CN[0]
        ISR = self.calculate_ISR(CN, N5)

        # ODE 계산
        ts = p_sys['ts']
        unit_con = p_sys['unit_con']
        r1 = p_sys['r1']
        rm1 = p_sys['rm1']
        rm2 = p_sys['rm2']
        rm3 = p_sys['rm3']
        si = self.theta['si']


        dGdt = HGP + OGTT_rate - (Eg0 + unit_con * si * I) * G
        dIdt = (b * ISR) / BV - k * I
        dN5dt = ts * (rm1 * CN1 * N5 - (r1 + rm2) * N5 + r2 * N6)
        dN6dt = ts * (r3 + rm2 * N5 - (rm3 + r2) * N6)

        dydt = dGdt, dIdt, dN5dt, dN6dt
        return dydt

    def simulate(self, t_span, initial_conditions, t_eval=None):
        if t_eval==None:
            t_eval = np.linspace(0, 120, 121)

        solution = solve_ivp(
            self.GI_ode_universal,
            t_span,
            initial_conditions,
            method='BDF',
            t_eval=t_eval
        )
        return solution

        
    def calculate_metabolic_rate(self, G):
        """
        Calculates the metabolic rate M as a function of glucose rate G.

        Parameters:
        G : float
            Current glucose rate.

        Returns:
        M : float
            Metabolic rate.

        Equation:
        M = Mmax * G^kM / (alpha_M^kM + G^kM)
        """
        p_sys = self.sys_params

        Mmax = p_sys['Mmax']
        alpha_M = p_sys['alpha_M']
        kM = p_sys['kM']

        numerator = Mmax * G ** kM
        denominator = alpha_M ** kM + G ** kM
        M = numerator / denominator

        return M


    def calculate_ogtt_flux(self, t):
        """
        Calculates the glucose infusion rate during an OGTT at time t.

        Parameters:
        t : float
            Current time point.

        Returns:
        OGTT_flux : float
            OGTT glucose infusion rate.

        Equation:
        OGTT_flux = OGTT_bar * [Piecewise function based on time intervals]
        """
        p_ode = self.ode_params
        p_sys = self.sys_params

        t1 = p_ode['t1']
        t2 = p_ode['t2']
        t3 = p_ode['t3']
        a1 = p_ode['a1']
        a2 = p_ode['a2']
        a3 = p_ode['a3']

        OGTT_bar = p_sys['OGTT_bar']

        # [Note: Vectorization Fix]
        # scipy.solve_ivp의 BDF/LSODA 솔버는 Jacobian 계산 등을 위해 시간 t를 
        # 스칼라가 아닌 벡터(array) 형태로 전달할 수 있습니다.
        # 따라서 Python 기본 if문 대신 NumPy의 벡터 연산(np.select)을 사용해야 합니다.
        # 절대 if 0 < t <= t1: 형태로 되돌리지 마세요!
        if 0 < t <= t1:
            OGTT_flux = t * a1 / t1
        elif t1 < t <= t2:
            OGTT_flux = ((t - t2) * (a2 - a1) / (t2 - t1)) + a2
        elif t2 < t <= t3:
            OGTT_flux = (t - t3) * (a3 - a2) / (t3 - t2)
        else:
            OGTT_flux = 0

        return OGTT_bar * OGTT_flux

    def calculate_HGP(self, I):
        """
        Calculates the hepatic glucose production (HGP) as a function of insulin rate I.

        Parameters:
        I : float
            Current insulin rate.

        Returns:
        HGP : float
            Hepatic glucose production rate.

        Equations:
        hepa_max = hepa_bar / (hepa_k + si) + hepa_b
        alpha_HGP = alpha_max / (alpha_k + si) + alpha_b
        HGP = hepa_max / (alpha_HGP + hepasi * I) + HGP_b
        """
        p_sys = self.sys_params
        p_ode = self.ode_params

        hepa_bar = p_sys['hepa_bar']
        hepa_k = p_sys['hepa_k']
        hepa_b = p_sys['hepa_b']

        si = self.theta['si']
        hepasi = p_ode['hepasi']

        hepa_max = hepa_bar / (hepa_k + si) + hepa_b

        alpha_max = p_sys['alpha_max']
        alpha_b = p_sys['alpha_b']
        alpha_k = p_sys['alpha_k']

        alpha_HGP = alpha_max / (alpha_k + si) + alpha_b
    
        HGP_b = p_sys['HGP_b']
        HGP = hepa_max / (alpha_HGP + hepasi * I) + HGP_b

        return HGP
    
    def calculate_GF(self, G):
        """
        Calculates the Glucose Amplifying Factor (GF) as a function of glucose rate G.

        Parameters:
        G : float
            Current glucose rate.

        Returns:
        GF : float
            Glucose Amplifying Factor.

        Equation:
        GF = [GF_bar * (G - shGF)^kGF] / [alpha_GF^kGF + (G - shGF)^kGF] + GF_b
        """
        p_sys = self.sys_params

        GF_bar = p_sys['GF_bar']
        kGF = p_sys['kGF']
        alpha_GF = p_sys['alpha_GF']
        shGF = p_sys['shGF']
        GF_b = p_sys['GF_b']

        numerator = GF_bar * (G - shGF) ** kGF
        denominator = alpha_GF ** kGF + (G - shGF) ** kGF
        GF = numerator / denominator + GF_b

        return GF

    def calculate_ci(self, M):
        """
        Calculates the microdomain calcium ci as a function of metabolic rate M.

        Parameters:
        M : float
            Metabolic rate.

        Returns:
        ci : float
            Microdomain calcium.

        Equation:
        ci = [ca_bar * (M + gamma_bar * gamma)^kca] / [alpha_ca^kca + (M + gamma_bar * gamma)^kca] + ca_b
        """
        p_sys = self.sys_params
        p_ode = self.ode_params

        ca_bar = p_sys['ca_bar']
        kca = p_sys['kca']
        alpha_ca = p_sys['alpha_ca']
        ca_b = p_sys['ca_b']
        gamma = p_ode['gamma']
        gamma_bar = p_ode['gamma_bar']

        ci_input = M + gamma_bar * gamma
        numerator = ca_bar * ci_input ** kca
        denominator = alpha_ca ** kca + ci_input ** kca
        ci = numerator / denominator + ca_b

        return ci

    def calculate_cmd(self, ci):
        p_sys = self.sys_params

        cmd_factor = p_sys['cmd_factor']
        cmd_b = p_sys['cmd_b']
        cik = p_sys['cik']
        cialpha = p_sys['cialpha']

        numerator = cmd_factor * ci ** cik
        denominator = cialpha ** cik + ci ** cik
        cmd = numerator / denominator + cmd_b

        return cmd

    def calculate_r2(self, ci):
        p_ode = self.ode_params
        p_sys = self.sys_params

        r20 = p_ode['r20']
        Kp2 = p_sys['Kp2']

        r2 = r20 * ci / (ci + Kp2)
        return r2

    def calculate_r3(self, ci, GF):
        p_ode = self.ode_params
        p_sys = self.sys_params

        r30 = p_sys['r30']
        sigma = self.theta['sigma']
        Kp2 = p_sys['Kp2']

        r3 = sigma * GF * r30 * ci / (ci + Kp2)
        return r3

    def calculate_CN(self, cmd):
        p_sys = self.sys_params

        k1 = p_sys['k1']
        km1 = p_sys['km1']
        r1 = p_sys['r1']
        rm1 = p_sys['rm1']
        u1 = p_sys['u1']

        # Fast-slow analysis 변수 계산
        N1_C = km1 / (3 * k1 * cmd + rm1)
        N1_D = r1 / (3 * k1 * cmd + rm1)
        N2_E = 3 * k1 * cmd / (2 * k1 * cmd + km1)
        N2_F = 2 * km1 / (2 * k1 * cmd + km1)
        N3_L = 2 * k1 * cmd / (2 * km1 + k1 * cmd)
        N3_N = 3 * km1 / (2 * km1 + k1 * cmd)

        # Fast-slow analysis by considering N6 and N5 slow and all other fast
        CN4 = (k1 * cmd) / (3 * km1 + u1)
        CN3 = N3_L / (1 - N3_N * CN4)
        CN2 = N2_E / (1 - N2_F * CN3)
        CN1 = N1_D / (1 - N1_C * CN2)

        return (CN1, CN2, CN3, CN4)

    def calculate_ISR(self, CN, N5):
        p_sys = self.sys_params

        u1 = p_sys['u1']
        u2 = p_sys['u2']
        u3 = p_sys['u3']
        ts = p_sys['ts']

        CN1, CN2, CN3, CN4 = CN

        N1 = CN1 * N5
        N2 = CN2 * N1
        N3 = CN3 * N2
        N4 = CN4 * N3
        NF = u1 * N4 / u2
        NR = (u2 / u3) * NF

        ISR = ts * 9 * (u3 * NR)

        return ISR


def find_steady_state_N(oglu0, model):
    """
    Given initial glucose value (oglu(0)),
    compute equilibrium states of N5 and N6 (dN5/dt = 0, dN6/dt =0) via algebra.
    """
    M = model.calculate_metabolic_rate(oglu0)
    ci = model.calculate_ci(M)
    GF = model.calculate_GF(oglu0)
    cmd = model.calculate_cmd(ci)
    
    CN = model.calculate_CN(cmd)
    CN1 = CN[0]

    
    rm1, rm2, rm3 = model.sys_params['rm1'], model.sys_params['rm2'], model.sys_params['rm3']
    r1, r2, r3 = model.sys_params['r1'], model.calculate_r2(ci), model.calculate_r3(ci, GF)
    
    A = np.array([
        [rm1 * CN1 - (r1 + rm2), r2],
        [rm2, -(rm3 + r2)]
    ])
    b = np.array([0, -r3])
    
    try:
        n5_ss, n6_ss = np.linalg.solve(A, b)
        return n5_ss, n6_ss
    except np.linalg.LinAlgError:
        return 1.0, 0.5

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

