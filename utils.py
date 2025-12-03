# utils.py
import torch
import numpy as np
import os
import pandas as pd
import uuid
from datetime import datetime
import git
from abc import ABC, abstractmethod
from scipy.interpolate import UnivariateSpline, lagrange


class Normalizer:
    # 1. 초기화 메서드에 use_log_params 인자 추가
    def __init__(self, system, device, state_scales=None, param_bounds=None, use_log_params=False):
        self.device = device
        self.use_log_params = use_log_params # [추가] 로그 변환 사용 여부

        # --- Parameter Normalization Init ---
        if param_bounds is not None:
            mins, maxs = param_bounds
            # print(f"[Normalizer] Using Data-Driven Parameter Bounds")
        else:
            mins = [system.param_ranges[name][0] for name in system.param_names]
            maxs = [system.param_ranges[name][1] for name in system.param_names]
        
        self.p_min = torch.tensor(mins, device=self.device, dtype=torch.float32)
        self.p_max = torch.tensor(maxs, device=self.device, dtype=torch.float32)

        # [핵심 수정] 로그 변환 모드일 경우, Min/Max 경계값도 로그를 취해줌
        if self.use_log_params:
            print("[Normalizer] ✅ Enabled Log-Space Normalization for Parameters.")
            # 0이나 음수가 들어오지 않도록 안전장치(1e-6) 후 로그 변환
            self.p_min = torch.log(torch.maximum(self.p_min, torch.tensor(1e-6, device=self.device)))
            self.p_max = torch.log(torch.maximum(self.p_max, torch.tensor(1e-6, device=self.device)))

        # 범위 계산 (로그 변환된 min/max 기반으로 계산됨)
        self.p_range = torch.maximum(
            self.p_max - self.p_min, 
            torch.tensor(1e-6, device=self.device)
        )

        # --- Input State Normalization Init (기존과 동일) ---
        if state_scales is not None:
            self.state_scales = torch.tensor(state_scales, device=self.device, dtype=torch.float32)
            self.state_scales = torch.maximum(self.state_scales, torch.tensor(1e-6, device=self.device))
        else:
            self.state_scales = torch.tensor([600.0, 3000.0], device=self.device, dtype=torch.float32)

    def normalize_params(self, p):
        """ [min, max] -> [-1, 1] (Log option applied) """
        # [핵심 수정] 입력값에 먼저 로그를 취함
        if self.use_log_params:
            p = torch.log(torch.maximum(p, torch.tensor(1e-6, device=self.device)))
            
        p_norm = (p - self.p_min) / self.p_range
        return p_norm * 2.0 - 1.0
        # return p
    
    def denormalize_params(self, p_norm):
        """ [-1, 1] -> [min, max] (Log option applied) """
        p_01 = (p_norm + 1.0) / 2.0
        p_recovered = p_01 * self.p_range + self.p_min
        
        # [핵심 수정] 복원된 값(로그 상태)을 지수(Exp)로 변환
        if self.use_log_params:
            return torch.exp(p_recovered)
            
        return p_recovered
        #return p_norm

    def normalize_inputs(self, x, variable_type=None):
        """ X -> X / Scale """
        n_obs = 1
        n_hid = 1
        
        scale_obs = self.state_scales[0] # Glucose Scale
        scale_hid = self.state_scales[1] # Insulin Scale
        
        # 1. 명시적 타입이 지정된 경우 (Trainer에서 사용)
        if variable_type == 'observed':
            # x가 관측 변수(Glucose)일 때
            return x / scale_obs
        elif variable_type == 'hidden':
            # x가 숨겨진 변수(Insulin)일 때 -> 올바른 스케일 적용!
            return x / scale_hid
        
        # 입력 차원에 맞춰 스케일 벡터 확장 (Lagrangian 미분항 대응)
        # x shape: [Batch, Dim]
        base_dim = len(self.state_scales)
        
        if x.shape[1] == base_dim:
            scale = self.state_scales
        elif x.shape[1] == 2 * base_dim: # 미분값 포함 시
            # 미분값도 동일한 스케일로 나누어야 물리적 관계 보존됨 (중요!)
            scale = torch.cat([self.state_scales, self.state_scales])
        else:
            # 예외 처리: 일단 앞부분만이라도 맞춤
            scale = self.state_scales.repeat(x.shape[1] // base_dim + 1)[:x.shape[1]]
            
        return x / scale
        #return x

    def denormalize_inputs(self, x_norm, variable_type=None):
        """ X_norm -> X_norm * Scale """
        # normalize_inputs와 동일한 로직으로 scale 구성 후 곱셈
        
        scale_obs = self.state_scales[0]
        scale_hid = self.state_scales[1]
        
        if variable_type == 'observed':
            return x_norm * scale_obs
        elif variable_type == 'hidden':
            return x_norm * scale_hid
        
        base_dim = len(self.state_scales)
        if x_norm.shape[1] == 2 * base_dim:
            scale = torch.cat([self.state_scales, self.state_scales])
        else:
            scale = self.state_scales
        return x_norm * scale
        #return x_norm
    

class DerivativeEstimator(ABC):
    """
    관측된 OGTT 데이터로부터 미분값(기울기)을 추정하는 추상 클래스.
    구체적인 추정 방법은 서브클래스에서 구현해야 합니다.
    """
    @abstractmethod
    def estimate(self, t, y):
        """        
        Args:
            t (np.array): 시간 축 (Time points)
            y (np.array): 관측 값 (Observed values)
        Returns:
            dydt (np.array): 추정된 미분값 배열
        """
        pass

class SplineDerivative(DerivativeEstimator):
    """
    Smoothing Spline(Reinsch)을 사용하여 미분
    - s: Smoothing factor (None이면 interpolation)
    - k: Degree of spline (default=3, cubic)
    """
    def __init__(self, s=None, k=3):
        self.s = s
        self.k = k

    def estimate(self, t, y):
        try:
            # 데이터 개수가 k보다 작으면 에러가 발생하므로 방어 코드 작성
            if len(t) <= self.k:
                k_safe = len(t) - 1
            else:
                k_safe = self.k
            
            spline = UnivariateSpline(t, y, s=self.s, k=k_safe)
            return spline.derivative()(t)
        except Exception as e:
            print(f"[SplineDerivative] Error in spline fitting: {e}. Returning zeros.")
            return np.zeros_like(y)

class PolynomialDerivative(DerivativeEstimator):
    """
    Polynomial regression을 이용한 미분
    - order: 다항식 차수 (defaut=3)
    """
    def __init__(self, order=3):
        self.order = order
    
    def estimate(self, t, y):
        try:
            order_safe = min(self.order, len(t) - 1)
            poly_coeffs = np.polyfit(t, y, order_safe)
            deriv_coeffs = np.polyder(poly_coeffs)
            return np.polyval(deriv_coeffs, t)
        except Exception as e:
            print(f"[PolynomialDerivative] Error in polynomial fitting: {e}. Returning zeros.")
            return np.zeros_like(y)  
        
class LagrangeDerivative(DerivativeEstimator):
    """
    Lagrange 보간법을 이용한 미분
    """
    def estimate(self, t, y):
        try:
            poly = lagrange(t, y)
            deriv_poly = np.polyder(poly)
            return deriv_poly(t)
        except Exception as e:
            print(f"[LagrangeDerivative] Error in Lagrange fitting: {e}. Returning zeros.")
            return np.zeros_like(y)

class FiniteDifferenceDerivative(DerivativeEstimator):
    """
    유한차분법을 이용한 미분
    """
    def estimate(self, t, y):
        return np.gradient(y, t)
    
# Factory Function
def get_derivative_estimator(method='spline', **kwargs):
    """
    미분 추정기 객체를 생성하는 팩토리 함수.
    
    Args:
        method (str): 'spline', 'poly', 'lagrange', 'finite_diff' 중 하나.
        kwargs: 각 추정기별 추가 파라미터.
        
    Returns:
        DerivativeEstimator 인스턴스
    """
    estimators = {
        'spline': SplineDerivative,
        'poly': PolynomialDerivative,
        'lagrange': LagrangeDerivative,
        'finite_diff': FiniteDifferenceDerivative
    }
    
    if method not in estimators:
        raise ValueError(f"Unknown derivative method: {method}. Choose from {list(estimators.keys())}")
    
    return estimators[method](**kwargs)
    
# Euler-Maruyama SDE Solver
def euler_maruyama(drift_func, diffusion_func, t_span, y0, t_eval, params, seed=None, dt_sim=1.0, system=None):
    """
    Euler-Maruyama method for solving SDEs with fine time steps.
    
    Args:
        drift_func: function(t, y, params) -> dy/dt (4-vector)
        diffusion_func: function(t, y, params) -> diffusion matrix (4x4)
        t_span: [t_start, t_end]
        y0: initial state vector (4-vector)
        t_eval: time points to evaluate (e.g., [0, 30, 60, 90, 120])
        params: system parameters (si, sigma)
        seed: random seed
        dt_sim: Internal simulation time step (e.g., 1.0 minute)
        system: (Optional) System instance to retrieve state_bounds for clamping.
    
    Returns:
        y_full: Solution at every dt_sim step (shape: [n_vars, n_steps])
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start, t_end = t_span
    n_vars = len(y0)
    
    

    # 시스템 객체에서 bounds 정보가 있으면 가져옴
    if system is not None and hasattr(system, 'state_bounds'):
        lower_bounds, upper_bounds = system.state_bounds
    else:
        # 기본 하한 안전장치 (User defined 1e-6)
        # Clamping Bounds 설정
        lower_bounds = 1e-6
        upper_bounds = 1e+3

    # 시뮬레이션에 사용할 전체 시간 스텝
    n_total_steps = int(np.ceil((t_end - t_start) / dt_sim))
    t_sim_points = np.linspace(t_start, t_end, n_total_steps + 1)
    dt_actual = t_sim_points[1] - t_sim_points[0]
    
    y_curr = np.array(y0)
    y_res = [y_curr.copy()] # (t=0)
    
    for i in range(n_total_steps):
        t_curr = t_sim_points[i]
        
        # Drift & Diffusion 계산
        f = np.array(drift_func(t_curr, y_curr, params))
        G = np.array(diffusion_func(t_curr, y_curr, params))
        
        # Brownian Motion increment dW ~ N(0, dt_actual)
        # G가 4x4 행렬이므로, dW는 4개의 독립적인 Wiener Process를 가짐 (4-vector)
        dW = np.random.normal(0, np.sqrt(dt_actual), size=n_vars)
        
        # SDE Update: Y_{t+dt} = Y_t + f*dt + G * dW
        # G * dW는 행렬-벡터 곱셈 (4x4) * (4x1) -> (4x1)
        y_next = y_curr + f * dt_actual + G @ dW
        # Clamping (Physical Constraints)
        y_next = np.clip(y_next, lower_bounds, upper_bounds)
        
        y_res.append(y_next.copy())
        y_curr = y_next

    y_full = np.array(y_res).T # (n_vars, n_sim_steps+1)
    
    # t_eval 위치로 보간 (Resampling)
    y_out = np.zeros((n_vars, len(t_eval)))
    for k in range(n_vars):
        y_out[k, :] = np.interp(t_eval, t_sim_points, y_full[k, :])
        
    return y_out


def get_git_hash():
    """현재 Git 커밋 해시를 반환합니다."""
    try:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        return git_hash[:7]
    except:
        return "no_git"

class ExperimentLogger:
    """
    실험 결과를 CSV 파일로 저장하는 로거 클래스.
    """
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.uuid = str(uuid.uuid4())[:8]
        
        self.exp_dir_name = f"{self.timestamp}_{config.EXPERIMENT_NAME}_{self.uuid}"
        self.results_dir = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, self.exp_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # Configi 저장
        self._save_config()

    def _save_config(self):
        # Config 객체를 dict로 변환하여 저장
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('__') and not callable(v)}
        # 직렬화 불가능한 객체 처리
        config_dict['DEVICE'] = str(config_dict.get('DEVICE', 'cpu'))
        if 'EXPERIMENTS' in config_dict: del config_dict['EXPERIMENTS']
        
        import json
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

    def log_result_to_csv(self, metrics_dict):
        """
        실험 결과를 중앙 CSV 레지스트리에 등록합니다.
        metrics_dict: {'val_loss': 0.1, 'test_error': 0.05, ...}
        """
        registry_path = os.path.join(self.config.RESULTS_DIR, 'experiment_registry.csv')
        
        # 기본 정보 구성
        log_data = {
            'timestamp': self.timestamp,
            'system': self.config.SYSTEM_NAME,
            'experiment': self.config.EXPERIMENT_NAME,
            'uuid': self.uuid,
            'git_hash': get_git_hash(),
            'use_sde': getattr(self.config, 'USE_SDE', False),
            'use_lagrangian': getattr(self.config, 'USE_LAGRANGIAN', False)
        }
        # 메트릭 병합
        log_data.update(metrics_dict)
        
        df_new = pd.DataFrame([log_data])
        
        if os.path.exists(registry_path):
            df_old = pd.read_csv(registry_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined.to_csv(registry_path, index=False)
        print(f"[Logger] Experiment registered to {registry_path}")

    def get_save_path(self, filename):
        return os.path.join(self.results_dir, filename)