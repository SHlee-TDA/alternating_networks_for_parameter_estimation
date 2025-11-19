# utils.py
import torch
import numpy as np
import os
import pandas as pd
import uuid
from datetime import datetime
import git

class Normalizer:
    """
    파라미터의 Min-Max 정규화 및 역정규화를 담당하는 클래스.
    모든 파라미터를 [0, 1] 범위로 스케일링합니다.
    """
    def __init__(self, system, device):
        """
        시스템 객체로부터 파라미터의 최솟값(min)과 범위(range)를 계산합니다.
        """
        self.device = device
        mins = []
        maxs = []
        
        for name in system.param_names:
            mins.append(system.param_ranges[name][0])
            maxs.append(system.param_ranges[name][1])
        
        self.min = torch.tensor(mins, device=self.device, dtype=torch.float32)
        self.max = torch.tensor(maxs, device=self.device, dtype=torch.float32)
        # 0으로 나누는 것을 방지하기 위해 작은 epsilon 추가
        self.range = self.max - self.min + 1e-8

    def normalize(self, p):
        """파라미터 텐서를 [0, 1] 범위로 정규화합니다."""
        #return (p - self.min) / self.range
        return p
    def denormalize(self, p_norm):
        """정규화된 텐서를 원래의 파라미터 스케일로 되돌립니다."""
        #return p_norm * self.range + self.min
        return p_norm
    

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
    
    # Clamping Bounds 설정
    lower_bounds = -np.inf
    upper_bounds = np.inf

    # 시스템 객체에서 bounds 정보가 있으면 가져옴
    if system is not None and hasattr(system, 'state_bounds'):
        lower_bounds, upper_bounds = system.state_bounds
    else:
        # 기본 하한 안전장치 (User defined 1e-6)
        lower_bounds = 1e-6

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