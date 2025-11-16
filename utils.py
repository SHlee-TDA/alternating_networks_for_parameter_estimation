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
    

def euler_maruyama(drift_func, diffusion_func, 
                   t_span, y_0, t_eval,
                   params,
                   seed=None):
    """
    Euler-Maruyama method for solving SDEs.
    dY_t = f(t, Y_t) dt + g(t, Y_t) dW_t
    where f is the drift function and g is the diffusion function.

    Args:
        drift_func: function(t, y, params) -> dy/dt (deterministic)
        diffusion_func: function(t, y, params) -> noise scale (stochastic)
        t_span: tuple (t0, tf)
        y_0: initial condition (numpy array)
        t_eval: time points to evaluate the solution (numpy array)
        params: parameters for the SDE functions
        seed: random seed for reproducibility
    
    Returns:
        y_eval: solution at t_eval points (shape: [n_vars, len(t_eval)])
    """
    if seed is not None:
        np.random.seed(seed)

    t_start, t_end = t_span
    # t_eval 간격보다 더 촘촘한 시뮬레이션 step이 필요할 수 있음 (여기선 간소화)
    # 실제로는 dt를 t_eval 간격보다 작게 설정하고 보간해야 정밀하지만, 
    # 데이터 생성용으로는 t_eval 간격을 dt로 써도 무방한 경우가 많음.

    dt_values = np.diff(t_eval)
    mean_dt = np.mean(dt_values)
    if not np.allclose(dt_values, mean_dt, rtol=1e-2):
        # Time step이 불규칙하면 가장 작은 간격을 기준으로 보간 필요 (TODO)
        pass

    dt = mean_dt
    n_steps = len(t_eval)
    n_vars = len(y_0)

    y_curr = np.array(y_0)
    y_res = [y_curr.copy()]

    current_t = t_eval[0]

    for i in range(1, n_steps):
        target_t = eval[i]
        dt = target_t - current_t

        # Drift and Diffusion 계산
        f = np.array(drift_func(current_t, y_curr, params))
        g = np.array(diffusion_func(current_t, y_curr, params))

        # Brownian Motion increment dW ~ N(0, dt)
        dW = np.random.normal(0.0, np.sqrt(dt, size=n_vars))

        # Update state
        y_next = y_curr + f * dt + g * dW

        y_res.append(y_next.copy())
        y_curr = y_next
        current_t = target_t

    return np.array(y_res).T  # shape: [n_vars, n_steps]


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