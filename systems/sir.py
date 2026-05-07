import numpy as np
import torch
from .base_system import System

class Sir(System):
    """
    SIR 시스템의 상세 명세 
    (Baseline을 무너뜨리기 위해 파라미터 범위와 Sparsity를 극단적으로 조정한 버전)
    """
    name = 'sir'
    param_names = ['beta', 'gamma']
    
    # 1. 파라미터 범위 확대: R0 가 0.02 부터 50 까지 다양하게 분포되도록 설정
    param_ranges = {'beta': [0.01, 0.5], 'gamma': [0.01, 0.5]}
    
    # SIR은 y0가 고정된 값 (S=49, I=1, R=0)
    initial_conditions = ([49.0], [1.0], [0.0])
    
    t_span = [0, 110]
    
    # 2. 관측 샘플 수 축소: 6개 -> 4개의 극단적인 Sparse 관측
    t_points = np.array([0, 30, 60, 90])
    
    observed_var_idx = 0  # S (Susceptible)
    hidden_var_idx = 1    # I (Infected)

    def sample_initial_conditions(self, params_dict):
        return [
            self.initial_conditions[0][0],
            self.initial_conditions[1][0],
            self.initial_conditions[2][0]
        ]

    @staticmethod
    def ode_func(t, y, params):
        S, I, R = y
        beta, gamma = params
        N = S + I + R
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
        
    @staticmethod
    def ode_func_torch(u, theta):
        """
        u: (Batch, 3) -> [S, I, R]
        theta: (2,) -> [beta, gamma]
        """
        S, I, R = u[:, 0], u[:, 1], u[:, 2]
        beta, gamma = theta[0], theta[1]
        N = S + I + R  # 총 인구 보존 법칙 활용
        
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        
        return torch.stack([dSdt, dIdt, dRdt], dim=1)