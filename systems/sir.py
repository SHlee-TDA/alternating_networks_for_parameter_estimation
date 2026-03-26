# systems/sir.py
import numpy as np
import torch
from .base_system import System

class Sir(System):
    """SIR 시스템의 상세 명세"""
    name = 'sir'
    param_names = ['beta', 'gamma']
    param_ranges = {'beta': [0.05, 0.15], 'gamma': [0.05, 0.35]}
    
    # SIR은 y0가 고정된 값
    initial_conditions = ([50.0 - 1.0 - 0.0], [1.0], [0.0])
    
    t_span = [0, 110]
    t_points = np.array([0, 20, 40, 60, 80, 100])
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