# systems/lotka_volterra.py
import numpy as np
import torch
from .base_system import System

class LotkaVolterra(System):
    """Lotka-Volterra 시스템의 상세 명세"""
    name = 'lotka_volterra'
    param_names = ['alpha', 'beta', 'delta', 'gamma']
    param_ranges = {
        'alpha': [0.6, 1.0], 'beta': [0.4, 0.8],
        'delta': [0.2, 0.6], 'gamma': [0.6, 1.0]
    }
    initial_conditions = ([5, 15], [1, 5])
    t_span = [0, 20]
    t_points = np.linspace(0, 20, 10)
    observed_var_idx = 0  # x (prey)
    hidden_var_idx = 1    # y (predator)

    def sample_initial_conditions(self, params_dict):
        return [
            np.random.uniform(*self.initial_conditions[0]),
            np.random.uniform(*self.initial_conditions[1])
        ]

    @staticmethod
    def ode_func(t, y, params):
        x, y_h = y
        alpha, beta, delta, gamma = params
        dxdt = alpha * x - beta * x * y_h
        dydt = delta * x * y_h - gamma * y_h
        return [dxdt, dydt]

    @staticmethod
    def ode_func_torch(u, theta):
        x, y = u[:, 0], u[:, 1]
        # param_names 순서에 정확히 맞춤
        alpha, beta, delta, gamma = theta[0], theta[1], theta[2], theta[3]
        
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        
        return torch.stack([dxdt, dydt], dim=1)    