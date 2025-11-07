# systems/lotka_volterra.py
import numpy as np
from .base_system import System

class LotkaVolterra(System):
    """Lotka-Volterra 시스템의 상세 명세"""
    name = 'lotka_volterra'
    param_names = ['alpha', 'beta', 'delta', 'gamma']
    param_ranges = {
        'alpha': [0.8, 1.2], 'beta': [0.4, 0.8],
        'delta': [0.2, 0.6], 'gamma': [0.8, 1.2]
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
