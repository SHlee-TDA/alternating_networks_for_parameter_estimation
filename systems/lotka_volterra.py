# systems/lotka_volterra.py
import numpy as np
import torch
from .base_system import System

class LotkaVolterra(System):
    """Lotka-Volterra 시스템의 상세 명세 (호환성 및 Energy-based Sampling 유지)"""
    name = 'lotka_volterra'
    param_names = ['alpha', 'beta', 'delta', 'gamma']
    param_ranges = {
        'alpha': [0.6, 1.0], 'beta': [0.4, 0.8],
        'delta': [0.2, 0.6], 'gamma': [0.6, 1.0]
    }
    
    # 다른 모듈과의 호환성을 위해 기존 박스 범위 유지
    initial_conditions = ([5, 15], [1, 5]) # ranges for x and y initial conditions
    
    # 논문의 Task 난이도(Effective Sparsity) 제어를 위한 에너지 바운드 추가
    energy_bounds = {'min_delta_V': 0.1, 'max_delta_V': 1.5} 
    
    t_span = [0, 20]
    t_points = np.linspace(0, 20, 10)
    observed_var_idx = 0  # x (prey)
    hidden_var_idx = 1    # y (predator)

    def sample_initial_conditions(self, params_dict):
        """기존 박스 내에서 샘플링하되, 에너지(Delta V) 조건을 만족하는지 검증 (Rejection Sampling)"""
        alpha = params_dict['alpha']
        beta = params_dict['beta']
        delta = params_dict['delta']
        gamma = params_dict['gamma']

        # 1. 평형점 및 기준 에너지 계산
        x_eq = gamma / delta
        y_eq = alpha / beta
        V_eq = delta * x_eq - gamma * np.log(x_eq) + beta * y_eq - alpha * np.log(y_eq)

        min_dv = self.energy_bounds['min_delta_V']
        max_dv = self.energy_bounds['max_delta_V']

        max_retries = 1000  # 무한 루프 방지용 안전장치
        
        # 2. initial_conditions 박스 안에서 뽑되, 에너지가 조건에 맞을 때만 채택
        for _ in range(max_retries):
            x0 = np.random.uniform(*self.initial_conditions[0])
            y0 = np.random.uniform(*self.initial_conditions[1])

            V_0 = delta * x0 - gamma * np.log(x0) + beta * y0 - alpha * np.log(y0)
            delta_V = V_0 - V_eq

            if min_dv <= delta_V <= max_dv:
                return [float(x0), float(y0)]
                
        # 3. Fallback: 극단적인 파라미터 조합으로 인해 박스 내에 조건에 맞는 에너지가 아예 없을 경우
        # 파이프라인 에러를 막기 위해 평형점 근처에서 동적으로 조건을 맞춰 반환
        while True:
            x0_fallback = np.random.uniform(x_eq * 0.1, x_eq * 4.0)
            y0_fallback = np.random.uniform(y_eq * 0.1, y_eq * 4.0)
            V_0 = delta * x0_fallback - gamma * np.log(x0_fallback) + beta * y0_fallback - alpha * np.log(y0_fallback)
            
            if min_dv <= (V_0 - V_eq) <= max_dv:
                return [float(x0_fallback), float(y0_fallback)]

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
        alpha, beta, delta, gamma = theta[0], theta[1], theta[2], theta[3]
        
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        
        return torch.stack([dxdt, dydt], dim=1)