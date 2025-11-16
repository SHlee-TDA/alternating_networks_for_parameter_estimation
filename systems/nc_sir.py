import numpy as np
from .sir import Sir

class NcSir(Sir):
    """Non-Conservative SIR 시스템의 상세 명세"""
    name = 'nc_sir'
    param_ranges = {
        'beta': [0.08 / 50.0, 0.12 / 50.0], 
        'gamma': [0.09, 0.11] # gamma는 N의 영향을 받지 않으므로 그대로 둠
    }

    def sample_initial_conditions(self, params_dict):
        return [
            self.initial_conditions[0][0],
            self.initial_conditions[1][0],
            self.initial_conditions[2][0]
        ]

    @staticmethod
    def ode_func(t, y, params):
        """
        ODE 함수에서 '/ N' 부분을 제거하여 보존 법칙을 파괴합니다.
        """
        S, I, R = y
        beta, gamma = params
        
        # '/ N'이 제거된 비보존(Non-conserved) 방정식
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        
        return [dSdt, dIdt, dRdt]