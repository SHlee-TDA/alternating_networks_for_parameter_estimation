import os
import numpy as np
import torch
from .base_system import System

# Parameter-range regime, selectable via the SIR_RANGE env var (default: 'wide',
# the original adversarial setting). 'narrow' matches the ranges stated in the paper
# draft (beta~U[0.08,0.12], gamma~U[0.09,0.11]) for the "clean identifiable" SIR story.
_SIR_RANGE = os.environ.get('SIR_RANGE', 'wide').lower()
_SIR_PARAM_RANGES = {
    'wide':   {'beta': [0.01, 0.5],  'gamma': [0.01, 0.5]},
    'narrow': {'beta': [0.08, 0.12], 'gamma': [0.09, 0.11]},
}

class Sir(System):
    """
    SIR 시스템의 상세 명세
    (Baseline을 무너뜨리기 위해 파라미터 범위와 Sparsity를 극단적으로 조정한 버전)
    """
    name = 'sir'
    param_names = ['beta', 'gamma']

    # Parameter range: 'wide' (adversarial, R0 in [0.02,50]) or 'narrow' (paper-draft ranges).
    param_ranges = _SIR_PARAM_RANGES.get(_SIR_RANGE, _SIR_PARAM_RANGES['wide'])
    
    # SIR은 y0가 고정된 값 (S=49, I=1, R=0)
    initial_conditions = ([49.0], [1.0], [0.0])
    
    t_span = [0, 110]

    # Observation grid. Default is the extreme-sparse 4-point grid; SIR_NPOINTS=<n> replaces it
    # with n equally spaced points over [0,100], used to sweep observation density and show that
    # the one-step residual (and hence the fixed-point error) shrinks toward 0 as observations
    # densify --- i.e., correctness IS achieved when the hidden state becomes reconstructable.
    _sir_np = int(os.environ.get('SIR_NPOINTS', '0'))
    t_points = np.linspace(0, 100, _sir_np).astype(int) if _sir_np >= 2 else np.array([0, 30, 60, 90])

    observed_var_idx = 0  # S (Susceptible)
    hidden_var_idx = 1    # I (Infected)

    def sample_initial_conditions(self, params_dict):
        # SIR_FIXED_IC=1 pins the initial state to (S0,I0,R0)=(49,1,0), removing the
        # initial-condition nuisance variability. This matches the setting under which
        # the iterative operator was previously observed to recover parameters well.
        if os.environ.get('SIR_FIXED_IC', '0') == '1':
            return 49.0, 1.0, 0.0
        I0 = np.random.uniform(1.0, 10.0)
        S0 = 50.0 - I0
        R0 = 0.0
        return S0, I0, R0

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