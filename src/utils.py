# utils.py
"""
This module provides essential tools for:
1. Data Normalization: Scaling preserving physical constraints.
2. Derivative Estimation: .
3. Stochastic Simulation: Euler-Maruyama solver for SDEs.
4. Experiment Logging: Management of artifacts and configurations.
"""


import os
import json
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, lagrange


# Derivative Estimation Strategy ---
class DerivativeEstimator(ABC):
    """
    Abstract Base Class for estimating derivatives (dy/dt) from discrete observations.
    """
    @abstractmethod
    def estimate(self, t, y):
        """        
        Args:
            t (np.array): Time points
            y (np.array): Observed values
        Returns:
            dydt (np.array): Estimated derivatives
        """
        pass

class SplineDerivative(DerivativeEstimator):
    """
    Estimates derivatives using Reinsch Smoothing Splines.
    
    Rationale:
        For sparse and noisy biological data (e.g., OGTT), simple finite differences 
        amplify noise. Smoothing splines minimize a tradeoff between fitting error 
        and curvature, controlled by the smoothing factor 's' (derived from sigma^2).
    """
    def __init__(self, s=None, k=3):
        self.s = s
        self.k = k

    def estimate(self, t, y):
        try:
            # Fallback for very few points
            k_safe = min(self.k, len(t) - 1)
            if k_safe < 1: return np.zeros_like(y)
            
            spline = UnivariateSpline(t, y, s=self.s, k=k_safe)
            return spline.derivative()(t)
        except Exception as e:
            print(f"[Warning] Spline fitting failed: {e}. Defaulting to zeros.")
            return np.zeros_like(y)

class PolynomialDerivative(DerivativeEstimator):
    """Polynomial regression-based derivative estimation."""
    def __init__(self, order=3):
        self.order = order
    
    def estimate(self, t, y):
        try:
            order_safe = min(self.order, len(t) - 1)
            coeffs = np.polyfit(t, y, order_safe)
            return np.polyval(np.polyder(coeffs), t)
        except Exception:
            return np.zeros_like(y)  
        
class LagrangeDerivative(DerivativeEstimator):
    """Lagrange interpolatin-based derivative estimation"""
    def estimate(self, t, y):
        try:
            poly = lagrange(t, y)
            deriv_poly = np.polyder(poly)
            return deriv_poly(t)
        except Exception as e:
            print(f"[LagrangeDerivative] Error in Lagrange fitting: {e}. Returning zeros.")
            return np.zeros_like(y)

class FiniteDifferenceDerivative(DerivativeEstimator):
    """Finite difference-based derivative estimation"""
    def estimate(self, t, y):
        return np.gradient(y, t)
    
def get_derivative_estimator(method='spline', **kwargs):
    """Factory function for derivative estimators."""
    estimators = {
        'spline': SplineDerivative,
        'poly': PolynomialDerivative,
        'lagrange': LagrangeDerivative,
        'finite_diff': FiniteDifferenceDerivative
    }
    if method not in estimators:
        raise ValueError(f"Unknown derivative method: {method}. Choose from {list(estimators.keys())}")
    return estimators[method](**kwargs)


# Data Normalization ---
class Normalizer:
    """
    Handles data normalization to ensure stability during neural network training.
    
    Note:
        Neural networks are sensitive to the scale of input features. This class 
        normalizes observed states (Glucose), hidden states (Insulin), and parameters
        to a consistent range (typically [-1, 1] or similar) to prevent gradient explosion
        and accelerate convergence.
    """
    def __init__(self, system, device,
                 config=None, 
                 state_scales=None, param_bounds=None, use_log_params=False,
                 use_normalization=True):
        self.system = system
        self.device = device
        self.use_log_params = use_log_params
        self.use_normalization = use_normalization
        
        # 1. 값 불러오기 (우선순위: Config > Argument > Default)
        _state_scales = getattr(config, 'NORMALIZER_STATE_SCALES', None) if config else None
        _state_scales = _state_scales or state_scales or [1.0, 1.0]
        
        _p_bounds = getattr(config, 'NORMALIZER_PARAM_BOUNDS', None) if config else None
        _p_bounds = _p_bounds or param_bounds
        
        # 2. State Scales 할당
        self.state_scales = torch.tensor(_state_scales, dtype=torch.float32, device=device)
        
        # 3. Parameter Bounds 전처리 (JSON 리스트가 들어와도 에러가 나지 않도록 numpy 배열로 강제 변환)
        if _p_bounds:
            p_min = np.array(_p_bounds[0], dtype=np.float32)
            p_max = np.array(_p_bounds[1], dtype=np.float32)
        else:
            ranges = np.array([self.system.param_ranges[n] for n in self.system.param_names])
            p_min, p_max = ranges[:, 0], ranges[:, 1]

        # 4. Log 변환 및 Tensor 할당
        if self.use_log_params:
            p_min = np.log(p_min + 1e-9)
            p_max = np.log(p_max + 1e-9)
            
        self.p_min = torch.tensor(p_min, dtype=torch.float32, device=device)
        self.p_max = torch.tensor(p_max, dtype=torch.float32, device=device)

    def normalize_inputs(self, x, variable_type='observed'):
        """
        Normalizes observed (X) or hidden (Y) states.
        x: (Batch, Time, Dim) or (Batch, FlatDim)
        """
        if variable_type == 'observed':
            scale = self.state_scales[0] # Glucose scale
        elif variable_type == 'hidden':
            scale = self.state_scales[1] # Insulin scale
        else:
            scale = 1.0
            
        if self.use_normalization:
            return x / (scale + 1e-8)
        else:
            return x

    def denormalize_inputs(self, x_norm, variable_type='observed'):
        if variable_type == 'observed':
            scale = self.state_scales[0]
        elif variable_type == 'hidden':
            scale = self.state_scales[1]
        else:
            scale = 1.0
        if self.use_normalization:    
            return x_norm * scale
        else:
            return x_norm

    def normalize_params(self, p):
        """
        Physical Params -> Log Space -> [-1, 1] Range
        """
        if self.use_log_params:
            p = torch.log(p + 1e-9)
        
        # Min-Max Scaling to [-1, 1]
        p_norm = 2 * (p - self.p_min) / (self.p_max - self.p_min + 1e-8) - 1
        if self.use_normalization:
            return p_norm
        else:
            return p

    def denormalize_params(self, p_norm):
        """
        Inverts the transformation applied by ``normalize_params``.

        When ``use_normalization`` is ``True`` the incoming tensor is assumed
        to lie in ``[-1,1]`` and is mapped back to the original parameter
        range (in log space if ``use_log_params`` is ``True``).  When
        ``use_normalization`` is ``False`` the tensor is returned unchanged
        except that, if ``use_log_params`` is ``True``, it is exponentiated
        to recover the physical value.
        """
        # handle normalization first, then apply log–exp if requested
        if self.use_normalization:
            p_log = (p_norm + 1) / 2 * (self.p_max - self.p_min) + self.p_min
        else:
            p_log = p_norm

        if self.use_log_params:
            return torch.exp(p_log)
        else:
            return p_log
        

# SDE solver
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
    
    # Bounds for Clamping
    if system is not None and hasattr(system, 'state_bounds'):
        lower_bounds, upper_bounds = system.state_bounds
    else:
        lower_bounds, upper_bounds = 1e-6, 1e+3

    # Time Steps Setup
    # Recalculate dt to hit t_end exactly
    n_total_steps = int(np.ceil((t_end - t_start) / dt_sim))
    t_sim_points = np.linspace(t_start, t_end, n_total_steps + 1)
    dt_actual = t_sim_points[1] - t_sim_points[0]
    sqrt_dt = np.sqrt(dt_actual)
    
    # Simulation Loop
    y_curr = np.array(y0)
    y_res = [y_curr.copy()] # (t=0)
    
    for i in range(n_total_steps):
        t_curr = t_sim_points[i]
        
        # Calculate Drifts & Diffusion
        # drift term(f) consists of ode func + bias
        # see `systems/ogtt_simul.py`
        f = np.array(drift_func(t_curr, y_curr, params))    
        G = np.array(diffusion_func(t_curr, y_curr, params))
        
        # Brownian Motion (Wiener Process) dW ~ N(0, sqrt(dt_actual))
        dW = np.random.normal(0, sqrt_dt, size=n_vars)
        
        # SDE Update: Y_{t+dt} = Y_t + f * dt + G * dW
        if G.ndim == 1:
            diffusion_term = G * dW
        else:
            diffusion_term = G @ dW
            
        y_next = y_curr + f * dt_actual + diffusion_term
        # Calmp
        y_next = np.clip(y_next, lower_bounds, upper_bounds)
        
        y_res.append(y_next.copy())
        y_curr = y_next

    y_full = np.array(y_res).T # (n_vars, n_sim_steps+1)
    
    # Interpolation (Resampling at t_eval)
    y_out = np.zeros((n_vars, len(t_eval)))
    for k in range(n_vars):
        y_out[k, :] = np.interp(t_eval, t_sim_points, y_full[k, :])
        
    return y_out

