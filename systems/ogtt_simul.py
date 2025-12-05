# systems/ogtt_simul.py
"""
systems/ogtt_simul.py
================================================================================
Implements the Oral Glucose Tolerance Test (OGTT) simulation environment.

This module defines the dynamical system for glucose-insulin interaction, 
incorporating both deterministic ODEs and stochastic SDE components.

Key Components:
1. OgttSimul: The main system interface for generating trajectories and 
   defining SDE drift/diffusion terms.
2. OGTTModel: The core differential equation solver representing the biological 
   mechanisms (e.g., insulin secretion, hepatic glucose production).

References:
- The model is based on the physiological diagram of glucose-insulin dynamics.
- SDE terms are calibrated using real-world data residuals (see analysis/).
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import scipy.stats as stats
from scipy.interpolate import CubicSpline

from .base_system import System

# --- Configuration & ODE Parameter Loading ---
def load_json_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CONFIG = load_json_config(CURRENT_DIR / 'config.json')
SYS_PARAMS = load_json_config(CURRENT_DIR / 'system_params.json')
ODE_PARAMS = CONFIG['ode_params']

# SDE Calibration Data Loading
# Loads empirically estimated diffusion (sigma) and drift bias terms.
SDE_PARAM_PATH = PROJECT_ROOT / 'data' / 'parameters' / 'calibrated_sde_params.json'

try:
    calib_data = load_json_config(SDE_PARAM_PATH)
    SIGMA_T_POINTS = np.array(calib_data.get('t_points', [0, 120]))
    # Diffusion terms
    SIGMA_G_T = np.array(calib_data.get('sigma_G', [0.0, 0.0]))
    SIGMA_I_T = np.array(calib_data.get('sigma_I', [0.0, 0.0]))
    # Drift bias terms
    MU_G_T = np.array(calib_data.get('mu_G', [0.0, 0.0]))
    MU_I_T = np.array(calib_data.get('mu_I', [0.0, 0.0]))
    BOUNDS_MAP = calib_data.get('bounds', {'G_max': 1e9, 'I_max': 1e9})
except FileNotFoundError:
    print(f"[Warning] SDE parameters not found at {SDE_PARAM_PATH}. Using default zeros.")
    SIGMA_T_POINTS = np.array([0, 120])
    SIGMA_G_T = np. zeros(2)
    SIGMA_I_T = np.zeros(2)
    MU_G_T, MU_I_T = np.zeros(2), np.zeros(2)
    BOUNDS_MAP = {'G_max': 1e9, 'I_max': 1e9}


class OgttSimul(System):
    """
    Defines the Stochastic Differential Equation (SDE) system for OGTT.
    
    System State (y):
        y[0] (G): Glucose concentration
        y[1] (I): Insulin concentration
        y[2] (N5): Delayed signal for insulin secretion (Incretin effect)
        y[3] (N6): Delayed signal for insulin secretion (Incretin effect)
        
    Parameters:
        si: Insulin sensitivity
        sigma: Generalized parameter related to secretion capacity
    """
    name='ogtt_simul'
    param_names = ['si', 'sigma']
    param_ranges = {
        'si': [0.0, 2.0],   
        'sigma': [0.0, 2.0]      
    }
    
    # Heuristic initial conditions range (minG, maxG), (minI, maxI)
    initial_conditions = ([80.0, 120.0], 
                         [10.0, 20.0])  
    t_span = [0, 120]    
    t_points = np.array([0, 30, 60, 90, 120])
    
    # Indices for observed vs hidden variables
    observed_var_idx = 0  # Glucose is observed
    hidden_var_idx = 1    # Insulin is considered 'hidden' in this experimental setup
    
    # Scaling factors for SDE terms
    bias_scale = 1.0
    diffusion_scale = 1.0
    
    def __init__(self):
        super().__init__()
        # Optimization: Pre-instantiate model to avoid overhead in loops
        self.model = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': 0.0, 'sigma': 0.0})
        
        # Cubic spline interpolators for smooth SDE terms        
        self.sigma_g_spline = CubicSpline(SIGMA_T_POINTS, SIGMA_G_T, bc_type='natural')
        self.sigma_i_spline = CubicSpline(SIGMA_T_POINTS, SIGMA_I_T, bc_type='natural')
        self.mu_g_spline = CubicSpline(SIGMA_T_POINTS, MU_G_T, bc_type='natural')
        self.mu_i_spline = CubicSpline(SIGMA_T_POINTS, MU_I_T, bc_type='natural')
        
    def sample_initial_conditions(self, params_dict):
        """
        Samples initial conditions (G0, I0) from log-normal priors derived from NIH data,
        and computes the corresponding steady-state values for N5 and N6.
        """        
        # Priors for G(0) and I(0) derived from NIH data
        oglu0 = stats.lognorm.rvs(s=0.1196, loc=0.0, scale=90.9547)
        oins0 = stats.lognorm.rvs(s=0.6901, loc=0.0, scale=5.9133)
        
        # Update model params to find consistent steady state
        self.model.theta['si'] = params_dict['si']
        self.model.theta['sigma'] = params_dict['sigma']
        
        # Solve algebraic equilibrium for auxiliary variables
        n5_ss, n6_ss = self.model.find_steady_state_N(oglu0)
        
        return [oglu0, oins0, n5_ss, n6_ss]
    
    def ode_func(self, t, y, params):
        si, sigma = params
        
        self.model.theta['si'] = si
        self.model.theta['sigma'] = sigma

        G, I, N5, N6 = y
        dydt = self.model.GI_ode_universal(t, [G, I, N5, N6])
        return dydt

    def drift_func(self, t, y, params):
        """
        SDE Drift Term: F(t, y) = f_ode(t, y) + mu_bias(t)
        
        Adds a data-driven bias correction term (mu) to the physical ODE drift.
        This accounts for structural model mismatch (misspecification).
        """
        dydt_ode = self.ode_func(t, y, params)
        
        # Add bias correction
        dydt_corrected = list(dydt_ode)
        dydt_corrected[0] += self.bias_scale * self.mu_g_spline(t) # Glucose bias
        dydt_corrected[1] += self.bias_scale * self.mu_i_spline(t) # Insulin bias
        
        return dydt_corrected
    
    def diffusion_func(self, t, y, params):
        """
        SDE Diffusion Term: G(t, y)
        
        Models the stochastic diversity of Glucose and Insulin.
        Returns a diagonal matrix assuming uncorrelated noise between states.
        """
        n_vars = 4
        diffusion_matrix = np.zeros((n_vars, n_vars))
        
        # Apply calibrated time-dependent volatility
        diffusion_matrix[0, 0] = self.sigma_g_spline(t) * self.diffusion_scale
        diffusion_matrix[1, 1] = self.sigma_i_spline(t) * self.diffusion_scale
        
        # N5, N6 are assumed to be deterministic (zero diffusion)
        return diffusion_matrix
    
    @property
    def state_bounds(self):
        """Physical constraints for state variables [G, I, N5, N6]."""
        lower = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        upper = np.array([
            BOUNDS_MAP.get('G_max', 1e9), 
            BOUNDS_MAP.get('I_max', 1e9), 
            1e9, 
            1e9
        ])
        return lower, upper

class OGTTModel:
    """
    Physiological model of Glucose-Insulin dynamics.
    
    Equations encapsulate:
    - Glucose appearing from oral ingestion (OGTT_rate)
    - Hepatic Glucose Production (HGP)
    - Insulin-dependent glucose uptake
    - Insulin secretion stimulated by glucose and incretins (N5, N6)
    """
    def __init__(self, ode_params, sys_params, theta):
        self.ode_params = ode_params
        self.sys_params = sys_params
        self.theta = theta

    def GI_ode_universal(self, t, y):
        """
        dG/dt = HGP + OGTT_rate - (Eg0 + Si * I) * G
        dI/dt = (b * ISR) / BV - k * I
        dN5/dt, dN6/dt : Incretin dynamics (delay chain)
        """
        G, I, N5, N6 = y
        p_sys = self.sys_params
        si = self.theta['si']

        # 1. Flux Calculations
        M = self.calculate_metabolic_rate(G)
        OGTT_rate = self.calculate_ogtt_flux(t)
        HGP = self.calculate_HGP(I)
        GF = self.calculate_GF(G) 

        # 2. Calcium & Secretion Dynamics
        ci = self.calculate_ci(M)
        cmd = self.calculate_cmd(ci)
        CN = self.calculate_CN(cmd)
        ISR = self.calculate_ISR(CN, N5)

        # 3. ODEs
        # Glucose Dynamics
        dGdt = HGP + OGTT_rate - (p_sys['Eg0'] + p_sys['unit_con'] * si * I) * G
        
        # Insulin Dynamics
        dIdt = (p_sys['b'] * ISR) / p_sys['BV'] - p_sys['k'] * I
        
        # Incretin/Signal Delay Dynamics (N5, N6)
        r2 = self.calculate_r2(ci)
        r3 = self.calculate_r3(ci, GF)
        rm1, rm2, rm3 = p_sys['rm1'], p_sys['rm2'], p_sys['rm3']
        r1 = p_sys['r1']
        
        dN5dt = p_sys['ts'] * (rm1 * CN[0] * N5 - (r1 + rm2) * N5 + r2 * N6)
        dN6dt = p_sys['ts'] * (r3 + rm2 * N5 - (rm3 + r2) * N6)

        return dGdt, dIdt, dN5dt, dN6dt

    def simulate(self, t_span, initial_conditions, t_eval=None):
        if t_eval is None:
            t_eval = np.linspace(0, 120, 121)

        return solve_ivp(
            self.GI_ode_universal,
            t_span,
            initial_conditions,
            method='BDF', # Stiff solver is recommended for biological systems
            t_eval=t_eval
        )

    # --- Sub-process Calculations (Helper Methods) ---
    # Note: These methods implement specific physiological transfer functions.
    
    def calculate_metabolic_rate(self, G):
        p = self.sys_params
        return p['Mmax'] * G**p['kM'] / (p['alpha_M']**p['kM'] + G**p['kM'])

    def calculate_ogtt_flux(self, t):
        """Computes oral glucose appearance rate (triangular/trapezoidal profile)."""
        p = self.ode_params
        p_sys = self.sys_params
        
        # Vectorized implementation for efficiency
        t1, t2, t3 = p['t1'], p['t2'], p['t3']
        a1, a2, a3 = p['a1'], p['a2'], p['a3']
        
        condlist = [(t > 0) & (t <= t1), (t > t1) & (t <= t2), (t > t2) & (t <= t3)]
        choicelist = [
            t * a1 / t1,
            ((t - t2) * (a2 - a1) / (t2 - t1)) + a2,
            (t - t3) * (a3 - a2) / (t3 - t2)
        ]
        flux = np.select(condlist, choicelist, default=0.0) if not np.isscalar(t) else \
               (t * a1 / t1 if 0 < t <= t1 else 
               ((t - t2) * (a2 - a1) / (t2 - t1)) + a2 if t1 < t <= t2 else 
               (t - t3) * (a3 - a2) / (t3 - t2) if t2 < t <= t3 else 0.0)
               
        return p_sys['OGTT_bar'] * flux

    def calculate_HGP(self, I):
        """Hepatic Glucose Production: Suppressed by Insulin."""
        p, sys = self.ode_params, self.sys_params
        si = self.theta['si']
        
        hepa_max = sys['hepa_bar'] / (sys['hepa_k'] + si) + sys['hepa_b']
        alpha_HGP = sys['alpha_max'] / (sys['alpha_k'] + si) + sys['alpha_b']
        
        return hepa_max / (alpha_HGP + p['hepasi'] * I) + sys['HGP_b']
    
    def calculate_GF(self, G):
        """Glucose Amplifying Factor (Potentiation)."""
        p = self.sys_params
        G_eff = G - p['shGF']
        num = p['GF_bar'] * G_eff**p['kGF']
        den = p['alpha_GF']**p['kGF'] + G_eff**p['kGF']
        return num / den + p['GF_b']

    def calculate_ci(self, M):
        p, ode = self.sys_params, self.ode_params
        ci_in = M + ode['gamma_bar'] * ode['gamma']
        return (p['ca_bar'] * ci_in**p['kca']) / (p['alpha_ca']**p['kca'] + ci_in**p['kca']) + p['ca_b']

    def calculate_cmd(self, ci):
        p = self.sys_params
        return (p['cmd_factor'] * ci**p['cik']) / (p['cialpha']**p['cik'] + ci**p['cik']) + p['cmd_b']

    def calculate_r2(self, ci):
        return self.ode_params['r20'] * ci / (ci + self.sys_params['Kp2'])

    def calculate_r3(self, ci, GF):
        return self.theta['sigma'] * GF * self.sys_params['r30'] * ci / (ci + self.sys_params['Kp2'])

    def calculate_CN(self, cmd):
        """Computes fraction of vesicles in different pools using fast-slow analysis."""
        p = self.sys_params
        k1_cmd = p['k1'] * cmd
        
        # Coefficients derived from equilibrium assumption on fast variables
        N3_L = 2 * k1_cmd / (2 * p['km1'] + k1_cmd)
        N3_N = 3 * p['km1'] / (2 * p['km1'] + k1_cmd)
        
        CN4 = k1_cmd / (3 * p['km1'] + p['u1'])
        CN3 = N3_L / (1 - N3_N * CN4)
        
        N2_E = 3 * k1_cmd / (2 * k1_cmd + p['km1'])
        N2_F = 2 * p['km1'] / (2 * k1_cmd + p['km1'])
        CN2 = N2_E / (1 - N2_F * CN3)
        
        N1_D = p['r1'] / (3 * k1_cmd + p['rm1'])
        N1_C = p['km1'] / (3 * k1_cmd + p['rm1'])
        CN1 = N1_D / (1 - N1_C * CN2)
        
        return (CN1, CN2, CN3, CN4)

    def calculate_ISR(self, CN, N5):
        """Insulin Secretion Rate."""
        p = self.sys_params
        # Chain of vesicle pools
        N4 = CN[3] * (CN[2] * (CN[1] * (CN[0] * N5))) # Nested multiplication
        NF = p['u1'] * N4 / p['u2']
        NR = (p['u2'] / p['u3']) * NF
        return p['ts'] * 9 * (p['u3'] * NR)

    def find_steady_state_N(self, oglu0):
        """
        Solves for the steady-state values of N5 and N6 given a baseline Glucose G(0).
        Ensures that the simulation starts from a biological equilibrium.
        """
        # Calculate static factors based on G(0)
        M = self.calculate_metabolic_rate(oglu0)
        ci = self.calculate_ci(M)
        GF = self.calculate_GF(oglu0)
        cmd = self.calculate_cmd(ci)
        CN1 = self.calculate_CN(cmd)[0]
        
        # Parameters
        p = self.sys_params
        r2 = self.calculate_r2(ci)
        r3 = self.calculate_r3(ci, GF)
        
        # Linear system Ax = b for [N5_ss, N6_ss]
        # Derived from dN5/dt = 0, dN6/dt = 0
        A = np.array([
            [p['rm1'] * CN1 - (p['r1'] + p['rm2']), r2],
            [p['rm2'], -(p['rm3'] + r2)]
        ])
        b = np.array([0, -r3])
        
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1.0, 0.5 # Fallback defaults
