# data_loader.py
from concurrent.futures import ProcessPoolExecutor
import os
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from scipy import stats
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from utils import euler_maruyama, get_derivative_estimator
from systems.ogtt_simul import OgttSimul


# --- 1. Synthetic Data Generation Utilities ---

# External function for parallel data generation
def _generate_one_sample(args):
    """
    Worker function for parallel data generation.
    Executes one simulation run (with potential augmentations).
    """
    system, config, seed, dist_params, bias_scale, diffusion_scale, aug_factor = args 
    
    np.random.seed(seed)
    
    # 1. Parameter Sampling (Log-Normal Priors)
    if dist_params is not None:
        si = sample_from_lognorm(dist_params['si'])
        sigma_p = sample_from_lognorm(dist_params['sigma'])
        params_list = [si, sigma_p]
        
        # Sample Initial Conditions consistent with priors
        g0 = sample_from_lognorm(dist_params['G0'])
        i0 = sample_from_lognorm(dist_params['I0'])
        
        # Resolve steady-state for hidden delay variables
        from systems.ogtt_simul import OGTTModel, ode_params, sys_params
        temp_model = OGTTModel(ode_params, sys_params, {'si': si, 'sigma': sigma_p})
        n5, n6 = temp_model.find_steady_state_N(g0)
        y0 = [g0, i0, n5, n6]
    else:
        # Fallback: Uniform sampling from ranges
        params_dict = {k: np.random.uniform(*v) for k, v in system.param_ranges.items()}
        params_list = [params_dict[p] for p in system.param_names]
        y0 = system.sample_initial_conditions(params_dict)
    
    # 2. Simulation Loop (SDE Augmentation)
    results_obs = []
    results_hid = []
    
    # Augmentation is only valid for stochastic models
    actual_aug_factor = aug_factor if getattr(config, 'USE_SDE', False) else 1
    
    # Configure System Scaling
    sys_instance = system() if isinstance(system, type) else system
    sys_instance.bias_scale = bias_scale
    sys_instance.diffusion_scale = diffusion_scale
    
    for k in range(actual_aug_factor):
        # Generate Trajectory
        if getattr(config, 'USE_SDE', False):
            # Stochastic Simulation (Euler-Maruyama)
            y_full = euler_maruyama(
                sys_instance.drift_func,
                sys_instance.diffusion_func,
                sys_instance.t_span,
                y0,
                sys_instance.t_points,
                params_list,
                dt_sim=0.01, # High resolution for stability
                system=sys_instance
            )
        else:
            # Deterministic Simulation (ODE)
            sol = solve_ivp(
                fun=lambda t, y: system.ode_func(t, y, params_list),
                t_span=system.t_span,
                y0=y0,
                t_eval=system.t_points 
            )
            y_full = sol.y if sol.success else np.tile(sol.y[:, -1][:, None], (1, len(system.t_points)))

        # 3. Feature Engineering (Derivative feature)
        # If enabled, appends time derivatives (dy/dt) to the state vector.
        if getattr(config, 'USE_LAGRANGIAN', False):
            t_points = sys_instance.t_points
            T = len(t_points)
            y_dot_full = np.zeros_like(y_full)
            for i in range(T):
                t_i = t_points[i]
                y_i = y_full[:, i]
                y_dot_full[:, i] = sys_instance.drift_func(t_i, y_i, params_list)
            y_full = np.concatenate([y_full, y_dot_full], axis=0)

        # Split into Observed and Hidden
        y_full_T = y_full.T
        obs_idx = [sys_instance.observed_var_idx]
        hid_idx = [sys_instance.hidden_var_idx]
        
        # Indices handling for features + derivatives
        if getattr(config, 'USE_LAGRANGIAN', False):
            n_vars = len(y0)
            obs_deriv_idx = [idx + n_vars for idx in obs_idx]
            obs_idx += obs_deriv_idx
        
        X_obs = y_full_T[:, obs_idx]
        Y_hid = y_full_T[:, hid_idx]
        
        results_obs.append(X_obs)
        results_hid.append(Y_hid)
    
    return results_obs, results_hid, [params_list] * actual_aug_factor
    
def sample_from_lognorm(dist_params, size=1, max_retries=100):
    """
    Rejection sampling wrapper for log-normal distribution to ensure physical positivity.
    """
    s = dist_params['s']
    loc = dist_params['loc']
    scale = dist_params['scale']

    samples = np.zeros(size)
    remaining_indices = np.arange(size)

    for _ in range(max_retries):
        if len(remaining_indices) == 0:
            break

        current_n = len(remaining_indices)
        # Scipy rvs returns samples in ORIGINAL scale (already shifted by loc)
        new_samples = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=current_n)
        
        # Check positivity
        valid_mask = new_samples > 1e-6 
        
        # Assign valid samples
        valid_indices = remaining_indices[valid_mask]
        samples[valid_indices] = new_samples[valid_mask]
        
        # Update remaining
        remaining_indices = remaining_indices[~valid_mask]
        
    if len(remaining_indices) > 0:
        # Fallback for extremely rare cases: force absolute value or mean
        # print(f"Warning: {len(remaining_indices)} samples failed rejection sampling. Using absolute values.")
        # Force positive by taking absolute or using mean (scale * exp(s^2/2) is roughly mean for loc=0)
        samples[remaining_indices] = np.abs(stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=len(remaining_indices))) + 1e-6
        
    return samples if size > 1 else samples[0]

class DataGenerator:
    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.dist_params = None

        # Load empirical distribution parameters if available
        dist_file = Path('data/parameters/distribution_params.json')
        if dist_file.exists():
            with open(dist_file, 'r') as f:
                self.dist_params = json.load(f)
            print("Loaded data-driven distribution parameters for sampling.")
        else:
            print("Data-driven distribution parameters file not found. Using uniform sampling.")

    def generate_data(self):
        # Setup paths
        data_root = Path('data')
        save_dir = data_root / self.config.SYSTEM_NAME
        os.makedirs(save_dir, exist_ok=True)

        suffix = "sde" if getattr(self.config, 'USE_SDE', False) else "ode"
        filename = f"augmented_data_{suffix}_{self.config.NUM_SAMPLES}.npz"
        save_path = save_dir / filename

        # Load Cached Data
        if save_path.exists():
            print(f"Loading existing data from {save_path}...")
            try:
                with np.load(save_path) as data:
                    observed_data = data['observed_data']
                    hidden_data = data['hidden']
                    params_data = data['params']
                    t_points = data['t_points']
                print(f"Loaded {len(observed_data)} samples")
                return observed_data, hidden_data, params_data, t_points
            except Exception as e:
                print(f"Failed to load existing data: {e}. Regenerating data...")

        
        # Configure SDE Scaling 
        scale_factor = getattr(self.config, 'SDE_SCALE_FACTORS', {'bias_scale': 1.0, 'diffusion_scale': 1.0})
        bias_scale = getattr(scale_factor, 'bias_scale', 1.0)
        diffusion_scale = getattr(scale_factor, 'diffusion_scale', 1.0)
        
        self.system.bias_scale = bias_scale
        self.system.diffusion_scale = diffusion_scale        
        
        # Parallel Generation
        num_samples = self.config.NUM_SAMPLES
        aug_factor = getattr(self.config, 'AUGMENTATION_FACTOR', 1)
        seeds = np.random.randint(0, 100000, size=num_samples)
        
        args_list = [(self.system, self.config, seeds[i], self.dist_params, bias_scale, diffusion_scale, aug_factor) 
                     for i in range(num_samples)]
        
        num_workers = min(8, os.cpu_count() or 1)
        print(f"Starting data generation with {num_workers} workers...")
        print(f"Generating samples (N={self.config.NUM_SAMPLES}, Aug={aug_factor}) using {suffix.upper()} model")
        print(f"Total # = {self.config.NUM_SAMPLES * aug_factor} ")
        
        

        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_generate_one_sample, args_list))
        print(f"Generation complete in {time.time() - start_time:.2f} seconds.")   

        # Assemble & Save
        observed_data = []
        hidden_data = []
        params_data = []
        
        for res in results:
            # res[0], res[1], res[2]는 각각 길이가 aug_factor인 리스트임
            observed_data.extend(res[0])
            hidden_data.extend(res[1])
            params_data.extend(res[2])

        observed_data = np.array(observed_data)
        hidden_data = np.array(hidden_data)
        params_data = np.array(params_data)
        t_points = np.asarray(self.system.t_points)
        
        # Save
        print(f"Saving data to {save_path}...")
        np.savez_compressed(
            save_path,
            observed_data=observed_data,
            hidden=hidden_data,
            params=params_data,
            t_points=t_points
        )
            
        return observed_data, hidden_data, params_data, t_points


# --- 2. Real Data Loading & Preprocessing ---

class RealOGTTDataLoader:
    """
    Loads and processes sparse NIH OGTT data.
    
    Key Functionality:
    - Loads 5-point OGTT data (0, 30, 60, 90, 120 min).
    - Estimates Lagrangian features (derivatives) using Smoothing Splines.
    - Smoothing parameter 's' is derived from SDE calibration (sigma^2).
    """
    def __init__(self, file_path, config, split_file=None):
        self.file_path = file_path
        self.config = config
        self.split_file = split_file
        self.t_points = np.array([0, 30, 60, 90, 120])

        # Load SDE calibration for smoothing parameter selection
        # Rationale: s approx sum(sigma^2) balances fidelity and noise suppression.
        root_path = Path(__file__).resolve().parent
        sigma_path = root_path / 'data' / 'parameters' / 'calibrated_sde_params.json'
        
        try:
            with open(sigma_path, 'r') as f:
                calib_data = json.load(f)
                self.s_glucose = np.sum(np.array(calib_data['sigma_G'])**2)
                self.s_insulin = np.sum(np.array(calib_data['sigma_I'])**2)
        except FileNotFoundError:
            print("[Warning] SDE params not found. Using default smoothing.")
            self.s_glucose = None
            self.s_insulin = None
            
        method = getattr(config, 'DERIVATIVE_METHOD', 'spline')
        
        kwargs = {}
        if method == 'spline':
            pass
        elif method == 'poly':
            kwargs['order'] = 3
            
        self.derivative_method = method
        self.derivative_kwargs = kwargs
            
            
    def _add_derivative_feature(self, t, y, s_val):
        """
        Estimates derivatives
        """
        N, T = y.shape
        y_aug = np.zeros((N, T, 2)) # [State, Derivative]
        
        if self.derivative_method == 'spline':
            estimator = get_derivative_estimator('spline', s=s_val)
        else:
            estimator = get_derivative_estimator(self.derivative_method, **self.derivative_kwargs)
        
        for i in range(N):
            y_aug[i, :, 0] = y[i, :]
            y_aug[i, :, 1] = estimator.estimate(t, y[i, :])
        
        return y_aug
        
    def load_data(self):
        """Loads data and applies derivative estimation if configured."""
        print(f"Loading real OGTT data from {self.file_path}...")
        try:
            df = pd.read_excel(self.file_path)
        except:
            df = pd.read_csv(self.file_path)

        glu_cols = ['oglu0', 'oglu30', 'oglu60', 'oglu90', 'oglu120']
        ins_cols = ['oins0', 'oins30', 'oins60', 'oins90', 'oins120']
        param_cols = ['si', 'sigma']

        # Clean Data
        df_clean = df[glu_cols + ins_cols + param_cols].dropna()
        
        glucose_raw = df_clean[glu_cols].values
        insulin_raw = df_clean[ins_cols].values
        params_data = df_clean[param_cols].values
        
        # Feature Engineering
        if getattr(self.config, 'USE_LAGRANGIAN', False):
            # Add derivatives to Glucose (Observed)
            observed_data = self._add_derivative_feature(self.t_points, glucose_raw, self.s_glucose)
            # Insulin (Hidden) is kept as is (N, T, 1) for compatibility
            hidden_data = insulin_raw[:, :, np.newaxis]
        else:
            observed_data = glucose_raw[:, :, np.newaxis]
            hidden_data = insulin_raw[:, :, np.newaxis]
        
        return observed_data, hidden_data, params_data, self.t_points
    