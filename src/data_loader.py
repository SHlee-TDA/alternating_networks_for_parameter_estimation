# data_loader.py
import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import torch
from torch.utils.data import (
    TensorDataset, DataLoader, random_split, 
    ConcatDataset, WeightedRandomSampler, Subset
)

from utils import euler_maruyama, get_derivative_estimator, Normalizer


# Simulation Data Generation Utilities
def _generate_one_sample(args):
    """
    Worker function for parallel data generation.
    Executes one simulation run (with potential augmentations).
    """
    system, config, seed, dist_params, bias_scale, diffusion_scale, aug_factor = args 
    
    np.random.seed(seed)
    
    # 1. Parameter Sampling (Log-Normal Priors or Uniform Fallback)
    if dist_params is not None:
        si = sample_from_lognorm(dist_params['si'])
        sigma_p = sample_from_lognorm(dist_params['sigma'])
        params_list = [si, sigma_p]
        
        # Sample Initial Conditions consistent with empirical priors
        g0 = sample_from_lognorm(dist_params['G0'])
        i0 = sample_from_lognorm(dist_params['I0'])
        
        # Resolve steady-state for hidden delay variables (Specific to OGTT)
        from systems.ogtt_simul import OGTTModel, ODE_PARAMS, SYS_PARAMS
        temp_model = OGTTModel(ODE_PARAMS, SYS_PARAMS, {'si': si, 'sigma': sigma_p})
        n5, n6 = temp_model.find_steady_state_N(g0)
        y0 = [g0, i0, n5, n6]
    else:
        # Fallback: Uniform sampling from defined system ranges
        params_dict = {k: np.random.uniform(*v) for k, v in system.param_ranges.items()}
        params_list = [params_dict[p] for p in system.param_names]
        y0 = system.sample_initial_conditions(params_dict)
    
    # 2. Simulation Loop (ODE/SDE Augmentation)
    results_obs = []
    results_hid = []
    
    # Augmentation is strictly applied to stochastic models
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
                dt_sim=0.01, # High resolution for numerical stability
                system=sys_instance
            )
        else:
            # Deterministic Simulation (ODE via SciPy)
            sol = solve_ivp(
                fun=lambda t, y: system.ode_func(t, y, params_list),
                t_span=system.t_span,
                y0=y0,
                t_eval=system.t_points 
            )
            # Handle integration failures gracefully
            y_full = sol.y if sol.success else np.tile(sol.y[:, -1][:, None], (1, len(system.t_points)))

        # 3. Feature Engineering (Derivative injection)
        # Appends time derivatives (dy/dt) to the state vector.
        if getattr(config, 'USE_DERIVATIVE', False):
            t_points = sys_instance.t_points
            T = len(t_points)
            y_dot_full = np.zeros_like(y_full)
            for i in range(T):
                t_i = t_points[i]
                y_i = y_full[:, i]
                y_dot_full[:, i] = sys_instance.drift_func(t_i, y_i, params_list)
            y_full = np.concatenate([y_full, y_dot_full], axis=0)

        # Split into Observed and Hidden sets
        y_full_T = y_full.T
        obs_idx = [sys_instance.observed_var_idx]
        hid_idx = [sys_instance.hidden_var_idx]
        
        # Adjust indices if derivative features are active
        if getattr(config, 'USE_DERIVATIVE', False):
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
    Rejection sampling wrapper for log-normal distribution to ensure strict positivity.
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
        new_samples = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=current_n)
        
        # Enforce positivity boundary
        valid_mask = new_samples > 1e-6 
        valid_indices = remaining_indices[valid_mask]
        samples[valid_indices] = new_samples[valid_mask]
        
        remaining_indices = remaining_indices[~valid_mask]
        
    if len(remaining_indices) > 0:
        # Fallback for extreme tails: force absolute shift
        samples[remaining_indices] = np.abs(stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=len(remaining_indices))) + 1e-6
        
    return samples if size > 1 else samples[0]

class DataGenerator:
    """Orchestrates the large-scale parallel generation of dynamical system datasets."""
    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.dist_params = None

        # Load empirical distribution priors for domain-specific models
        if self.config.SYSTEM_NAME == 'ogtt_simul':
            dist_file = Path('data/parameters/distribution_params.json')
            if dist_file.exists():
                with open(dist_file, 'r') as f:
                    self.dist_params = json.load(f)
                print("Loaded data-driven distribution parameters for sampling.")
            else:
                print("Data-driven distribution parameters file not found. Using uniform sampling.")
        else:
            print("Using default uniform sampling for parameters and initial conditions.")

    def generate_data(self):
        """Generates or loads cached simulation data."""
        data_root = Path('data')
        save_dir = data_root / self.config.SYSTEM_NAME
        os.makedirs(save_dir, exist_ok=True)

        suffix = "sde" if getattr(self.config, 'USE_SDE', False) else "ode"
        deriv_suffix = "_deriv" if getattr(self.config, 'USE_DERIVATIVE', False) else "_noderiv"
        filename = f"augmented_data_{suffix}{deriv_suffix}_{self.config.NUM_SAMPLES}.npz"        
        save_path = save_dir / filename

        # 1. Load Cached Data if available
        if save_path.exists():
            print(f"Loading cached data from {save_path}...")
            try:
                with np.load(save_path) as data:
                    observed_data = data['observed_data']
                    hidden_data = data['hidden']
                    params_data = data['params']
                    t_points = data['t_points']
                print(f"  -> Successfully loaded {len(observed_data)} samples.")
                return observed_data, hidden_data, params_data, t_points
            except Exception as e:
                print(f"  -> Warning: Failed to load cache ({e}). Regenerating...")

        # 2. Configure Noise/SDE Scaling 
        scale_factor = getattr(self.config, 'SDE_SCALE_FACTORS', {'bias_scale': 1.0, 'diffusion_scale': 1.0})
        bias_scale = scale_factor.get('bias_scale', 1.0)
        diffusion_scale = scale_factor.get('diffusion_scale', 1.0)
        self.system.bias_scale = bias_scale
        self.system.diffusion_scale = diffusion_scale        
        
        # 3. Prepare Parallel Execution
        num_samples = self.config.NUM_SAMPLES
        aug_factor = getattr(self.config, 'AUGMENTATION_FACTOR', 1)
        seeds = np.random.randint(0, 100000, size=num_samples)
        
        args_list = [(self.system, self.config, seeds[i], self.dist_params, bias_scale, diffusion_scale, aug_factor) 
                     for i in range(num_samples)]
        
        num_workers = min(8, os.cpu_count() or 1)
        # Optmization: Use chunksize to drastically reduce inter-process communication overhead
        optimal_chunksize = max(1, num_samples // (num_workers * 4))
        
        print(f"Starting parallel generation ({num_workers} workers, chunksize={optimal_chunksize})...")
        print(f"Target: {self.config.NUM_SAMPLES * aug_factor} total samples ({suffix.upper()} model)")
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_generate_one_sample, args_list, chunksize=optimal_chunksize))
        print(f"Generation complete in {time.time() - start_time:.2f} seconds.")   

        # 4. Assemble Results
        observed_data, hidden_data, params_data = [], [], []
        for res in results:
            observed_data.extend(res[0])
            hidden_data.extend(res[1])
            params_data.extend(res[2])

        observed_data = np.array(observed_data)
        hidden_data = np.array(hidden_data)
        params_data = np.array(params_data)
        t_points = np.asarray(self.system.t_points)
        
        # 5. Save to Cache
        print(f"Saving data to {save_path}...")
        np.savez_compressed(
            save_path,
            observed_data=observed_data,
            hidden=hidden_data,
            params=params_data,
            t_points=t_points
        )
            
        return observed_data, hidden_data, params_data, t_points



# Real Data Loading & Preprocessing
class RealOGTTDataLoader:
    """
    Loads and processes sparse NIH clinical OGTT data.
    
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

        # Load SDE calibration for optimal smoothing parameter selection
        root_path = Path(__file__).resolve().parent
        sigma_path = root_path / 'data' / 'parameters' / 'calibrated_sde_params.json'
        
        try:
            with open(sigma_path, 'r') as f:
                calib_data = json.load(f)
                self.s_glucose = np.sum(np.array(calib_data['sigma_G'])**2)
                self.s_insulin = np.sum(np.array(calib_data['sigma_I'])**2)
        except FileNotFoundError:
            print("[Warning] SDE params not found. Using default internal smoothing.")
            self.s_glucose = None
            self.s_insulin = None
            
        method = getattr(config, 'DERIVATIVE_METHOD', None)
        kwargs = {}
        if method == 'poly':
            kwargs['order'] = 3
            
        self.derivative_method = method
        self.derivative_kwargs = kwargs
            
    def _add_derivative_feature(self, t, y, s_val):
        """Estimates temporal derivatives to append as state features."""
        N, T = y.shape
        y_aug = np.zeros((N, T, 2)) # Shape: [Samples, Time, (State, Derivative)]
        
        if self.derivative_method == 'spline':
            estimator = get_derivative_estimator('spline', s=s_val)
        else:
            estimator = get_derivative_estimator(self.derivative_method, **self.derivative_kwargs)
        
        for i in range(N):
            y_aug[i, :, 0] = y[i, :]
            y_aug[i, :, 1] = estimator.estimate(t, y[i, :])
        
        return y_aug
        
    def load_data(self):
        """Loads data and optionally engineers derivative features."""
        print(f"Loading real clinical data from {self.file_path}...")
        try:
            df = pd.read_excel(self.file_path)
        except Exception:
            df = pd.read_csv(self.file_path)

        glu_cols = ['oglu0', 'oglu30', 'oglu60', 'oglu90', 'oglu120']
        ins_cols = ['oins0', 'oins30', 'oins60', 'oins90', 'oins120']
        param_cols = ['si', 'sigma']

        # Clean Data (Drop missing values)
        df_clean = df[glu_cols + ins_cols + param_cols].dropna()
        
        glucose_raw = df_clean[glu_cols].values
        insulin_raw = df_clean[ins_cols].values
        params_data = df_clean[param_cols].values
        
        # Feature Engineering
        if getattr(self.config, 'USE_DERIVATIVE', False):
            # Add derivatives strictly to Observed variable (Glucose)
            observed_data = self._add_derivative_feature(self.t_points, glucose_raw, self.s_glucose)
            hidden_data = insulin_raw[:, :, np.newaxis]
        else:
            observed_data = glucose_raw[:, :, np.newaxis]
            hidden_data = insulin_raw[:, :, np.newaxis]
        
        return observed_data, hidden_data, params_data, self.t_points



# DataLoader Setup Module for experiments
# data_loader.py (Update setup_dataloaders function)

def setup_dataloaders(exp_config, sim_data_tuple, system, global_config):
    """
    Creates PyTorch DataLoaders for Training, Validation, and Testing.
    Ensures strict data separation to prevent data leakage and handles
    robust scaling (normalization) of the variables.

    Returns:
        train_loader, val_loader, test_loader, real_test_loader, p_initial_guess, normalizer
    """
    obs_sim, hid_sim, params_sim, t_points = sim_data_tuple
    total_len = len(obs_sim)

    # ---------------------------------------------------------
    # Step 1: Split Data First (Prevents Data Leakage)
    # We split the raw data before normalization so that the test set
    # does not secretly influence our scaling rules.
    # ---------------------------------------------------------
    g = torch.Generator().manual_seed(global_config.SEED)
    shuffled_indices = torch.randperm(total_len, generator=g).tolist()
    
    test_len = int(total_len * global_config.TEST_SPLIT)
    val_len = int(total_len * global_config.TEST_SPLIT)
    train_len = total_len - val_len - test_len
    
    # Safely convert to numpy arrays for indexing
    train_idx = np.array(shuffled_indices[:train_len])
    val_idx = np.array(shuffled_indices[train_len:train_len+val_len])
    test_idx = np.array(shuffled_indices[train_len+val_len:])
    
    # Isolate splits
    obs_train, obs_val, obs_test = obs_sim[train_idx], obs_sim[val_idx], obs_sim[test_idx]
    hid_train, hid_val, hid_test = hid_sim[train_idx], hid_sim[val_idx], hid_sim[test_idx]
    params_train, params_val, params_test = params_sim[train_idx], params_sim[val_idx], params_sim[test_idx]

    # ---------------------------------------------------------
    # Step 2: Initialize Normalizer (Using Train Data Only)
    # Scales data to a stable range (e.g., [-1, 1]) for neural networks.
    # ---------------------------------------------------------
    use_log = (global_config.SYSTEM_NAME == 'ogtt_simul')
    use_normalization = (global_config.SYSTEM_NAME == 'ogtt_simul')
    
    # Check if we are loading a previously saved experiment
    loaded_scales = getattr(global_config, 'NORMALIZER_STATE_SCALES', None)
    loaded_bounds = getattr(global_config, 'NORMALIZER_PARAM_BOUNDS', None)

    if loaded_scales and loaded_bounds:
        print("  -> [Normalizer] Loading pre-calculated scales from config.")
        calc_scales = loaded_scales
        p_bounds = (np.array(loaded_bounds[0]), np.array(loaded_bounds[1]))
    else:
        print("  -> [Normalizer] Calculating scales safely from the training split.")
        # Use 99.9th percentile to ignore extreme outliers, with a 1e-6 safety minimum
        scale_obs = max(np.percentile(np.abs(obs_train), 99.9), 1e-6)
        scale_hid = max(np.percentile(np.abs(hid_train), 99.9), 1e-6)
        calc_scales = [float(scale_obs * 1.2), float(scale_hid * 1.2)] # 20% margin
        
        # Calculate safe boundaries for parameters (10% margin on the range)
        p_min = np.min(params_train, axis=0)
        p_max = np.max(params_train, axis=0)
        p_range = p_max - p_min
        p_bounds = (p_min - 0.1 * p_range, p_max + 0.1 * p_range)
        
        # Save these rules to config so they can be re-used during inference/testing
        global_config.NORMALIZER_STATE_SCALES = calc_scales
        global_config.NORMALIZER_PARAM_BOUNDS = [p_bounds[0].tolist(), p_bounds[1].tolist()]

    normalizer = Normalizer(
        system, 
        global_config.DEVICE, 
        config=global_config,
        state_scales=calc_scales, 
        param_bounds=p_bounds,
        use_log_params=use_log,
        use_normalization=use_normalization
    )
    
    # ---------------------------------------------------------
    # Step 3: Build Normalized Datasets
    # We keep data on CPU here to prevent GPU Out-Of-Memory (OOM) errors.
    # ---------------------------------------------------------
    def _build_normalized_dataset(obs, hid, params):
        X = torch.FloatTensor(obs).view(len(obs), -1)
        Y = torch.FloatTensor(hid).view(len(hid), -1)
        P = torch.FloatTensor(params)
        return TensorDataset(
            normalizer.normalize_inputs(X, 'observed'),
            normalizer.normalize_inputs(Y, 'hidden'),
            normalizer.normalize_params(P)
        )
        
    sim_train = _build_normalized_dataset(obs_train, hid_train, params_train)
    sim_val = _build_normalized_dataset(obs_val, hid_val, params_val)
    sim_test = _build_normalized_dataset(obs_test, hid_test, params_test)

    # ---------------------------------------------------------
    # Step 4: Prepare Real Clinical Data (If applicable)
    # Applies the exact same normalization rules learned from simulation.
    # ---------------------------------------------------------
    real_train, real_test = None, None
    if global_config.SYSTEM_NAME == 'ogtt_simul':
        real_loader = RealOGTTDataLoader(
            file_path='data/clean_sumner_n_612.xlsx', 
            config=global_config,
            split_file='data/data_split_indices.json'
        )
        obs_real, hid_real, params_real, _ = real_loader.load_data()
        
        X_real = torch.FloatTensor(obs_real).view(len(obs_real), -1)
        Y_real = torch.FloatTensor(hid_real).view(len(hid_real), -1)
        P_real = torch.FloatTensor(params_real)
        
        dataset_real = TensorDataset(
            normalizer.normalize_inputs(X_real, 'observed'),
            normalizer.normalize_inputs(Y_real, 'hidden'),
            normalizer.normalize_params(P_real)
        )
        
        with open('data/data_split_indices.json', 'r') as f:
            split_data = json.load(f)
        real_train = Subset(dataset_real, split_data['train_indices'])
        real_test = Subset(dataset_real, split_data['test_indices'])

    # ---------------------------------------------------------
    # Step 5: Finalize DataLoaders
    # Routes to Hybrid (Sim + Real) or Simulation-Only mode.
    # ---------------------------------------------------------
    scenario = exp_config.get('SCENARIO', 'sim_only')
    
    if scenario == 'hybrid' and real_train is not None:
        print(f"  -> [Hybrid Mode] Mixing Simulation ({len(sim_train)}) and Real Data ({len(real_train)})")
        final_train_set = ConcatDataset([sim_train, real_train])
        
        # Oversample the smaller real dataset to balance training ratios
        real_ratio = exp_config.get('REAL_RATIO', 0.3)
        w_sim = (1 - real_ratio) / len(sim_train)
        w_real = real_ratio / len(real_train)
        weights = [w_sim] * len(sim_train) + [w_real] * len(real_train)
        
        sampler = WeightedRandomSampler(weights, num_samples=len(final_train_set), replacement=True)
        train_loader = DataLoader(final_train_set, batch_size=global_config.BATCH_SIZE, sampler=sampler)
    else:
        print(f"  -> [Simulation Mode] Using Pure Sim Data ({len(sim_train)} samples)")
        train_loader = DataLoader(sim_train, batch_size=global_config.BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(sim_val, batch_size=global_config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(sim_test, batch_size=global_config.BATCH_SIZE, shuffle=False)
    
    # ---------------------------------------------------------
    # Step 6: Initial Parameter Guess (For inference speedup)
    # Calculates the geometric mean of training parameters safely.
    # ---------------------------------------------------------
    params_train_tensor = torch.FloatTensor(params_train)
    P_sim_safe = torch.clamp(params_train_tensor, min=1e-6)
    p_initial_guess = torch.exp(torch.log(P_sim_safe).mean(dim=0)).to(global_config.DEVICE)
    
    return train_loader, val_loader, test_loader, real_test, p_initial_guess, normalizer