# config.py
"""
Project Configuration Registry

This module centralizes all hyperparameters and settings for:
1. System & Environment (Device, Seeds)
2. Data Generation (ODE/SDE settings, Augmentation)
3. Model Architecture (Hidden dims, Activation)
4. Training Protocol (LR, Epochs, Early Stopping)
5. Experiment Scenarios (List of experiments to run)
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import torch

@dataclass
class Config:
    """
    Main Configuration Class.
    Usage:
        config = Config()
        print(config.LEARNING_RATE)
    """

    # --------------------------------------------------------------------------
    # 1. System & Environment
    # --------------------------------------------------------------------------
    SYSTEM_NAME: str = 'lotka_volterra' # 'lotka_volterra', 'sir', 'nc_sir', 'ogtt_simul'
    EXPERIMENT_NAME: str = 'baseline_comparison'
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR: str = 'results/baseline_comparison'
    RUN_BASELINE: bool = False  # Whether to run the single-network baseline for comparison
    # --------------------------------------------------------------------------
    # 2. Data Generation and Data Loading
    # -------------------------------------------------------------------------- 
    NUM_SAMPLES: int = 50000
    AUGMENTATION_FACTOR: int = 0   # Resampling number for SDE simulation
    SDE_SCALE_FACTORS: Dict[str, float] = field(default_factory=lambda: {
        'bias_scale': 1.0,
        'diffusion_scale': 1.27
    }
    )
    TEST_SPLIT: float = 0.2
    BATCH_SIZE: int = 256
    
    # --------------------------------------------------------------------------
    # 3. Trainig Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS: int = 10000
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    USE_EARLY_STOPPING: bool = True
    EARLY_STOPPING_PATIENCE: int = 200
    EARLY_STOPPING_MIN_DELTA: float = 1e-6
    
    USE_DERIVATIVE: bool = True       # Whether to use derivative features in state variable input
    DERIVATIVE_METHOD: str = 'spline'  # 'finite_diff', 'spline', 'lagrange', 'polynomial'
    
    LOSS_CONFIG: List[Tuple[str, float]] = field(default_factory=lambda: [
        ('supervised', 1.0),
        #('recurrent', 1.0)
    ])
    
    ITERATIONS: int = 10              # Number of iterations for parameter estimation during inference
    
    # --------------------------------------------------------------------------
    # 4. Model Architecture
    # --------------------------------------------------------------------------
    MODEL_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_net': {
            'hidden_dims': [64, 64, 64, 64],
            'activation': 'SiLU'
        },
        'param_net': {
            'hidden_dims': [64, 64, 64, 64],
            'activation': 'SiLU'
        }
    })
    
    USE_SPECTRAL_NORM: bool = True    # Whether to apply Spectral Normalization in models
    
    # --------------------------------------------------------------------------
    # 5. Experiment Scenarios
    # --------------------------------------------------------------------------
    SCENARIOS: List[Dict[str, Any]] = field(default_factory=lambda:
        [
            # Scenario 1: Training only on ODE data
            {
                'NAME': 'ode_only',
                'USE_SDE': False,
                'SCENARIO': 'sim_only',
                'VAL_SOURCE': 'sim'
            },
            
            # Scenario 2: Training only on SDE data
            # {
            #     'NAME': 'sde_only',
            #     'USE_SDE': True,
            #     'SCENARIO': 'sim_only',
            #     'VAL_SOURCE': 'sim'
            # }
        ])
    EXPERIMENTS: List[Dict[str, Any]] = field(init=False)
    
    def __post_init__(self):
        """
        Post-initialization to generate experiment configurations
        based on base scenarios and model/loss variants.
        """
        self.EXPERIMENTS = []
        main_loss_name = self.LOSS_CONFIG[-1][0] if self.LOSS_CONFIG else "vanilla"
        for scen in self.SCENARIOS:
            exp = scen.copy()
            
            exp['NAME'] = f"{scen['NAME']}_{main_loss_name}"
            self.EXPERIMENTS.append(exp)