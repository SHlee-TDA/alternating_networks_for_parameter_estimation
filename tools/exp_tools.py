import os
import json
import pandas as pd
from collections import defaultdict
import random
from datetime import datetime
import importlib

import numpy as np
import torch

def set_seed(seed):
    """Fixes random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentLogger:
    """Manages experiment directory creation and config saving."""
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.exp_dir_name = f"{self.timestamp}_{config.EXPERIMENT_NAME}"
        self.results_dir = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, self.exp_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)

        self._save_config()

    def _save_config(self):
        config_dict = {}
        for key in dir(self.config):
            if key.startswith('__'): 
                continue
            
            val = getattr(self.config, key)
            if callable(val): 
                continue
            
            if key == 'DEVICE':
                val = str(val)
            
            config_dict[key] = val
            
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

    def log_result_to_csv(self, metrics_dict):
        registry_path = os.path.join(self.config.RESULTS_DIR, 'experiment_registry.csv')
        
        log_data = {
            'timestamp': self.timestamp,
            'system': self.config.SYSTEM_NAME,
            'experiment': self.config.EXPERIMENT_NAME,
            'use_sde': getattr(self.config, 'USE_SDE', False),
            'use_derivative': getattr(self.config, 'USE_DERIVATIVE', False)
        }
        log_data.update(metrics_dict)
        
        df_new = pd.DataFrame([log_data])
        
        if os.path.exists(registry_path):
            df_old = pd.read_csv(registry_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined.to_csv(registry_path, index=False)
        print(f"[Logger] Experiment registered to {registry_path}")

    def get_save_path(self, filename):
        return os.path.join(self.results_dir, filename)
    
    
def get_system_class(system_name):
    """Dynamically imports the system class (e.g., systems.ogtt_simul.OgttSimul)."""
    module_name = f"systems.{system_name}"
    # Convention: snake_case file -> PascalCase class (e.g., ogtt_simul -> OgttSimul)
    class_name = "".join(word.capitalize() for word in system_name.split('_'))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)