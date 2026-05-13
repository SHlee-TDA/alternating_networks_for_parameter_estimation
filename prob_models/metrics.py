"""
This script contains evaluation metrics for probabilistic models
"""

import numpy as np
import torch

def negative_log_likelihood(theta_true, theta_pred):
    