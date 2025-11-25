# analysis/calibrate_sde_params.py
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 상위 디렉토리(프로젝트 루트)의 모듈을 import하기 위한 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# 프로젝트 모듈 import
from systems.ogtt_simul import OGTTModel, ode_params, sys_params
from data_loader import RealOGTTDataLoader
from config import Config

def remove_outliers(data, lower=1, upper=99):
    lb = np.percentile(data, lower)
    ub = np.percentile(data, upper)
    return data[(data >= lb) & (data <= ub)]

def calibrate_sde():
    print("=== SDE Parameter Calibration (Drift Bias & Diffusion) ====")

    # 1. Data Load
    config = Config()
    config.USE_LAGRANGIAN = False

    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists():
        data_path = project_root / 'clean_sumner_n_612.xlsx'

    loader = RealOGTTDataLoader(str(data_path), config)
    X_obs, Y_hid, P_true, t_points = loader.load_data() # X_obs: (N, 5, 1))
