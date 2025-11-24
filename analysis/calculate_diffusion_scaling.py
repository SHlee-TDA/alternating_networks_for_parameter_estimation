# analysis/calculate_diffusion_scaling.py
"""
SDE Diffusion Scaling Factor (Lambda) 계산기
============================================

기능:
1. 실제 데이터의 시간별 분산(Variance)을 계산합니다.
2. 현재 Calibration된 sigma가 Random Walk로 누적될 때의 이론적 분산을 계산합니다.
3. 두 분산의 비율을 통해 Scaling Factor (lambda)를 역산합니다.
4. 분산 변화 양상을 시각화하여 'diffusion_scaling_analysis.png'로 저장합니다.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from data_loader import RealOGTTDataLoader
from config import Config

def calculate_scaling():
    print("=== Calculating Diffusion Scaling Factor (Lambda) ===")
    
    # 1. 데이터 로드
    config = Config()
    config.USE_LAGRANGIAN = False
    
    # 경로 설정
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists(): data_path = project_root / 'clean_sumner_n_612.xlsx'
    
    param_path = project_root / 'data' / 'parameters' / 'calibrated_sde_params.json'
    
    # Real Data Load
    loader = RealOGTTDataLoader(str(data_path), config)
    X_obs, Y_hid, _, t_points = loader.load_data()
    
    # (N, T)
    real_G = X_obs[:, :, 0]
    real_I = Y_hid[:, :, 0]
    
    # SDE Param Load
    with open(param_path, 'r') as f:
        calib = json.load(f)
        sigma_G_emp = np.array(calib['sigma_G']) # Stepwise sigmas
        sigma_I_emp = np.array(calib['sigma_I'])
    
    # 2. 분산 계산 (Variance Analysis)
    # A. Real Data Variance (Target)
    var_G_real = np.var(real_G, axis=0) # [V0, V30, V60, V90, V120]
    var_I_real = np.var(real_I, axis=0)
    
    # B. Theoretical SDE Variance (Random Walk Assumption)
    # V(t+1) = V(t) + sigma^2 * dt
    def calc_theoretical_variance(v_init, sigmas, t_points):
        vars_theo = [v_init]
        curr_v = v_init
        for i in range(len(t_points)-1):
            dt = t_points[i+1] - t_points[i]
            # sigmas 리스트는 구간별 값이므로 i번째 사용 (마지막 패딩 제외)
            s = sigmas[i]
            added = (s ** 2) * dt
            curr_v += added
            vars_theo.append(curr_v)
        return np.array(vars_theo)

    var_G_theo = calc_theoretical_variance(var_G_real[0], sigma_G_emp, t_points)
    var_I_theo = calc_theoretical_variance(var_I_real[0], sigma_I_emp, t_points)
    
    # 3. Scaling Factor Lambda 유도
    # Lambda^2 = Mean(V_real) / Mean(V_theo)
    # (V_real이 V_theo보다 훨씬 작다면 Lambda < 1이 됨)
    
    # Glucose Lambda
    ratio_G = np.mean(var_G_real) / np.mean(var_G_theo)
    lambda_G = np.sqrt(ratio_G)
    
    # Insulin Lambda
    ratio_I = np.mean(var_I_real) / np.mean(var_I_theo)
    lambda_I = np.sqrt(ratio_I)
    
    # 보수적인 선택: 둘 중 더 작은 값을 공통 Lambda로 쓰거나, 각각 적용
    # 여기서는 안전하게 둘의 평균을 추천값으로 제시
    lambda_mean = (lambda_G + lambda_I) / 2
    
    print(f"\n[Calculated Results]")
    print(f"  -> Lambda (Glucose based): {lambda_G:.4f}")
    print(f"  -> Lambda (Insulin based): {lambda_I:.4f}")
    print(f"  => Recommended DIFFUSION_SCALE: {lambda_mean:.4f}")

    # 4. 시각화 (Visualization)
    # Lambda 적용 시 예상되는 분산 궤적
    var_G_scaled = calc_theoretical_variance(var_G_real[0], sigma_G_emp * lambda_G, t_points)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Glucose Plot
    ax[0].plot(t_points, var_G_real, 'ko-', label='Real Variance (Target)', linewidth=2)
    ax[0].plot(t_points, var_G_theo, 'r--', label='Unscaled SDE Variance (Over-diffused)')
    ax[0].plot(t_points, var_G_scaled, 'b-o', label=f'Scaled SDE Variance ($\lambda={lambda_G:.2f}$)')
    ax[0].set_title(f"Glucose Variance Evolution\nScaling Factor $\lambda = {lambda_G:.3f}$")
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel("Variance")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Insulin Plot (Insulin Lambda 적용)
    var_I_scaled = calc_theoretical_variance(var_I_real[0], sigma_I_emp * lambda_I, t_points)
    ax[1].plot(t_points, var_I_real, 'ko-', label='Real Variance (Target)', linewidth=2)
    ax[1].plot(t_points, var_I_theo, 'r--', label='Unscaled SDE Variance')
    ax[1].plot(t_points, var_I_scaled, 'g-o', label=f'Scaled SDE Variance ($\lambda={lambda_I:.2f}$)')
    ax[1].set_title(f"Insulin Variance Evolution\nScaling Factor $\lambda = {lambda_I:.3f}$")
    ax[1].set_xlabel("Time (min)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    save_path = current_dir / 'diffusion_scaling_analysis.png'
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[Visualization] Saved variance analysis plot to: {save_path}")

    return lambda_mean

if __name__ == "__main__":
    calculate_scaling()