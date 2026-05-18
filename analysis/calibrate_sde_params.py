# analysis/calibrate_sde_params.py
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from systems.ogtt_simul import OGTTModel, ode_params, sys_params
from src.data_loader import RealOGTTDataLoader
from config import Config

def plot_variable_analysis(stats_list, var_name, color, save_path):
    """변수별 상세 분석 시각화 함수 (기존과 동일)"""
    N_intervals = len(stats_list)
    fig, axes = plt.subplots(N_intervals, 3, figsize=(15, 3.5 * N_intervals))
    fig.suptitle(f'{var_name} Increment Analysis: Dist, State-Dep, Range', fontsize=16)
    
    for i, stat in enumerate(stats_list):
        # 1. Histogram
        sns.histplot(stat['dR_clean'], kde=True, ax=axes[i, 0], color=color, alpha=0.6)
        axes[i, 0].axvline(stat['mu'], color='red', linestyle='--', label=f"Mean: {stat['mu']:.2f}")
        axes[i, 0].set_title(f"Res Dist ({stat['label']})")
        axes[i, 0].legend()
        
        # 2. Scatter
        curr_state = stat['current_state']
        dr_vals = stat['dR_clean']
        if len(curr_state) != len(dr_vals):
            # 길이 불일치 안전장치 (이상치 제거 과정에서 길이 달라질 수 있음)
            min_len = min(len(curr_state), len(dr_vals))
            curr_state = curr_state[:min_len]
            dr_vals = dr_vals[:min_len]

        corr = np.corrcoef(curr_state, dr_vals)[0, 1] if len(dr_vals) > 1 else 0
        axes[i, 1].scatter(curr_state, dr_vals, alpha=0.3, s=15, color=color)
        axes[i, 1].set_title(f"State Dep: {var_name}(t) vs dR (Corr={corr:.2f})")
        axes[i, 1].set_xlabel(f"{var_name} at t_start")
        axes[i, 1].set_ylabel("Increment Residual")
        axes[i, 1].axhline(stat['mu'], color='red', linestyle='--')
        
        # 3. Boxplot
        sns.boxplot(y=dr_vals, ax=axes[i, 2], color=color)
        axes[i, 2].set_title("Variability Range")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Saved analysis plot to: {save_path}")

def calibrate_sde():
    print("=== [SDE Calibration] Starting Analysis (Train Set Only) ===")
    
    # 1. 데이터 로드
    config = Config()
    config.USE_LAGRANGIAN = False 
    
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists():
        data_path = project_root / 'clean_sumner_n_612.xlsx'
    
    loader = RealOGTTDataLoader(str(data_path), config)
    X_obs, Y_hid, P_true, t_points = loader.load_data()
    
    # 2. Train Split 적용
    split_file = project_root / 'data' / 'data_split_indices.json'
    if split_file.exists():
        with open(split_file, 'r') as f:
            split_data = json.load(f)
            train_idx = split_data['train_indices']
        
        print(f"  -> Applying Data Split: Using {len(train_idx)} Train samples.")
        
        # Numpy Indexing으로 Train 데이터만 추출
        real_G = X_obs[train_idx, :, 0]
        real_I = Y_hid[train_idx, :, 0]
        P_subset = P_true[train_idx, :] # 파라미터도 train만
        
    else:
        print("  -> [WARNING] Split file NOT found! Using ALL data (Data Leakage Risk).")
        print("  -> Please run 'utils/create_split.py' first.")
        real_G = X_obs[:, :, 0]
        real_I = Y_hid[:, :, 0]
        P_subset = P_true

    N_samples = len(real_G)
    N_time = len(t_points)

    # 3. 물리적 상한(Bounds) - 안전을 위해 전체 데이터 기준으로 해도 무방하나, 엄밀하게는 Train 기준
    max_G = np.nanmax(real_G)
    max_I = np.nanmax(real_I)
    BOUNDS = {
        'G_max': float(max_G * 1.2), # 20% 마진
        'I_max': float(max_I * 1.2)
    }
    
    # 4. 결정론적 시뮬레이션 (Reference Trajectory)
    print(f"  -> Simulating {N_samples} train patients for reference trajectories...")
    sim_G = np.zeros_like(real_G)
    sim_I = np.zeros_like(real_I)
    
    for i in range(N_samples):
        # Train Set의 파라미터 사용
        theta = {'si': P_subset[i, 0], 'sigma': P_subset[i, 1]}
        model = OGTTModel(ode_params, sys_params, theta)
        
        g0, i0 = real_G[i, 0], real_I[i, 0]
        n5, n6 = model.find_steady_state_N(g0)
        y0 = [g0, i0, n5, n6]
        
        sol = model.simulate([0, 120], y0, t_eval=t_points)
        if sol.success:
            sim_G[i] = sol.y[0]
            sim_I[i] = sol.y[1]
        else:
            sim_G[i] = real_G[i] 

    # 5. 구간별 파라미터 추정
    sigma_G_list, mu_G_list = [], []
    sigma_I_list, mu_I_list = [], []
    
    stats_G = []
    stats_I = []
    summary_data = []

    for i in range(N_time - 1):
        t_start, t_end = t_points[i], t_points[i+1]
        dt = t_end - t_start
        label = f"{t_start}-{t_end} min"
        
        # 증분 및 잔차 계산
        dG_real = real_G[:, i+1] - real_G[:, i]
        dG_sim = sim_G[:, i+1] - sim_G[:, i]
        dR_G = dG_real - dG_sim
        
        dI_real = real_I[:, i+1] - real_I[:, i]
        dI_sim = sim_I[:, i+1] - sim_I[:, i]
        dR_I = dI_real - dI_sim
        
        # --- Glucose Analysis ---
        lb_G, ub_G = np.percentile(dR_G, 1), np.percentile(dR_G, 99)
        mask_G = (dR_G >= lb_G) & (dR_G <= ub_G)
        
        dR_G_clean = dR_G[mask_G]
        curr_G_clean = real_G[mask_G, i] 
        
        mu_G = np.mean(dR_G_clean)
        std_G = np.std(dR_G_clean)
        
        mu_bias_G = mu_G / dt
        sigma_diff_G = std_G / np.sqrt(dt)
        
        mu_G_list.append(mu_bias_G)
        sigma_G_list.append(sigma_diff_G)
        
        stats_G.append({
            'label': label, 'dR_clean': dR_G_clean, 'current_state': curr_G_clean,
            'mu': mu_G, 'sigma': std_G
        })

        # --- Insulin Analysis ---
        lb_I, ub_I = np.percentile(dR_I, 1), np.percentile(dR_I, 99)
        mask_I = (dR_I >= lb_I) & (dR_I <= ub_I)
        
        dR_I_clean = dR_I[mask_I]
        curr_I_clean = real_I[mask_I, i]
        
        mu_I = np.mean(dR_I_clean)
        std_I = np.std(dR_I_clean)
        
        mu_bias_I = mu_I / dt
        sigma_diff_I = std_I / np.sqrt(dt)
        
        mu_I_list.append(mu_bias_I)
        sigma_I_list.append(sigma_diff_I)
        
        stats_I.append({
            'label': label, 'dR_clean': dR_I_clean, 'current_state': curr_I_clean,
            'mu': mu_I, 'sigma': std_I
        })
        
        summary_data.append({
            'Interval': label,
            'G_Bias': mu_bias_G, 'G_Diff': sigma_diff_G,
            'I_Bias': mu_bias_I, 'I_Diff': sigma_diff_I
        })

    # 6. 시각화 및 저장
    save_dir_vis = project_root / 'analysis' / 'calibration_plots'
    os.makedirs(save_dir_vis, exist_ok=True)
    
    plot_variable_analysis(stats_G, "Glucose", "blue", save_dir_vis / 'sde_calibration_glucose.png')
    plot_variable_analysis(stats_I, "Insulin", "green", save_dir_vis / 'sde_calibration_insulin.png')

    # 결과 JSON 저장
    t_save = t_points.tolist()
    
    # 마지막 구간 값 복제하여 길이 맞춤
    sigma_G_final = sigma_G_list + [sigma_G_list[-1]]
    sigma_I_final = sigma_I_list + [sigma_I_list[-1]]
    mu_G_final = mu_G_list + [mu_G_list[-1]]
    mu_I_final = mu_I_list + [mu_I_list[-1]]
    
    calibrated_data = {
        'meta': {'source': 'train_set_only', 'n_samples': N_samples},
        't_points': t_save,
        'sigma_G': sigma_G_final,
        'sigma_I': sigma_I_final,
        'mu_G': mu_G_final,
        'mu_I': mu_I_final,
        'bounds': BOUNDS
    }
    
    save_dir = project_root / 'data' / 'parameters'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / 'calibrated_sde_params.json'
    
    with open(save_path, 'w') as f:
        json.dump(calibrated_data, f, indent=4)
        
    df_summary = pd.DataFrame(summary_data)
    print("\n=== SDE Parameter Summary (from Train Set) ===")
    print(df_summary)
    print(f"\n[Done] Parameters saved to: {save_path}")

if __name__ == "__main__":
    calibrate_sde()