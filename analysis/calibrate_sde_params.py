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
from data_loader import RealOGTTDataLoader
from config import Config

def remove_outliers(data, lower=1, upper=99):
    """데이터의 상/하위 퍼센타일을 벗어나는 이상치를 제거"""
    if len(data) == 0: return data
    lb = np.percentile(data, lower)
    ub = np.percentile(data, upper)
    return data[(data >= lb) & (data <= ub)]

def plot_variable_analysis(stats_list, var_name, color, save_path):
    """변수별 상세 분석 시각화 함수"""
    N_intervals = len(stats_list)
    fig, axes = plt.subplots(N_intervals, 3, figsize=(15, 3.5 * N_intervals))
    fig.suptitle(f'{var_name} Increment Analysis: Dist, State-Dep, Range', fontsize=16)
    
    for i, stat in enumerate(stats_list):
        # 1. Histogram (Residual Distribution)
        sns.histplot(stat['dR_clean'], kde=True, ax=axes[i, 0], color=color, alpha=0.6)
        axes[i, 0].axvline(stat['mu'], color='red', linestyle='--', label=f"Mean: {stat['mu']:.2f}")
        axes[i, 0].set_title(f"Res Dist ({stat['label']})")
        axes[i, 0].legend()
        
        # 2. Scatter (State Dependency)
        corr = np.corrcoef(stat['current_state'], stat['dR_clean'])[0, 1] if len(stat['dR_clean']) > 1 else 0
        axes[i, 1].scatter(stat['current_state'], stat['dR_clean'], alpha=0.3, s=15, color=color)
        axes[i, 1].set_title(f"State Dep: {var_name}(t) vs dR (Corr={corr:.2f})")
        axes[i, 1].set_xlabel(f"{var_name} at t_start")
        axes[i, 1].set_ylabel("Increment Residual")
        axes[i, 1].axhline(stat['mu'], color='red', linestyle='--')
        
        # 3. Boxplot (Variability)
        sns.boxplot(y=stat['dR_clean'], ax=axes[i, 2], color=color)
        axes[i, 2].set_title("Variability Range")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Saved analysis plot to: {save_path}")


def calibrate_sde():
    print("=== [SDE Calibration] Starting Analysis & Parameter Estimation ===")
    
    # 1. 데이터 로드
    config = Config()
    config.USE_LAGRANGIAN = False 
    
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists():
        data_path = project_root / 'clean_sumner_n_612.xlsx'
    
    loader = RealOGTTDataLoader(str(data_path), config)
    X_obs, Y_hid, P_true, t_points = loader.load_data()
    
    real_G = X_obs[:, :, 0]
    real_I = Y_hid[:, :, 0]
    
    N_samples = len(real_G)
    N_time = len(t_points)

    # 2. 물리적 상한(Bounds) 계산
    max_G = np.nanmax(real_G)
    max_I = np.nanmax(real_I)
    BOUNDS = {
        'G_max': float(max_G * 1.1),
        'I_max': float(max_I * 1.1)
    }
    print(f"  -> Physical Bounds: G_max={BOUNDS['G_max']:.1f}, I_max={BOUNDS['I_max']:.1f}")

    # 3. 결정론적 시뮬레이션
    print(f"  -> Simulating {N_samples} patients for reference trajectories...")
    sim_G = np.zeros_like(real_G)
    sim_I = np.zeros_like(real_I)
    
    for i in range(N_samples):
        theta = {'si': P_true[i, 0], 'sigma': P_true[i, 1]}
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

    # 4. 구간별 파라미터 추정
    sigma_G_list, mu_G_list = [], []
    sigma_I_list, mu_I_list = [], []
    
    # 시각화를 위한 데이터 수집
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
        # 이상치 제거를 위한 마스크 생성 (Current State와 dR 매칭을 위해)
        lb_G, ub_G = np.percentile(dR_G, 1), np.percentile(dR_G, 99)
        mask_G = (dR_G >= lb_G) & (dR_G <= ub_G)
        
        dR_G_clean = dR_G[mask_G]
        curr_G_clean = real_G[mask_G, i] # 산점도용
        
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

    # 5. 시각화 실행 (Glucose & Insulin 각각 저장)
    plot_variable_analysis(stats_G, "Glucose", "blue", current_dir / 'sde_calibration_glucose.png')
    plot_variable_analysis(stats_I, "Insulin", "green", current_dir / 'sde_calibration_insulin.png')

    # 6. 결과 저장 (JSON)
    t_save = t_points.tolist()
    
    sigma_G_final = sigma_G_list + [sigma_G_list[-1]]
    sigma_I_final = sigma_I_list + [sigma_I_list[-1]]
    mu_G_final = mu_G_list + [mu_G_list[-1]]
    mu_I_final = mu_I_list + [mu_I_list[-1]]
    
    calibrated_data = {
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
        
    # 요약 출력
    df_summary = pd.DataFrame(summary_data)
    print("\n=== SDE Parameter Summary ===")
    print(df_summary)
    print(f"\n[Done] Parameters saved to: {save_path}")

if __name__ == "__main__":
    calibrate_sde()