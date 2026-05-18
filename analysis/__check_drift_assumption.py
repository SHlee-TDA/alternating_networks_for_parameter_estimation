### EXPIRED ####
# check_drift_assumption.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# 기존 모듈 활용
from systems.ogtt_simul import OGTTModel, ode_params, sys_params
from src.data_loader import RealOGTTDataLoader
from config import Config

def remove_outliers(data, lower=1, upper=99):
    lb = np.percentile(data, lower)
    ub = np.percentile(data, upper)
    return data[(data >= lb) & (data <= ub)]

def analyze_drift_statistics():
    print("=== Analyzing Increment Residuals for SDE Strategy ===")
    
    # 1. 데이터 로드
    config = Config()
    data_path = 'data/clean_sumner_n_612.xlsx' 
    if not os.path.exists(data_path):
         data_path = 'clean_sumner_n_612.xlsx'
         
    loader = RealOGTTDataLoader(data_path, config)
    
    # X_obs: (N, T, D) 형태. D=2 (Value, Derivative)일 수 있음.
    X_obs_glucose, Y_hid_insulin, P_true, t_points = loader.load_data()
    
    N_samples = len(X_obs_glucose)
    N_time = len(t_points)
    
    # [수정] 0번 채널(농도)만 추출
    # reshape은 불필요 (이미 N, T 형태임)
    real_G = X_obs_glucose[:, :, 0]
    real_I = Y_hid_insulin[:, :, 0]
    
    print(f"Loaded Data Shape: {real_G.shape}") # (N, T) 확인용

    # 2. 시뮬레이션 (결정론적 궤적)
    print(f"Simulating {N_samples} patients...")
    sim_G = np.zeros((N_samples, N_time))
    sim_I = np.zeros((N_samples, N_time))
    
    for i in range(N_samples):
        theta = {'si': P_true[i, 0], 'sigma': P_true[i, 1]}
        model = OGTTModel(ode_params, sys_params, theta)
        
        # 초기값
        g0, i0 = real_G[i, 0], real_I[i, 0]
        n5, n6 = model.find_steady_state_N(g0)
        y0 = [g0, i0, n5, n6]
        
        sol = model.simulate([0, 120], y0, t_eval=t_points)
        if sol.success:
            sim_G[i] = sol.y[0]
            sim_I[i] = sol.y[1]
        else:
            sim_G[i] = real_G[i] # Failback
            sim_I[i] = real_I[i]

    # 3. 증분 잔차 (Incremental Residual) 분석
    intervals = [(t_points[i], t_points[i+1]) for i in range(N_time-1)]
    
    fig, axes = plt.subplots(len(intervals), 3, figsize=(18, 4 * len(intervals)))
    fig.suptitle("Increment Residual Analysis: Distributions & State Dependency", fontsize=16)
    
    stats_summary = []

    for idx, (t_start, t_end) in enumerate(intervals):
        dt = t_end - t_start
        
        # 증분 계산
        dG_real = real_G[:, idx+1] - real_G[:, idx]
        dG_sim = sim_G[:, idx+1] - sim_G[:, idx]
        dR_G = dG_real - dG_sim # 증분 잔차
        
        # 이상치 제거 (분석용)
        mask = (dR_G >= np.percentile(dR_G, 1)) & (dR_G <= np.percentile(dR_G, 99))
        dR_G_clean = dR_G[mask]
        current_G_clean = real_G[mask, idx] # 시작 시점의 Glucose 값
        
        # 통계량
        mu = np.mean(dR_G_clean)
        std = np.std(dR_G_clean)
        
        # SDE 파라미터 환산
        drift_bias = mu / dt
        diffusion = std / np.sqrt(dt)
        
        stats_summary.append({
            'Interval': f"{t_start}-{t_end}",
            'Mean_dR': mu, 'Std_dR': std,
            'Drift_Bias': drift_bias, 'Diffusion_Coeff': diffusion
        })
        
        # --- 시각화 ---
        # 1. 히스토그램 (분포 확인)
        sns.histplot(dR_G_clean, kde=True, ax=axes[idx, 0], color='blue')
        axes[idx, 0].axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.2f}')
        axes[idx, 0].set_title(f"dG Residual Dist ({t_start}-{t_end} min)")
        axes[idx, 0].legend()
        
        # 2. State Dependency (Scatter: Current G vs dR)
        corr = np.corrcoef(current_G_clean, dR_G_clean)[0, 1]
        axes[idx, 1].scatter(current_G_clean, dR_G_clean, alpha=0.3, s=10)
        axes[idx, 1].set_title(f"State Dependency: G({t_start}) vs dR (Corr={corr:.2f})")
        axes[idx, 1].set_xlabel(f"Glucose at t={t_start}")
        axes[idx, 1].set_ylabel("Increment Residual (dG_real - dG_sim)")
        axes[idx, 1].axhline(mu, color='red', linestyle='--')
        
        # 3. Boxplot (변동성 범위 확인)
        sns.boxplot(y=dR_G_clean, ax=axes[idx, 2], color='lightblue')
        axes[idx, 2].set_title(f"Variability Range ({t_start}-{t_end})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("drift_assumption_check.png")
    
    # 결과 출력
    df_stats = pd.DataFrame(stats_summary)
    print("\n=== Analysis Result: Glucose Increment Residuals ===")
    print(df_stats)
    print("\nSaved plot to 'drift_assumption_check.png'")

if __name__ == "__main__":
    analyze_drift_statistics()