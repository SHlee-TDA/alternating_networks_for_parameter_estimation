import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import copy
from scipy.integrate import solve_ivp

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import Config
from data_loader import RealOGTTDataLoader
from systems.ogtt_simul import OgttSimul, OGTTModel, ode_params, sys_params
from utils import euler_maruyama

# 시각화 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans' 

def visualize_sde_effect(sample_idx=0):
    print(f"=== Visualizing SDE Effect for Sample {sample_idx} ===")
    
    # 1. 설정 및 데이터 로드
    config = Config()
    config.USE_LAGRANGIAN = False # Raw Data만 필요
    
    # 데이터 경로 (환경에 맞춰 자동 조정)
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists():
        data_path = project_root / 'clean_sumner_n_612.xlsx'
    
    loader = RealOGTTDataLoader(str(data_path), config)
    X_obs, Y_hid, P_true, t_points = loader.load_data()
    
    # 선택한 환자의 데이터 추출
    real_G = X_obs[sample_idx, :, 0]
    real_I = Y_hid[sample_idx, :, 0]
    params = P_true[sample_idx] # [si, sigma]
    
    # 2. 시뮬레이션 준비 (High Resolution)
    t_eval_fine = np.linspace(0, 120, 121) # 0, 1, ..., 120 (1분 간격)
    
    system = OgttSimul()
    # Config의 Diffusion Scale 적용
    system.diffusion_scale = getattr(config, 'DIFFUSION_SCALE', 1.27)
    #print(f"  -> Applied Diffusion Scale: {system.diffusion_scale}")
    
    # 초기값 설정 (Steady State 계산)
    g0, i0 = real_G[0], real_I[0]
    
    # OGTTModel을 이용해 N5, N6 평형점 찾기
    temp_model = OGTTModel(ode_params, sys_params, {'si': params[0], 'sigma': params[1]})
    n5, n6 = temp_model.find_steady_state_N(g0)
    y0 = [g0, i0, n5, n6]
    
    # 3. ODE Simulation (Deterministic)
    print("  -> Running ODE Simulation...")
    sol_ode = solve_ivp(
            fun=lambda t, y: system.ode_func(t, y, params),
            t_span=system.t_span,
            y0=y0,
            t_eval=t_eval_fine
        )
    ode_G = sol_ode.y[0]
    ode_I = sol_ode.y[1]
    
    # 4. SDE Simulation (Stochastic Ensemble)
    n_sde_samples = 30
    print(f"  -> Running {n_sde_samples} SDE Simulations...")
    
    sde_G_list = []
    sde_I_list = []
    
    for i in range(n_sde_samples):
        # Seed를 다르게 하여 다양성 생성
        seed = 42 + i 
        y_sde = euler_maruyama(
            system.drift_func,
            system.diffusion_func,
            [0, 120],
            y0,
            t_eval_fine,
            params,
            seed=seed,
            dt_sim=0.01, # 정밀한 계산
            system=system
        )
        sde_G_list.append(y_sde[0])
        sde_I_list.append(y_sde[1])
        
    sde_G_arr = np.array(sde_G_list)
    sde_I_arr = np.array(sde_I_list)
    
    # 통계량 계산 (Mean, 95% CI)
    sde_G_mean = np.mean(sde_G_arr, axis=0)
    sde_G_std = np.std(sde_G_arr, axis=0)
    sde_G_upper = sde_G_mean + 1.96 * sde_G_std
    sde_G_lower = sde_G_mean - 1.96 * sde_G_std
    
    sde_I_mean = np.mean(sde_I_arr, axis=0)
    sde_I_std = np.std(sde_I_arr, axis=0)
    sde_I_upper = sde_I_mean + 1.96 * sde_I_std
    sde_I_lower = sde_I_mean - 1.96 * sde_I_std
    
    # 5. 시각화 (Visualization)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Glucose Plot ---
    ax = axes[0]
    # SDE Individual Trajectories
    for i in range(n_sde_samples):
        ax.plot(t_eval_fine, sde_G_list[i], color='royalblue', alpha=0.15, linewidth=1)
    
    # SDE Mean & CI
    ax.plot(t_eval_fine, sde_G_mean, color='royalblue', linewidth=2, label='SDE Mean', linestyle='-')
    ax.fill_between(t_eval_fine, sde_G_lower, sde_G_upper, color='royalblue', alpha=0.2, label='SDE 95% CI')
    
    # ODE
    ax.plot(t_eval_fine, ode_G, color='crimson', linewidth=2.5, label='ODE (Deterministic)', linestyle='--')
    
    # Real Data (Piecewise Linear)
    ax.plot(t_points, real_G, 'ko-', linewidth=2, markersize=8, label='Real Data', zorder=10)
    
    ax.set_title("Glucose Dynamics (Sim vs Real)", fontsize=16)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Insulin Plot ---
    ax = axes[1]
    # SDE Individual
    for i in range(n_sde_samples):
        ax.plot(t_eval_fine, sde_I_list[i], color='green', alpha=0.15, linewidth=1)
        
    # SDE Mean & CI
    ax.plot(t_eval_fine, sde_I_mean, color='green', linewidth=2, label='SDE Mean', linestyle='-')
    ax.fill_between(t_eval_fine, sde_I_lower, sde_I_upper, color='green', alpha=0.2, label='SDE 95% CI')
    
    # ODE
    ax.plot(t_eval_fine, ode_I, color='darkorange', linewidth=2.5, label='ODE (Deterministic)', linestyle='--')
    
    # Real Data
    ax.plot(t_points, real_I, 'ko-', linewidth=2, markersize=8, label='Real Data', zorder=10)
    
    ax.set_title("Insulin Dynamics (Sim vs Real)", fontsize=16)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Insulin (uU/ml)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"SDE Data Augmentation (Patient #{sample_idx})", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = current_dir / 'sde_effect_visualization.png'
    plt.savefig(save_path)
    print(f"\n[Done] Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # 원하는 샘플 인덱스를 변경하며 테스트 가능 (예: 0, 10, 42 등)
    visualize_sde_effect(sample_idx=0)