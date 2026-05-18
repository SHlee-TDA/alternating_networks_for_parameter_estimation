# analyze_model_bias.py
"""
결정론적 모델 편향(Bias) 분석 스크립트 (Legacy: noise_calibration.py)
=================================================================

이 스크립트는 시뮬레이션 값과 실제 값의 단순 차이(State Residual)를 분석합니다.
결과: 잔차의 평균이 0이 아님을 보여줌으로써, Drift Correction의 필요성을 입증하는 근거 자료로 사용됩니다.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from pathlib import Path

# 상위 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from systems.ogtt_simul import OGTTModel, ode_params, sys_params
from src.data_loader import RealOGTTDataLoader
from config import Config
# 기존 시스템 정의 모듈 활용
from systems.ogtt_simul import OGTTModel, ode_params, sys_params
from src.data_loader import RealOGTTDataLoader
from config import Config


def remove_outliers(data, lower_percentile=1, upper_percentile=99):
    """
    데이터에서 상/하위 퍼센타일을 벗어나는 이상치를 제거합니다.
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    # 범위 내의 데이터만 필터링
    return data[(data >= lower_bound) & (data <= upper_bound)]

def analyze_bias():
    # 1. 데이터 로드
    config = Config()
    data_path = 'data/clean_sumner_n_612.xlsx' # 실제 데이터 파일 경로

    print("=== Analyzing Deterministic Model Bias (State Residuals) ===")
    loader = RealOGTTDataLoader(data_path, config)
    X_obs_glucose, Y_hid_insulin, P_true, t_points = loader.load_data()
    
    N_T = len(t_points) # 5개 (0, 30, 60, 90, 120)
    
    # Compute physical upper bound from data
    max_G_observed = np.nanmax(X_obs_glucose)
    max_I_observed = np.nanmax(Y_hid_insulin)

    # Add 10% margin
    # 10% 여유분(Margin) 추가
    UPPER_BOUND_G = float(max_G_observed * 1.1)
    UPPER_BOUND_I = float(max_I_observed * 1.1)
    
    print(f"\n[Physical Constraints Analysis]")
    print(f"  Max Observed Glucose: {max_G_observed:.2f} -> Upper Bound: {UPPER_BOUND_G:.2f}")
    print(f"  Max Observed Insulin: {max_I_observed:.2f} -> Upper Bound: {UPPER_BOUND_I:.2f}")

    # 시간 인덱스별 잔차를 저장할 리스트 (Residuals by Time Index)
    residuals_G_by_t = [[] for _ in range(N_T)]
    residuals_I_by_t = [[] for _ in range(N_T)]
    
    print(f"Simulating {len(X_obs_glucose)} patients to calculate residuals...")
    
    # 2. 시뮬레이션 및 잔차 계산 (시간별 분리)
    for i in range(len(X_obs_glucose)):
        si_val = P_true[i, 0]
        sigma_val = P_true[i, 1]
        theta = {'si': si_val, 'sigma': sigma_val}
        
        model = OGTTModel(ode_params, sys_params, theta)
        
        g0 = X_obs_glucose[i, 0, 0] 
        i0 = Y_hid_insulin[i, 0, 0]
        
        n5_0, n6_0 = model.find_steady_state_N(g0)
        y0 = [g0, i0, n5_0, n6_0]
        
        sol = model.simulate(t_span=[0, 120], initial_conditions=y0, t_eval=t_points)
        
        if sol.success:
            sim_G = sol.y[0] 
            sim_I = sol.y[1]
            real_G = X_obs_glucose[i].flatten()
            real_I = Y_hid_insulin[i].flatten()
            
            # 각 시간 인덱스별로 잔차를 분리하여 저장
            for t_idx in range(N_T):
                residuals_G_by_t[t_idx].append(real_G[t_idx] - sim_G[t_idx])
                residuals_I_by_t[t_idx].append(real_I[t_idx] - sim_I[t_idx])
        # else:
        #     print(f"Simulation failed for patient {i}") # 테스트 통과했으므로 생략

    # 3. 통계 분석 및 결과 출력
    # 각 시간 인덱스별 표준편차(sigma) 계산
    mu_emp_G_t, sigma_emp_G_t = [], []
    mu_emp_I_t, sigma_emp_I_t = [], []

    fig, axes = plt.subplots(N_T, 2, figsize=(10, 3 * N_T))
    fig.suptitle('Robust Residual Analysis (1-99%ile)', fontsize=14)

    for t_idx in range(N_T):
        # 이상치 제거 (하위 10%, 상위 10% 제거)
        clean_res_G = remove_outliers(np.array(residuals_G_by_t[t_idx]), 1, 99)
        clean_res_I = remove_outliers(np.array(residuals_I_by_t[t_idx]), 1, 99)
        
        # 통계량 계산
        mu_G, std_G = np.mean(clean_res_G), np.std(clean_res_G)
        mu_I, std_I = np.mean(clean_res_I), np.std(clean_res_I)
        
        mu_emp_G_t.append(mu_G)
        sigma_emp_G_t.append(std_G)
        mu_emp_I_t.append(mu_I)
        sigma_emp_I_t.append(std_I)
        
        # 시각화
        t = t_points[t_idx]
        axes[t_idx, 0].hist(clean_res_G, bins=30, color='blue', alpha=0.7, density=True)
        axes[t_idx, 0].set_title(f'G Res (t={t}): $\mu$={mu_G:.1f}, $\sigma$={std_G:.1f}')
        axes[t_idx, 0].axvline(mu_G, color='red', linestyle='--', label='Mean')
        
        axes[t_idx, 1].hist(clean_res_I, bins=30, color='green', alpha=0.7, density=True)
        axes[t_idx, 1].set_title(f'I Res (t={t}): $\mu$={mu_I:.1f}, $\sigma$={std_I:.1f}')
        axes[t_idx, 1].axvline(mu_I, color='red', linestyle='--', label='Mean')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(current_dir / 'model_bias_analysis_result.png')

    calibrated_data = {
        't_points': t_points.tolist(),
        # Std
        'std_G': sigma_emp_G_t,
        'std_I': sigma_emp_I_t,
        # Mu
        'mu_G': mu_emp_G_t,
        'mu_I': mu_emp_I_t,
        'bounds': {
            'G_max': UPPER_BOUND_G,
            'I_max': UPPER_BOUND_I
        }
    }
    save_dir = project_root / 'data' / 'parameters'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / 'model_bias_statistics.json'
    
    with open(save_path, 'w') as f:
        json.dump(calibrated_data, f, indent=4)
        
    print("\n[Result Summary]")
    print(f"Glucose Sigma(t): {[f'{v:.2f}' for v in sigma_emp_G_t]}")
    print(f"Glucose Mean(t) : {[f'{v:.2f}' for v in mu_emp_G_t]}")
    print(f"Insulin Sigma(t): {[f'{v:.2f}' for v in sigma_emp_I_t]}")
    print(f"Insulin Mean(t) : {[f'{v:.2f}' for v in mu_emp_I_t]}")
    print("Calibrated data saved to 'model_bias_statistics.json'")

    return calibrated_data

if __name__ == "__main__":
    analyze_bias()