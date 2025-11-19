# noise_calibration.py
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# 기존 시스템 정의 모듈 활용
from systems.ogtt_simul import OGTTModel, ode_params, sys_params
from data_loader import RealOGTTDataLoader
from config import Config

def calibrate_noise():
    # 1. 데이터 로드
    config = Config()
    data_path = 'data/clean_sumner_n_612.xlsx' # 실제 데이터 파일 경로
    
    print("=== Starting Time-Dependent Noise Calibration ===")
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
    sigma_emp_G_t = np.array([np.std(res_list) for res_list in residuals_G_by_t])
    sigma_emp_I_t = np.array([np.std(res_list) for res_list in residuals_I_by_t])
    
    # 결과를 SDE Solver가 사용할 수 있도록 JSON 파일로 저장 (임시)
    calibrated_sigmas = {
        't_points': t_points.tolist(),
        'sigma_G': sigma_emp_G_t.tolist(),
        'sigma_I': sigma_emp_I_t.tolist(),
        'bounds': {'G_max': UPPER_BOUND_G, 'I_max': UPPER_BOUND_I}
    }
    with open('calibrated_sigmas.json', 'w') as f:
        json.dump(calibrated_sigmas, f, indent=4)
        
    print("\n" + "="*40)
    print("Calibrated Time-Dependent Noise Levels (Sigma Empirical)")
    print(f"Time Points (min): {t_points}")
    print(f"Glucose Sigma: {sigma_emp_G_t}")
    print(f"Insulin Sigma: {sigma_emp_I_t}")
    print("Saved calibrated_sigmas.json")
    print("="*40)
    
    # 4. 시각화 (각 시간 스텝별 분포 확인)
    fig, axes = plt.subplots(N_T, 2, figsize=(10, 2 * N_T))
    fig.suptitle('Residual Distribution per Time Point', fontsize=14)
    
    for t_idx in range(N_T):
        t = t_points[t_idx]
        
        # Glucose
        axes[t_idx, 0].hist(residuals_G_by_t[t_idx], bins=20, color='blue', alpha=0.7, density=True)
        axes[t_idx, 0].set_title(f'G Residuals (t={t}min), $\sigma$={sigma_emp_G_t[t_idx]:.2f}', fontsize=8)
        
        # Insulin
        axes[t_idx, 1].hist(residuals_I_by_t[t_idx], bins=20, color='green', alpha=0.7, density=True)
        axes[t_idx, 1].set_title(f'I Residuals (t={t}min), $\sigma$={sigma_emp_I_t[t_idx]:.2f}', fontsize=8)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('noise_calibration_result.png')
    print("Saved calibration plot to 'noise_calibration_result.png'")

    return sigma_emp_G_t, sigma_emp_I_t, t_points

if __name__ == "__main__":
    calibrate_noise()