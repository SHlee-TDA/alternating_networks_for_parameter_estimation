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
    # 파일 경로는 환경에 맞게 조정해주세요
    data_path = 'data/clean_sumner_n_612.xlsx' 
    
    print("=== Starting Noise Calibration ===")
    loader = RealOGTTDataLoader(data_path, config)
    
    # 수정된 리턴 구조: (Observed=Glucose, Hidden=Insulin, Params, Time)
    # X_obs shape: (N, 5, 1), Y_hid shape: (N, 5, 1)
    X_obs_glucose, Y_hid_insulin, P_true, t_points = loader.load_data()
    
    residuals_G = []
    residuals_I = []
    
    print(f"Simulating {len(X_obs_glucose)} patients to calculate residuals...")
    
    # 2. 시뮬레이션 및 잔차 계산
    for i in range(len(X_obs_glucose)):
        # 파라미터 추출
        si_val = P_true[i, 0]
        sigma_val = P_true[i, 1]
        theta = {'si': si_val, 'sigma': sigma_val}
        
        # 모델 초기화
        model = OGTTModel(ode_params, sys_params, theta)
        
        # 초기값 설정: 데이터의 0분 시점 값 사용
        # shape이 (N, T, 1)이므로 [i, 0, 0]으로 접근
        g0 = X_obs_glucose[i, 0, 0] 
        i0 = Y_hid_insulin[i, 0, 0]
        
        # Steady State 계산으로 나머지 초기값 추정
        n5_0, n6_0 = model.find_steady_state_N(g0)
        y0 = [g0, i0, n5_0, n6_0]
        
        # 결정론적 시뮬레이션 (t=15 제외된 t_points 사용)
        sol = model.simulate(t_span=[0, 120], initial_conditions=y0, t_eval=t_points)
        
        if sol.success:
            # 시뮬레이션 결과: (4, 5) -> 전치하면 (5, 4) -> G, I 추출
            # sol.y[0] = G, sol.y[1] = I
            sim_G = sol.y[0] 
            sim_I = sol.y[1]
            
            # 실제 데이터 (N, 5, 1) -> (5,)
            real_G = X_obs_glucose[i].flatten()
            real_I = Y_hid_insulin[i].flatten()
            
            # 잔차 = 실제 - 시뮬레이션
            res_G = real_G - sim_G
            res_I = real_I - sim_I
            
            residuals_G.append(res_G)
            residuals_I.append(res_I)
        else:
            print(f"Simulation failed for patient {i}")

    # 3. 통계 분석
    residuals_G = np.concatenate(residuals_G)
    residuals_I = np.concatenate(residuals_I)
    
    sigma_emp_G = np.std(residuals_G)
    sigma_emp_I = np.std(residuals_I)
    
    print("\n" + "="*40)
    print(f"Calibrated Noise Levels (Sigma Empirical)")
    print("="*40)
    print(f"Glucose Noise (std): {sigma_emp_G:.4f}")
    print(f"Insulin Noise (std): {sigma_emp_I:.4f}")
    print("="*40)
    
    # 4. 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(residuals_G, bins=50, color='blue', alpha=0.7, density=True)
    axes[0].set_title(f'Glucose Residuals\nstd={sigma_emp_G:.2f}')
    axes[0].set_xlabel('Residual (mg/dL)')
    
    axes[1].hist(residuals_I, bins=50, color='green', alpha=0.7, density=True)
    axes[1].set_title(f'Insulin Residuals\nstd={sigma_emp_I:.2f}')
    axes[1].set_xlabel('Residual (uU/ml)')
    
    plt.tight_layout()
    plt.savefig('noise_calibration_result.png')
    print("Saved calibration plot to 'noise_calibration_result.png'")

    return sigma_emp_G, sigma_emp_I

if __name__ == "__main__":
    calibrate_noise()