# tests/test_sde_core.py
import sys
import os
import numpy as np
import json
import time

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.ogtt_simul import OgttSimul, OGTTModel, ode_params, sys_params, interpolate_sigma
from utils import euler_maruyama
from config import Config

def test_sde_functionality():
    """Phase 3: Euler-Maruyama Solver 및 Diffusion Term 활성화 테스트"""
    print("\n[Phase 3 Test] SDE Solver Functionality Check...")
    
    # --- 1. 환경 설정 및 초기값 준비 ---
    system = OgttSimul()
    config = Config()
    
    # 테스트에 사용할 고정 파라미터 및 초기 조건
    test_params_list = [1.0, 0.8] # si=1.0, sigma=0.8
    test_params_dict = {'si': 1.0, 'sigma': 0.8}
    test_y0 = system.sample_initial_conditions(test_params_dict) # [G, I, N5, N6]
    
    t_span = system.t_span
    
    # 1분 간격 시뮬레이션 포인트 (t=0부터 t=120까지 121개)
    t_eval_high_res = np.linspace(t_span[0], t_span[1], 121) 
    dt_sim = 1.0
    
    # --- 2. 확산 계수 보간 테스트 (핵심) ---
    print("  -> Checking Sigma Interpolation...")
    try:
        # ogtt_simul.py가 로드한 SIGMA_G_T를 사용한다고 가정
        t_intermediate = 15.0 # t=0과 t=30 사이의 값
        
        # 파일이 생성되어야만 작동하므로, 테스트를 위해 0으로 시작하는 임시 데이터를 로드
        sigma_t_points = np.array([0, 30, 60, 90, 120])
        sigma_g_test = np.array([10.0, 15.0, 12.0, 5.0, 3.0])
        
        sigma_at_15 = interpolate_sigma(t_intermediate, sigma_t_points, sigma_g_test)
        
        # 0과 30 사이의 선형 보간: 10 + (15/30) * (15 - 10) = 12.5
        assert np.isclose(sigma_at_15, 12.5), f"❌ Interpolation failed: Expected 12.5, Got {sigma_at_15}"
        print(f"  ✅ Interpolation successful: sigma(15min)={sigma_at_15:.2f}")

    except Exception as e:
        print(f"  ❌ Interpolation test failed. Ensure 'calibrated_sigmas.json' is created. Error: {e}")
        return

    # --- 3. SDE/ODE 궤적 비교 ---
    num_runs = 50 # SDE 궤적 앙상블 크기
    sde_trajectories = []
    
    # 결정론적 ODE 궤적 (비교 기준)
    ode_model = OGTTModel(ode_params, sys_params, test_params_dict)
    sol_ode = ode_model.simulate(t_span, test_y0, t_eval=t_eval_high_res)
    ode_G = sol_ode.y[0] # Glucose 궤적
    
    # SDE 궤적 생성
    start_time = time.time()

    for i in range(num_runs):
        seed_base = int(time.time() * 1000) # 현재 시간을 가장 큰 정수로
        safe_seed = (seed_base + i) % (2**32) # 32비트 정수 범위 내로 조정

        # euler_maruyama는 (n_vars, n_steps)를 반환
        y_sde = euler_maruyama(system.drift_func, system.diffusion_func, 
                               t_span, test_y0, t_eval_high_res, 
                               test_params_list, seed=safe_seed, 
                               dt_sim=dt_sim,
                               system=system)
        sde_trajectories.append(y_sde[0]) # Glucose 궤적만 저장
        
    end_time=time.time()
    elapsed_time = end_time - start_time
    print(f"  -> SDE Trajectory Generation Time (Total for {num_runs} runs): {elapsed_time:.2f} seconds")
    print(f"  -> Estimated Time per Sample (1min resolution): {elapsed_time/num_runs * 1000:.2f} ms")
    sde_trajectories = np.array(sde_trajectories) # (50, 121)
    
    # --- 4. 확산 활성화 검증 (Variance Check) ---
    # 최종 시점(t=120)에서의 분산을 확인
    final_time_variance = np.var(sde_trajectories[:, -1])
    
    print(f"  -> Final Time Variance (t=120min): {final_time_variance:.4f}")
    
    # 분산이 0이 아니면 (즉, 궤적들이 서로 다르면) 확산이 활성화된 것
    # 허용 오차는 SDE 노이즈가 실제로 작을 수 있으므로 넉넉하게 설정
    if final_time_variance > 1e-4:
        print("  ✅ Diffusion Activation successful (Non-zero Variance).")
    else:
        print("  ❌ Diffusion Activation FAILED (Variance too low/Zero).")
        # 실패 시 SDE와 ODE 궤적이 다른지 확인 (노이즈가 약해도 ODE와는 달라야 함)
        sde_mean_G = np.mean(sde_trajectories, axis=0)
        max_diff = np.max(np.abs(sde_mean_G - ode_G))
        if max_diff < 1e-6:
             print("     Hint: Check if calibrated_sigmas.json has non-zero values.")
             
        assert final_time_variance > 1e-4, "Diffusion term is not active or too weak."


    # --- 5. 최종 검증 ---
    print("✅ SDE Core Unit Test Passed.")


if __name__ == "__main__":
    # noise_calibration.py를 먼저 실행하여 calibrated_sigmas.json을 생성해야 합니다.
    # python noise_calibration.py
    
    # Note: 이 테스트는 calibrated_sigmas.json 파일에 비영(Non-zero) 시그마 값이 
    # 정의되어 있을 때만 'Diffusion Activation'이 성공합니다.
    test_sde_functionality()