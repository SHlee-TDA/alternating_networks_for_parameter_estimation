# analysis/debug_single_step.py
import sys
import os
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from systems.ogtt_simul import OgttSimul, OGTTModel, ode_params, sys_params
from config import Config

def debug_sde_step_by_step():
    print("=== SDE Simulation Step-by-Step Debugging ===")
    
    # 1. 설정 및 시스템 초기화
    config = Config()
    # 테스트를 위해 Diffusion Scale을 명시적으로 설정 (0.1)
    # config.py에 설정된 값이 0.1이라고 가정하지만, 여기서 강제 주입하여 확인
    target_scale = 0.1
    
    system = OgttSimul()
    system.diffusion_scale = target_scale
    print(f"[Setup] Diffusion Scale set to: {system.diffusion_scale}")
    
    # 2. 초기값 설정 (Random Sampling 대신 고정값 사용)
    # G0=100, I0=10 (일반적인 공복 수치)
    g0, i0 = 100.0, 10.0
    
    # Steady State N5, N6 계산
    # (인스턴스 메서드 대신 클래스 내부 로직을 활용하거나 임시 모델 생성)
    temp_model = OGTTModel(ode_params, sys_params, {'si': 0.5, 'sigma': 0.5})
    n5, n6 = temp_model.find_steady_state_N(g0)
    
    y_curr = np.array([g0, i0, n5, n6])
    params = [0.5, 0.5] # si, sigma (임의의 고정값)
    
    print(f"[Init] y0 = [G={y_curr[0]:.2f}, I={y_curr[1]:.2f}, N5={y_curr[2]:.2f}, N6={y_curr[3]:.2f}]")
    print("-" * 60)
    
    # 3. Step-by-Step Simulation (t=0 to 10 min)
    dt = 1.0 # 1분 간격
    
    for t in range(0,120, 10): # 0분부터 120분까지 확인
        print(f"\n>>> Time t = {t} min")
        
        # A. Flux 확인 (가장 의심스러운 부분)
        # OgttSimul 내부 로직을 흉내내어 Flux 계산 값 확인
        # system.ode_func 내부에서 OGTTModel을 매번 생성하므로, 여기서도 동일하게 생성
        model = OGTTModel(ode_params, sys_params, {'si': params[0], 'sigma': params[1]})
        flux_val = model.calculate_ogtt_flux(float(t)) # float 형변환하여 전달
        print(f"  [Check Flux] OGTT_flux({t}) = {flux_val:.4f}")
        
        if t > 0 and flux_val == 0.0:
             print("  🔴 WARNING: Flux is 0! (Should be positive during OGTT)")

        # B. Drift 계산 (Bias 포함 여부 확인)
        drift = np.array(system.drift_func(t, y_curr, params))
        # 순수 ODE 값도 비교를 위해 계산
        drift_pure = np.array(system.ode_func(t, y_curr, params))
        bias = drift - drift_pure
        
        print(f"  [Check Drift] dG_dt (Total): {drift[0]:.4f}")
        print(f"                dG_dt (ODE)  : {drift_pure[0]:.4f}")
        print(f"                dG_dt (Bias) : {bias[0]:.4f}")

        # C. Diffusion 계산 (Scale 적용 여부 확인)
        diffusion_matrix = system.diffusion_func(t, y_curr, params)
        sigma_g = diffusion_matrix[0, 0]
        sigma_i = diffusion_matrix[1, 1]
        
        print(f"  [Check Diff]  Sigma_G (Scaled): {sigma_g:.4f}")
        if sigma_g > 10.0: # 0.1 scale이라면 보통 0.x ~ 3.0 사이여야 함 (원본 sigma가 30 정도라도)
             print(f"  🔴 WARNING: Sigma_G seems too large! (Is Scale={system.diffusion_scale} applied?)")

        # D. Update (Euler-Maruyama)
        # 노이즈 생성 (랜덤이지만 디버깅을 위해 고정된 값 1.0 가정해볼 수도 있음)
        # 여기서는 실제 랜덤 생성
        dW = np.random.normal(0, np.sqrt(dt), size=4)
        
        # y_next = y + drift*dt + diffusion*dW
        diffusion_term = diffusion_matrix @ dW
        y_next = y_curr + (drift_pure + 0.1*bias) * dt + diffusion_term
        
        # Clamping
        lower, upper = system.state_bounds
        y_next_clamped = np.clip(y_next, lower, upper)
        
        print(f"  [Update]      G_next = {y_curr[0]:.2f} + ({drift_pure[0]:.2f} + 0.1*{bias[0]:.2f}) * {dt} + ({diffusion_term[0]:.2f}) = {y_next_clamped[0]:.2f}")
        
        # 상태 업데이트
        y_curr = y_next_clamped

if __name__ == "__main__":
    debug_sde_step_by_step()