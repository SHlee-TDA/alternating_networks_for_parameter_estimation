# tests/test_data_aug.py
import sys
import os
import numpy as np
import torch
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from systems.ogtt_simul import OgttSimul
from data_loader import DataGenerator

def test_data_generation_pipeline():
    print("\n[Phase 4 Test] Data Augmentation Pipeline Check...")
    
    # 1. 테스트 환경 설정
    config = Config()
    config.NUM_SAMPLES = 100 # 테스트용 소량 생성
    config.USE_SDE = True     # SDE 모드
    config.USE_LAGRANGIAN = True # Lagrangian Feature (미분) 사용 가정
    
    # 파일 경로 설정
    save_dir = Path('data') / config.SYSTEM_NAME
    expected_file = save_dir / f"augmented_data_sde_{config.NUM_SAMPLES}.npz"
    
    # 기존 파일 삭제 (새로 생성 테스트)
    if expected_file.exists():
        try:
            os.remove(expected_file)
            print(f"  -> Removed existing test file: {expected_file}")
        except PermissionError:
            print(f"  -> Warning: Could not remove {expected_file}. Testing overwrite.")

    # 2. DataGenerator 실행
    system = OgttSimul()
    generator = DataGenerator(system, config)
    
    print(f"  -> Generating {config.NUM_SAMPLES} samples (SDE mode)...")
    # generator.generate_data() returns: observed, hidden, params, t_points
    obs, hid, params, t_points = generator.generate_data()
    
    # 3. 검증 로직
    # 3-1. 파일 생성 확인
    assert expected_file.exists(), f"❌ .npz file was not created at {expected_file}"
    print(f"  ✅ File saving successful at {expected_file}")
    
    # 3-2. 차원 확인
    # USE_LAGRANGIAN=True 이면 관측 변수는 2차원 (값, 미분)
    expected_obs_dim = 2 if getattr(config, 'USE_LAGRANGIAN', False) else 1
    expected_hid_dim = 1 
    
    assert obs.shape == (config.NUM_SAMPLES, 5, expected_obs_dim), \
        f"❌ Observed shape mismatch: Expected (..., {expected_obs_dim}), Got {obs.shape}"
    assert hid.shape == (config.NUM_SAMPLES, 5, expected_hid_dim), \
        f"❌ Hidden shape mismatch: Expected (..., {expected_hid_dim}), Got {hid.shape}"
    assert params.shape == (config.NUM_SAMPLES, 2), \
        f"❌ Params shape mismatch: {params.shape}"
    print(f"  ✅ Data dimensions correct (Observed dim: {expected_obs_dim}).")
    
    # 3-3. 물리적 정합성 (양수 확인) - Rejection Sampling 검증
    print("\n  -> Checking Physical Constraints...")
    
    # [핵심 수정] 농도(Concentration)와 도함수(Derivative) 분리 검사
    # obs[:, :, 0] = 농도 (양수여야 함)
    # obs[:, :, 1] = 미분 (음수일 수 있음)
    
    glucose_conc = obs[:, :, 0]
    insulin_conc = hid[:, :, 0]
    
    min_g_conc = glucose_conc.min()
    min_i_conc = insulin_conc.min()
    min_params = params.min()
    
    print(f"     Min Glucose Conc : {min_g_conc:.4f}")
    print(f"     Min Insulin Conc : {min_i_conc:.4f}")
    print(f"     Min Params       : {min_params:.4f}")
    
    # 농도는 반드시 양수
    assert min_g_conc > 0, f"❌ Found non-positive Glucose Concentration! ({min_g_conc})"
    assert min_i_conc > 0, f"❌ Found non-positive Insulin Concentration! ({min_i_conc})"
    assert min_params > 0, f"❌ Found non-positive Parameters! ({min_params})"
    print("     ✅ Concentration variables are positive.")

    # 도함수는 음수 허용 (정보 출력만)
    if obs.shape[2] > 1:
        glucose_deriv = obs[:, :, 1]
        print(f"     Glucose Derivative Range: [{glucose_deriv.min():.4f}, {glucose_deriv.max():.4f}]")
        print("     ✅ Derivative values exist (Negative values are natural).")

    # 3-4. 시간 축 확인
    expected_t = np.array([0, 30, 60, 90, 120])
    assert np.array_equal(t_points, expected_t), f"❌ Time points mismatch. Got {t_points}"
    print("  ✅ Time points correct.")

    print("\n🎉 Data Augmentation Pipeline Test Passed!")

if __name__ == "__main__":
    if not os.path.exists("distribution_params.json"):
        print("⚠️ 'distribution_params.json' not found. Run 'distribution_analysis.py' first.")
    else:
        test_data_generation_pipeline()