# tests/debug_01_generation.py
import sys
import os
import shutil
import numpy as np
import torch
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from systems.ogtt_simul import OgttSimul
from data_loader import DataGenerator
from config import Config

def verify_generation():
    print("="*60)
    print("DEBUG 01: Data Generation Pipeline Check")
    print("="*60)
    
    # 1. 테스트용 설정 및 생성기 초기화
    config = Config()
    
    # [설정 수정] 테스트를 위해 소량의 데이터만 생성하고 SDE/ODE 모드 설정
    config.NUM_SAMPLES = 50           # 테스트용 샘플 수
    config.USE_SDE = True            # 기본 ODE 테스트 (필요시 True로 변경 가능)
    config.USE_LAGRANGIAN = False     # 기본 차원 확인
    config.SYSTEM_NAME = 'test_ogtt'  # 결과 파일이 저장될 폴더명 (기존 데이터 덮어쓰기 방지)
    
    # [임시 폴더 정리] 이전 테스트 잔여물 삭제
    test_data_dir = Path('data') / config.SYSTEM_NAME
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
        print(f"-> Cleared temporary test directory: {test_data_dir}")

    # 시스템 및 제너레이터 인스턴스화
    system = OgttSimul()
    generator = DataGenerator(system, config) 
    
    print(f"-> Initialized DataGenerator for system: {system.name}")
    print(f"-> Generating {config.NUM_SAMPLES} samples (Parallel Execution)...")
    
    # 2. 데이터 생성 실행 (DataGenerator.generate_data 호출)
    # 이 메서드는 내부적으로 _generate_one_sample을 병렬로 실행하고 .npz로 저장 후 로드합니다.
    try:
        # returns: observed_data, hidden_data, params_data, t_points
        obs_data, hid_data, params_data, t_points = generator.generate_data()
    except Exception as e:
        print(f"❌ CRITICAL FAIL: Generator raised an exception: {e}")
        return

    # 3. 데이터 로드 확인
    print(f"-> Generation Complete.")
    print(f"   Observed Shape: {obs_data.shape} (Expect ({config.NUM_SAMPLES}, 5, 1))")
    print(f"   Hidden Shape  : {hid_data.shape} (Expect ({config.NUM_SAMPLES}, 5, 1))")
    print(f"   Params Shape  : {params_data.shape} (Expect ({config.NUM_SAMPLES}, 2))")

    # 4. 정합성 검증 (Checkpoints)
    print("\n[Checkpoints]")
    
    # CP 1: 데이터 차원 검증
    # DataGenerator는 (N, T, Dim) 형태로 반환해야 함
    if obs_data.ndim == 3 and obs_data.shape[1] == 5:
        print("   ✅ PASS: Data dimensions are correct.")
    else:
        print(f"   ❌ FAIL: Incorrect dimensions. Got {obs_data.shape}")

    # CP 2: 생리학적 범위 검증 (Glucose & Insulin)
    # Glucose: 보통 50 ~ 400 mg/dL 사이
    # Insulin: 보통 0 ~ 300 uU/mL 사이 (로그 정규분포라 0에 가까울 수 있음)
    min_g, max_g = np.min(obs_data), np.max(obs_data)
    min_i, max_i = np.min(hid_data), np.max(hid_data)
    
    print(f"   Glucose Range: {min_g:.2f} ~ {max_g:.2f}")
    print(f"   Insulin Range: {min_i:.2f} ~ {max_i:.2f}")
    
    if min_g < 0 or min_i < 0:
        print("   ❌ FAIL: Negative physiological values found!")
    elif max_g > 1000 or max_i > 5000:
        print("   ⚠️ WARNING: Extremely high physiological values found. Check stability.")
    else:
        print("   ✅ PASS: Physiological ranges look reasonable.")

    # CP 3: 파라미터 범위 검증 (si, sigma)
    # ogtt_simul.py에 정의된 범위: si [0, 2], sigma [0, 2]
    # 하지만 LogNormal 샘플링 시 이 범위를 벗어날 수도 있음 (분포 파라미터에 따라)
    # 여기서는 음수 여부와 극단적 값 여부를 체크
    si_vals = params_data[:, 0]
    sigma_vals = params_data[:, 1]
    
    min_si, max_si = np.min(si_vals), np.max(si_vals)
    min_sig, max_sig = np.min(sigma_vals), np.max(sigma_vals)
    
    print(f"   Param 'si' Range   : {min_si:.4f} ~ {max_si:.4f}")
    print(f"   Param 'sigma' Range: {min_sig:.4f} ~ {max_sig:.4f}")

    if min_si < 0 or min_sig < 0:
        print("   ❌ FAIL: Negative parameters generated!")
    elif max_si > 10 or max_sig > 10:
        print("   ⚠️ WARNING: Parameters are unusually large (>10). Check distribution params.")
    else:
        print("   ✅ PASS: Parameter ranges look valid.")

    # CP 4: NaN / Inf 검증
    if np.isnan(obs_data).any() or np.isinf(obs_data).any():
        print("   ❌ FAIL: NaN or Inf values found in Observed Data!")
    elif np.isnan(hid_data).any() or np.isinf(hid_data).any():
        print("   ❌ FAIL: NaN or Inf values found in Hidden Data!")
    else:
        print("   ✅ PASS: No NaN/Inf values detected.")

    # 테스트 종료 후 정리 (선택 사항)
    # shutil.rmtree(test_data_dir)
    # print(f"\n-> Cleaned up test directory.")

if __name__ == "__main__":
    verify_generation()