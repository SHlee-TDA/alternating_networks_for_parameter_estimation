# tests/test_pipeline.py
import sys
import os
import numpy as np
import torch

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import RealOGTTDataLoader
from systems.ogtt_simul import OGTTModel, ode_params, sys_params

def test_dataloader_integrity():
    print("\n[Test 1] Checking RealOGTTDataLoader Integrity...")
    config = Config()
    # 파일 경로는 실제 위치에 맞게 조정해주세요
    file_path = 'data/clean_sumner_n_612.xlsx'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None, None, None

    loader = RealOGTTDataLoader(file_path, config)
    X_obs, Y_hid, P_true, t_points = loader.load_data()
    
    # 1. 차원 검증
    N = X_obs.shape[0]
    print(f"  -> Loaded {N} samples.")
    
    # Glucose (Observed): (N, 5, 1)
    assert X_obs.ndim == 3 and X_obs.shape[1] == 5 and X_obs.shape[2] == 1, \
        f"❌ Glucose shape mismatch: Expected (N, 5, 1), Got {X_obs.shape}"
    
    # Insulin (Hidden): (N, 5, 1)
    assert Y_hid.ndim == 3 and Y_hid.shape[1] == 5 and Y_hid.shape[2] == 1, \
        f"❌ Insulin shape mismatch: Expected (N, 5, 1), Got {Y_hid.shape}"
        
    # Params: (N, 2)
    assert P_true.ndim == 2 and P_true.shape[1] == 2, \
        f"❌ Parameter shape mismatch: Expected (N, 2), Got {P_true.shape}"
        
    # Time points check
    expected_t = np.array([0, 30, 60, 90, 120])
    assert np.array_equal(t_points, expected_t), \
        f"❌ Time points mismatch: Expected {expected_t}, Got {t_points}"
        
    # NaN Check
    assert not np.isnan(X_obs).any(), "❌ Found NaNs in Glucose data!"
    assert not np.isnan(Y_hid).any(), "❌ Found NaNs in Insulin data!"
    assert not np.isnan(P_true).any(), "❌ Found NaNs in Parameters!"
    
    print("✅ Data Loader integrity check passed.")
    return X_obs, Y_hid, P_true, t_points

def test_simulation_dry_run(X_obs, Y_hid, P_true, t_points):
    print("\n[Test 2] Running Simulation Dry-Run (First 3 Samples)...")
    
    # 처음 3개의 샘플에 대해서만 시뮬레이션 테스트
    subset_n = 3
    
    for i in range(subset_n):
        try:
            # 파라미터 설정
            si_val = P_true[i, 0]
            sigma_val = P_true[i, 1]
            theta = {'si': si_val, 'sigma': sigma_val}
            
            model = OGTTModel(ode_params, sys_params, theta)
            
            # 초기값 설정
            g0 = X_obs[i, 0, 0]
            i0 = Y_hid[i, 0, 0]
            n5_0, n6_0 = model.find_steady_state_N(g0)
            y0 = [g0, i0, n5_0, n6_0]
            
            # 시뮬레이션
            sol = model.simulate(t_span=[0, 120], initial_conditions=y0, t_eval=t_points)
            
            if not sol.success:
                print(f"❌ Simulation failed for sample {i}: {sol.message}")
            else:
                # 결과 shape 확인 (4 variables, 5 time points)
                if sol.y.shape != (4, 5):
                     print(f"❌ Output shape mismatch for sample {i}: {sol.y.shape}")
                else:
                    print(f"  -> Sample {i}: Simulation success. G(end)={sol.y[0, -1]:.2f}")
                    
        except Exception as e:
            print(f"❌ Exception during simulation sample {i}: {str(e)}")
            raise e
            
    print("✅ Simulation dry-run passed.")

if __name__ == "__main__":
    try:
        X, Y, P, t = test_dataloader_integrity()
        if X is not None:
            test_simulation_dry_run(X, Y, P, t)
            print("\n🎉 All tests passed! System is ready for next steps.")
        else:
            print("\n⚠️ Test skipped due to missing data file.")
    except Exception as e:
        print(f"\n❌ Test Failed with error: {e}")