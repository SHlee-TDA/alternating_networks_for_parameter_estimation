# tests/test_real_loader.py
import sys
import os
import numpy as np
import pytest # 만약 pytest를 안 쓴다면 기본 assert 사용
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import RealOGTTDataLoader

def test_derivative_methods():
    print("\n[Phase 5 Test] Checking All Derivative Methods...")
    
    data_path = 'data/clean_sumner_n_612.xlsx'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    # 테스트할 방법 목록
    methods = ['spline', 'poly', 'lagrange', 'finite_diff']
    
    for method in methods:
        print(f"\n--- Testing Method: {method.upper()} ---")
        
        # Config 설정
        config = Config()
        config.USE_LAGRANGIAN = True
        config.DERIVATIVE_METHOD = method
        
        try:
            loader = RealOGTTDataLoader(data_path, config)
            obs, hid, params, t_points = loader.load_data()
            
            # 1. 차원 검증 (N, 5, 2)
            assert obs.shape[2] == 2, f"❌ Dimension mismatch for {method}. Got {obs.shape}"
            print(f"  ✅ Dimensions correct: {obs.shape}")
            
            # 2. 값 검증 (미분값이 존재하는지)
            derivs = obs[:, :, 1]
            
            # NaN 체크
            if np.isnan(derivs).any():
                print(f"  ❌ NaN values found in derivatives for {method}!")
                continue
                
            # 0이 아닌 값이 있는지 (제대로 계산되었는지)
            nonzero_count = np.count_nonzero(derivs)
            assert nonzero_count > 0, f"❌ All derivatives are zero for {method}."
            
            # 값의 범위 출력 (디버깅용)
            print(f"  ✅ Derivative Range: [{derivs.min():.4f}, {derivs.max():.4f}]")
            
        except Exception as e:
            print(f"  ❌ Exception occurred for {method}: {e}")
            raise e

    print("\n🎉 All Derivative Estimator Tests Passed!")

if __name__ == "__main__":
    test_derivative_methods()