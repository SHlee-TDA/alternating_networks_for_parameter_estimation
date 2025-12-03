# tests/debug_02_loader.py
import sys
import os
import torch
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from systems.ogtt_simul import OgttSimul
from main import get_experiment_dataloaders
from utils import Normalizer

def verify_loader():
    print("="*60)
    print("🧪 TEST 02: Data Loader & Preprocessing Logic Check (Complete)")
    print("="*60)
    
    # 1. 테스트 환경 설정
    config = Config()
    config.BATCH_SIZE = 10 
    system = OgttSimul()
    
    print("-> Creating dummy simulation data (Raw Scale)...")
    # Glucose: 80 ~ 120, Insulin: 10 ~ 50, Params: 0.1 ~ 1.5
    N = 100
    T = 5
    obs_sim = np.random.uniform(80, 120, (N, T, 1)) 
    hid_sim = np.random.uniform(10, 50, (N, T, 1))
    params_sim = np.random.uniform(0.1, 1.5, (N, 2))
    t_points = np.array([0, 30, 60, 90, 120])
    
    sim_data_tuple = (obs_sim, hid_sim, params_sim, t_points)
    
    print("-> Initializing DataLoader...")
    exp_config = {'scenario': 'sim_only'}
    try:
        train_loader, _, _, _, normalizer = get_experiment_dataloaders(
            exp_config, sim_data_tuple, system, config
        )
    except Exception as e:
        print(f"❌ CRITICAL FAIL: Loader initialization failed: {e}")
        return
    
    print("\n-> Fetching first batch...")
    try:
        x_batch, y_batch, p_batch = next(iter(train_loader))
    except Exception as e:
        print(f"❌ CRITICAL FAIL: Failed to fetch batch: {e}")
        return
    
    # 2. 검증 (Checkpoints)
    print("\n[Checkpoints]")
    
    # CP 1: 입력 데이터(X - Glucose) 확인
    x_mean = x_batch.mean().item()
    x_max = x_batch.max().item()
    
    print(f"1. Input(X: Glucose) Stats: Mean={x_mean:.4f}, Max={x_max:.4f}")
    if x_mean > 50:
        print("   ℹ️  STATUS: Glucose is RAW SCALE (~100).")
    elif x_max < 5:
        print("   ℹ️  STATUS: Glucose is NORMALIZED (~1).")
        
    # [추가됨] CP 2: 히든 데이터(Y - Insulin) 확인
    y_mean = y_batch.mean().item()
    y_max = y_batch.max().item()
    y_min = y_batch.min().item()
    
    print(f"2. Hidden(Y: Insulin) Stats: Mean={y_mean:.4f}, Min={y_min:.4f}, Max={y_max:.4f}")
    
    if y_mean > 10:
        print("   ℹ️  STATUS: Insulin is RAW SCALE (~30).")
    elif y_max < 5:
        print("   ℹ️  STATUS: Insulin is NORMALIZED (~1).")
    
    if y_min < 0:
        print("   ❌ FAIL: Insulin contains negative values! Check normalization logic.")
    else:
        print("   ✅ PASS: Insulin is non-negative.")

    # CP 3: 타겟 데이터(P) 확인
    p_mean = p_batch.mean().item()
    p_min = p_batch.min().item()
    
    print(f"3. Target(P) Stats: Mean={p_mean:.4f}, Min={p_min:.4f}")
    
    if p_min < 0:
        print("   ⚠️ WARNING: Target contains negative values.")
    elif p_mean == 0:
         print("   ❌ FAIL: Target is all zeros!")
    else:
        print("   ✅ PASS: Target range looks valid.")
        
    # CP 4: Normalizer 로직 정합성
    print("4. Checking Normalizer Reversibility...")
    dummy_p_raw = torch.tensor([[0.5, 1.0]], device=config.DEVICE)
    p_norm = normalizer.normalize_params(dummy_p_raw)
    p_recovered = normalizer.denormalize_params(p_norm)
    
    if torch.allclose(dummy_p_raw, p_recovered, atol=1e-5):
        print("   ✅ PASS: Normalizer is reversible.")
    else:
        print("   ❌ FAIL: Normalizer is NOT reversible!")

if __name__ == "__main__":
    verify_loader()