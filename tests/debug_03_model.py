# tests/debug_03_model_init.py
import sys
import os
import torch
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from systems.ogtt_simul import OgttSimul
from main import get_experiment_dataloaders
from models import HiddenVarPredictor, ParameterEstimator

def verify_model_init():
    print("="*60)
    print("🧪 TEST 03: Model Initialization & Forward Pass Check")
    print("="*60)

    # 1. 환경 설정 및 데이터 준비 (Test 02와 동일)
    config = Config()
    config.BATCH_SIZE = 10
    device = torch.device(config.DEVICE)
    system = OgttSimul()

    print("-> Preparing Normalized Data Batch...")
    # 더미 데이터 (Raw Scale)
    N, T = 100, 5
    obs_sim = np.random.uniform(80, 120, (N, T, 1)) 
    hid_sim = np.random.uniform(10, 50, (N, T, 1))
    params_sim = np.random.uniform(0.1, 1.5, (N, 2))
    t_points = np.array([0, 30, 60, 90, 120])
    sim_data_tuple = (obs_sim, hid_sim, params_sim, t_points)
    
    try:
        # main.py의 로더 생성 (Normalizer 적용됨)
        train_loader, _, _, _, _ = get_experiment_dataloaders(
            {'scenario': 'sim_only'}, sim_data_tuple, system, config
        )
        x_batch, y_batch, p_batch = next(iter(train_loader))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        p_batch = p_batch.to(device)
        print(f"   Batch Loaded. X:{x_batch.shape}, Y:{y_batch.shape}, P:{p_batch.shape}")
    except Exception as e:
        print(f"❌ Setup Failed: {e}")
        return

    # 2. 모델 초기화
    print("\n-> Initializing Models (f_theta, g_phi)...")
    try:
        flat_x_dim = x_batch.shape[1]
        flat_y_dim = y_batch.shape[1]
        num_params = p_batch.shape[1]
        
        # Config에서 첫 번째 실험 설정 가져오기 (Spectral Norm 여부 등)
        exp_config = config.EXPERIMENTS[0] if config.EXPERIMENTS else {'use_spectral_norm': False}
        use_sn = exp_config.get('use_spectral_norm', False)
        
        # f_theta (Hidden Predictor)
        f_theta = HiddenVarPredictor(
            flat_x_dim, flat_y_dim, num_params,
            model_config=config.MODEL_CONFIG['f_theta'],
            use_spectral_norm=use_sn, 
            initialization_config=config.MODEL_CONFIG.get('initialization')
        ).to(device)
        
        # g_phi (Parameter Estimator)
        g_phi = ParameterEstimator(
            flat_x_dim, flat_y_dim, num_params,
            model_config=config.MODEL_CONFIG['g_phi'],
            use_spectral_norm=use_sn,
            initialization_config=config.MODEL_CONFIG.get('initialization')
        ).to(device)
        
        print("   ✅ Models initialized successfully.")
        print(f"      Spectral Norm: {use_sn}")
        
    except Exception as e:
        print(f"   ❌ Model Init Failed: {e}")
        return

    # 3. Forward Pass 검증 (Checkpoints)
    print("\n[Checkpoints]")
    
    # CP 1: f_theta (X, P -> Y)
    print("1. Checking f_theta Output (Predict Hidden)...")
    try:
        y_pred = f_theta(x_batch, p_batch)
        y_mean = y_pred.mean().item()
        y_std = y_pred.std().item()
        
        print(f"   Output Stats: Mean={y_mean:.4f}, Std={y_std:.4f}")
        
        if torch.isnan(y_pred).any():
             print("   ❌ FAIL: NaN detected in f_theta output.")
        elif y_std == 0:
             print("   ⚠️ WARNING: Output is constant (Dead Neuron?).")
        else:
             print("   ✅ PASS: f_theta output looks valid.")
    except Exception as e:
        print(f"   ❌ FAIL: Error in f_theta: {e}")

    # CP 2: g_phi (X, Y -> P)
    print("2. Checking g_phi Output (Predict Params)...")
    try:
        p_pred = g_phi(x_batch, y_batch)
        p_mean = p_pred.mean().item()
        p_std = p_pred.std().item()
        p_min = p_pred.min().item()
        p_max = p_pred.max().item()
        
        print(f"   Output Stats: Mean={p_mean:.4f}, Std={p_std:.4f}")
        print(f"   Range: [{p_min:.4f}, {p_max:.4f}]")
        
        if torch.isnan(p_pred).any():
            print("   ❌ FAIL: NaN detected in g_phi output.")
        
        elif p_mean == 0 and p_std == 0:
            print("   ❌ FAIL: Output is ALL ZEROS!")
            print("      -> Suggestion: Apply 'Bias Initialization' to start near 0.5.")
            
        elif p_max > 2.0 or p_min < -2.0:
             print("   ⚠️ WARNING: Output range is quite large. Check initialization.")
             
        else:
             print("   ✅ PASS: g_phi output looks valid.")
             
    except Exception as e:
        print(f"   ❌ FAIL: Error in g_phi: {e}")

if __name__ == "__main__":
    verify_model_init()