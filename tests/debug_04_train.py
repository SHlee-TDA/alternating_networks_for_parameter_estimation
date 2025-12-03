# tests/debug_04_train.py
import sys
import os
import torch
import torch.nn as nn
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from systems.ogtt_simul import OgttSimul
from main import get_experiment_dataloaders
from models import HiddenVarPredictor, ParameterEstimator

def verify_train_step():
    print("="*60)
    print("🧪 TEST 04: Training Step & Overfitting Check (Diagnosis Mode)")
    print("="*60)

    # 1. 환경 설정
    config = Config()
    config.BATCH_SIZE = 10
    device = torch.device(config.DEVICE)
    system = OgttSimul()

    print("-> Preparing Normalized Data Batch...")
    # 더미 데이터 (Raw Scale) - Test 02 통과 기준
    N, T = 100, 5
    obs_sim = np.random.uniform(80, 120, (N, T, 1)) 
    hid_sim = np.random.uniform(10, 50, (N, T, 1))
    params_sim = np.random.uniform(0.1, 1.5, (N, 2))
    t_points = np.array([0, 30, 60, 90, 120])
    sim_data_tuple = (obs_sim, hid_sim, params_sim, t_points)
    
    try:
        train_loader, _, _, _, _ = get_experiment_dataloaders(
            {'scenario': 'sim_only'}, sim_data_tuple, system, config
        )
        x_batch, y_batch, p_batch = next(iter(train_loader))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        p_batch = p_batch.to(device)
    except Exception as e:
        print(f"❌ Setup Failed: {e}")
        return

    # 2. 모델 초기화
    print("\n-> Initializing Models...")
    flat_x_dim = x_batch.shape[1]
    flat_y_dim = y_batch.shape[1]
    num_params = p_batch.shape[1]
    
    # g_phi (Parameter Estimator) 집중 테스트
    g_phi = ParameterEstimator(
        flat_x_dim, flat_y_dim, num_params,
        model_config=config.MODEL_CONFIG['g_phi']
    ).to(device)
    
    # Bias 초기화 강제 적용 (확인용)
    last_layer = list(g_phi.net.network.modules())[-1]
    if isinstance(last_layer, nn.Linear):
        nn.init.constant_(last_layer.bias, 0.5)
        # 가중치를 매우 작게 해서 초기 출력을 Bias(0.5)에 가깝게 유도
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.001) 

    # [중요] 학습률을 낮춰서 테스트 (1e-3 -> 1e-4)
    # 기존 1e-3에서 발산했다면, LR이 너무 큰 것일 수 있음
    TEST_LR = 1e-4 
    print(f"-> Optimizer LR: {TEST_LR}")
    
    optimizer = torch.optim.Adam(g_phi.parameters(), lr=TEST_LR)
    loss_fn = nn.MSELoss()

    # 3. 오버피팅 루프 (Diagnosis)
    print("\n[Running Overfitting Loop]")
    print(f"{'Iter':<5} | {'Loss':<10} | {'Pred Mean':<10} | {'Pred Min':<10} | {'Pred Max':<10}")
    print("-" * 55)
    
    losses = []
    
    # 초기 상태 확인
    with torch.no_grad():
        p_init = g_phi(x_batch, y_batch)
        loss_init = loss_fn(p_init, p_batch)
        print(f"{'Init':<5} | {loss_init.item():<10.4f} | {p_init.mean().item():<10.4f} | {p_init.min().item():<10.4f} | {p_init.max().item():<10.4f}")
        losses.append(loss_init.item())

    for i in range(50): # 횟수를 좀 더 늘림
        optimizer.zero_grad()
        
        p_pred = g_phi(x_batch, y_batch)
        loss = loss_fn(p_pred, p_batch)
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print(f"{i+1:<5} | {loss.item():<10.4f} | {p_pred.mean().item():<10.4f} | {p_pred.min().item():<10.4f} | {p_pred.max().item():<10.4f}")
        
        losses.append(loss.item())
        
    # 결과 분석
    print("-" * 55)
    start_loss = losses[0]
    end_loss = losses[-1]
    print(f"Start Loss: {start_loss:.6f} -> End Loss: {end_loss:.6f}")
    
    if end_loss < start_loss * 0.5:
        print("✅ PASS: Loss decreased significantly.")
    elif end_loss > start_loss:
        print("❌ FAIL: Loss INCREASED (Divergence).")
        print("   -> Check if Learning Rate is too high or Gradients are exploding.")
    else:
        print("⚠️ WARNING: Loss decreased but slowly.")

if __name__ == "__main__":
    verify_train_step()