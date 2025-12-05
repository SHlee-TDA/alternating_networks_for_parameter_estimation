import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from systems.ogtt_simul import OgttSimul
from main import get_experiment_dataloaders
from models import HiddenVarPredictor, ParameterEstimator

def experiment_recurrent_loss():
    print("="*60)
    print("🧪 EXPERIMENT: Static vs. Recurrent (with Spectral Norm)")
    print("   Hypothesis: SN guarantees stability, Recurrent Loss improves accuracy.")
    print("="*60)

    # 1. 설정
    config = Config()
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    
    # [핵심 수정] 이론적 전제조건인 Spectral Norm을 반드시 켜야 합니다!
    config.USE_SPECTRAL_NORM = True 
    print(f"-> Spectral Normalization: {config.USE_SPECTRAL_NORM} (Essential for stability)")
    
    device = torch.device(config.DEVICE)
    system = OgttSimul()

    # 데이터 준비
    print("-> Loading Data...")
    # (데이터 생성 및 로더 준비 코드는 기존과 동일)
    N, T = 1000, 5
    obs_sim = np.random.uniform(80, 120, (N, T, 1))
    hid_sim = np.random.uniform(10, 50, (N, T, 1))
    params_sim = np.random.uniform(0.1, 1.5, (N, 2))
    t_points = np.array([0, 30, 60, 90, 120])
    sim_data_tuple = (obs_sim, hid_sim, params_sim, t_points)
    
    train_loader, _, _, _, normalizer = get_experiment_dataloaders(
        {'scenario': 'sim_only'}, sim_data_tuple, system, config
    )
    
    # 모델 초기화
    print("-> Initializing Models...")
    x_batch, y_batch, p_batch = next(iter(train_loader))
    x_dim, y_dim, p_dim = x_batch.shape[1], y_batch.shape[1], p_batch.shape[1]
    
    # Config의 설정을 명시적으로 전달
    model_conf_f = config.MODEL_CONFIG['f_theta']
    model_conf_g = config.MODEL_CONFIG['g_phi']
    
    base_f = HiddenVarPredictor(
        x_dim, y_dim, p_dim, model_conf_f, 
        use_spectral_norm=config.USE_SPECTRAL_NORM # [중요] SN 적용
    )
    base_g = ParameterEstimator(
        x_dim, y_dim, p_dim, model_conf_g, 
        use_spectral_norm=config.USE_SPECTRAL_NORM # [중요] SN 적용
    )
    
    # Bias Init 적용 (Normalized Space 기준 0.0 근처가 보통 중간값)
    # Log-Normalizer라면 0.0이 중간값이 아닐 수 있으므로 주의 필요
    # 여기서는 학습 안정성을 위해 0.0 (혹은 mean)으로 초기화
    #for m in [base_f, base_g]:
    #    for layer in reversed(m.net.network):
    #        if isinstance(layer, nn.Linear):
    #            nn.init.constant_(layer.bias, 0.0) 
    #            break

    model_static = {'f': copy.deepcopy(base_f).to(device), 'g': copy.deepcopy(base_g).to(device)}
    model_recur  = {'f': copy.deepcopy(base_f).to(device), 'g': copy.deepcopy(base_g).to(device)}
    
    opt_static = torch.optim.Adam(list(model_static['f'].parameters()) + list(model_static['g'].parameters()), lr=config.LEARNING_RATE)
    opt_recur  = torch.optim.Adam(list(model_recur['f'].parameters()) + list(model_recur['g'].parameters()), lr=config.LEARNING_RATE)
    
    loss_fn = nn.MSELoss()

    # 2. 학습 루프 (Normalized Space에서 학습)
    epochs = 1000
    print(f"\n-> Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        static_losses = []
        recur_losses = []
        
        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            
            # --- Model A: Static ---
            opt_static.zero_grad()
            y_hat = model_static['f'](x, p)
            p_hat = model_static['g'](x, y_hat)
            loss_s = loss_fn(p_hat, p)
            loss_s.backward()
            opt_static.step()
            static_losses.append(loss_s.item())
            
            # --- Model B: Recurrent ---
            opt_recur.zero_grad()
            
            # Noise Injection (Normalized Scale에서 노이즈 추가)
            noise = torch.randn_like(p) * 0.1 
            p_curr = p + noise
            
            K = 3
            traj_loss = 0
            for k in range(K):
                y_hat = model_recur['f'](x, p_curr)
                p_next = model_recur['g'](x, y_hat)
                traj_loss += loss_fn(p_next, p) 
                p_curr = p_next
            
            traj_loss.backward()
            opt_recur.step()
            recur_losses.append(traj_loss.item() / K)

        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}: Static Loss={np.mean(static_losses):.4f} | Recur Loss={np.mean(recur_losses):.4f}")

    # 3. 검증 및 시각화 (Denormalized Scale)
    print("\n-> Testing Convergence (Physical Scale)...")
    
    x_test, _, p_true = next(iter(train_loader))
    x_test, p_true = x_test[0:1].to(device), p_true[0:1].to(device)
    
    # 시작점을 정답에서 약간 떨어진 곳으로 설정 (Normalized Scale)
    p_start_norm = p_true + torch.tensor([[0.5, -0.5]]).to(device) 
    
    def get_trajectory_denorm(models, p_init_norm, steps=10):
        # 시작점 역정규화
        traj = [normalizer.denormalize_params(p_init_norm).detach().cpu().numpy()]
        p_curr = p_init_norm
        
        with torch.no_grad():
            for _ in range(steps):
                y = models['f'](x_test, p_curr)
                p_curr = models['g'](x_test, y)
                
                # 매 스텝 결과 역정규화하여 저장
                p_curr_denorm = normalizer.denormalize_params(p_curr)
                traj.append(p_curr_denorm.detach().cpu().numpy())
                
        return np.concatenate(traj, axis=0)

    # 궤적 추출 (물리적 단위로 변환됨)
    traj_static = get_trajectory_denorm(model_static, p_start_norm)
    traj_recur = get_trajectory_denorm(model_recur, p_start_norm)
    
    # 정답 역정규화
    p_true_denorm = normalizer.denormalize_params(p_true).detach().cpu().numpy()

    # 4. 시각화
    plt.figure(figsize=(8, 6))
    
    # Target
    plt.plot(p_true_denorm[0, 0], p_true_denorm[0, 1], 'kX', markersize=15, label='Target (True)')
    
    # Start
    plt.plot(traj_static[0, 0], traj_static[0, 1], 'go', markersize=10, label='Start')
    
    # Trajectories
    plt.plot(traj_static[:, 0], traj_static[:, 1], 'b-o', label='Static Model', alpha=0.7)
    plt.plot(traj_recur[:, 0], traj_recur[:, 1], 'r-o', label='Recurrent Model', alpha=0.7)
    
    # End Points
    plt.plot(traj_static[-1, 0], traj_static[-1, 1], 'b*', markersize=12)
    plt.plot(traj_recur[-1, 0], traj_recur[-1, 1], 'r*', markersize=12)
    
    plt.xlabel("Parameter 1 (si)")
    plt.ylabel("Parameter 2 (sigma)")
    plt.title("Convergence Comparison (Physical Scale)")
    plt.legend()
    plt.grid(True)
    
    save_path = 'test_recurrent_convergence_denorm.png'
    plt.savefig(save_path)
    print(f"   Saved plot to {save_path}")
    
    # 거리 계산 (물리적 거리)
    dist_static = np.linalg.norm(traj_static[-1] - p_true_denorm)
    dist_recur = np.linalg.norm(traj_recur[-1] - p_true_denorm)
    
    print(f"\n[Final Error Distance (Physical)]")
    print(f"   Static Model: {dist_static:.4f}")
    print(f"   Recurrent Model: {dist_recur:.4f}")

if __name__ == "__main__":
    experiment_recurrent_loss()