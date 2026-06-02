"""
Probabilistic Inference Module

Handles the generation of parameter distributions from the trained CVAE models.
All outputs are kept as PyTorch Tensors on the original device.
"""

import torch
import numpy as np

@torch.no_grad()
def single_cvae_sampling(baseline_net, x_observed, num_samples=1000):
    """
    [Baseline] Single CVAE의 1-shot Inference.
    은닉 상태를 무시하고 오직 관측치에만 의존하여 분포를 생성합니다.
    """
    baseline_net.eval()
    
    # baseline_net.sample() 내부에서 x_observed 차원 확장을 처리한다고 가정
    theta_samples = baseline_net.sample(x_observed, num_samples=num_samples)
    
    return theta_samples

@torch.no_grad()
def pseudo_gibbs_sampling(
    hidden_cvae, param_cvae, x_observed, theta_dim=None,
    num_chains=100, num_steps=50, burn_in=10, 
    temperature_y=1.0, temperature_p=1.0,
    bounds_y=None, bounds_p=None,
    init_theta=None
):
    hidden_cvae.eval()
    param_cvae.eval()
    
    batch_size = x_observed.size(0)
    device = x_observed.device
    if  theta_dim is None:
        if hasattr(param_cvae, 'theta_dim'):
            theta_dim = param_cvae.theta_dim
        else:
            raise AttributeError("theta_dim must be provided or param_cvae must have attribute 'theta_dim'")
    
    x_repeated = x_observed.repeat_interleave(num_chains, dim=0)
    
    if init_theta is None:
        theta_curr = torch.randn(batch_size * num_chains, theta_dim, device=device)
    else:
        theta_curr = init_theta.repeat_interleave(num_chains, dim=0)

    # [수정] 궤적(history)을 저장할 리스트만 남깁니다.
    theta_history = []

    base_noise_y = getattr(hidden_cvae, 'infer_noise_y', 0.05) 
    base_noise_p = getattr(param_cvae, 'infer_noise_p', 0.05)
    
    infer_noise_y = base_noise_y * temperature_y
    infer_noise_p = base_noise_p * temperature_p
    
    # Ping-pong loop
    for step in range(num_steps):
        # 1. 은닉 상태(y) Langevin Step
        z_A = torch.randn(batch_size * num_chains, hidden_cvae.latent_dim, device=device)
        y_mean = hidden_cvae.decode(z_A, x_repeated, theta_curr)
        y_curr = y_mean + torch.randn_like(y_mean) * infer_noise_y
        
        if bounds_y is not None:
            y_curr = torch.clamp(y_curr, min=bounds_y[0], max=bounds_y[1])
            
        # 2. 파라미터(theta) Langevin Step
        z_B = torch.randn(batch_size * num_chains, param_cvae.latent_dim, device=device)
        theta_mean = param_cvae.decode(z_B, x_repeated, y_curr)
        theta_curr = theta_mean + torch.randn_like(theta_mean) * infer_noise_p
        
        if bounds_p is not None:
            theta_curr = torch.clamp(theta_curr, min=bounds_p[0], max=bounds_p[1])
            
        # 3. 궤적 기록 (매 스텝마다 저장)
        theta_history.append(theta_curr.view(batch_size, num_chains, -1).cpu())

    # [수정] 4. 루프가 모두 끝난 후, 1,000개 체인의 가장 마지막 상태(Final)만 추출
    final_y_samples = y_curr.view(batch_size, num_chains, -1)
    final_theta_samples = theta_curr.view(batch_size, num_chains, -1)
    
    # History는 (Batch, Chains, Steps, Dim) 형태로 스택하여 추후 수렴성 분석에 사용
    theta_history = torch.stack(theta_history, dim=2) 
    
    return final_y_samples, final_theta_samples, theta_history