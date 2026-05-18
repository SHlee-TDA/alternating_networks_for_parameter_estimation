import torch
import numpy as np

@torch.no_grad()
def pseudo_gibbs_sampling(
    hidden_cvae, param_cvae, x_observed, 
    num_chains=100, num_steps=50, burn_in=10, 
    temperature_y=1.0, temperature_p=1.0,
    bounds_y=None, bounds_p=None,
    init_theta=None
):
    hidden_cvae.eval()
    param_cvae.eval()
    
    batch_size = x_observed.size(0)
    device = x_observed.device
    theta_dim = param_cvae.decoder_net[-2].out_features
    
    x_repeated = x_observed.repeat_interleave(num_chains, dim=0)
    
    if init_theta is None:
        theta_curr = torch.randn(batch_size * num_chains, theta_dim, device=device)
    else:
        theta_curr = init_theta.repeat_interleave(num_chains, dim=0)

    # 샘플을 저장할 리스트
    y_samples_list = []
    theta_samples_list = []

    base_noise_y = getattr(hidden_cvae, 'infer_noise_y', 0.05) 
    base_noise_p = getattr(param_cvae, 'infer_noise_p', 0.05)
    
    infer_noise_y = base_noise_y * temperature_y
    infer_noise_p = base_noise_p * temperature_p
    
    # 3. 핑퐁 루프 (The Alternating Process)
    for step in range(num_steps):
        z_A = torch.randn(batch_size * num_chains, hidden_cvae.latent_dim, device=device)
        y_mean = hidden_cvae.decode(z_A, x_repeated, theta_curr)
        y_curr = y_mean + torch.randn_like(y_mean) * infer_noise_y
        if bounds_y is not None:
            y_curr = torch.clamp(y_curr, min=bounds_y[0], max=bounds_y[1])
            
        z_B = torch.randn(batch_size * num_chains, param_cvae.latent_dim, device=device)
        theta_mean = param_cvae.decode(z_B, x_repeated, y_curr)
        theta_curr = theta_mean + torch.randn_like(theta_mean) * infer_noise_p
        if bounds_p is not None:
            theta_curr = torch.clamp(theta_curr, min=bounds_p[0], max=bounds_p[1])
        
        if step >= burn_in:
            y_samples_list.append(y_curr.view(batch_size, num_chains, -1))
            theta_samples_list.append(theta_curr.view(batch_size, num_chains, -1))

    final_y_samples = torch.cat(y_samples_list, dim=1)
    final_theta_samples = torch.cat(theta_samples_list, dim=1)
    
    return final_y_samples, final_theta_samples