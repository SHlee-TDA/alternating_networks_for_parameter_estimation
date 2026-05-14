import torch
import numpy as np

@torch.no_grad()
def pseudo_gibbs_sampling(
    hidden_cvae, param_cvae, x_observed, 
    num_chains=100, num_steps=50, burn_in=10, 
    init_theta=None
):
    hidden_cvae.eval()
    param_cvae.eval()
    
    batch_size = x_observed.size(0)
    device = x_observed.device
    theta_dim = param_cvae.decoder_net[-1].out_features
    
    x_repeated = x_observed.repeat_interleave(num_chains, dim=0)
    
    if init_theta is None:
        theta_curr = torch.randn(batch_size * num_chains, theta_dim, device=device)
    else:
        theta_curr = init_theta.repeat_interleave(num_chains, dim=0)

    # 샘플을 저장할 리스트
    y_samples_list = []
    theta_samples_list = []

    # 3. 핑퐁 루프 (The Alternating Process)
    for step in range(num_steps):
        z_A = torch.randn(batch_size * num_chains, hidden_cvae.latent_dim, device=device)
        y_curr = hidden_cvae.decode(z_A, x_repeated, theta_curr)
        
        z_B = torch.randn(batch_size * num_chains, param_cvae.latent_dim, device=device)
        theta_curr = param_cvae.decode(z_B, x_repeated, y_curr)
        
        if step >= burn_in:
            y_samples_list.append(y_curr.view(batch_size, num_chains, -1))
            theta_samples_list.append(theta_curr.view(batch_size, num_chains, -1))

    final_y_samples = torch.cat(y_samples_list, dim=1)
    final_theta_samples = torch.cat(theta_samples_list, dim=1)
    
    return final_y_samples, final_theta_samples