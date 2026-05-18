import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
from tqdm import tqdm
from .metrics import (
    calculate_point_metrics, 
    calculate_prediction_interval, 
    calculate_crps, 
    calculate_parametric_nll
)

# =====================================================================
# 1. Pseudo-Gibbs Sampling (추론 엔진)
# =====================================================================
@torch.no_grad()
def pseudo_gibbs_sampling(hidden_cvae, param_cvae, x_sparse, 
                          infer_noise_y, infer_noise_p,
                          num_chains=100, num_steps=50, burn_in=10):
    hidden_cvae.eval()
    param_cvae.eval()
    
    batch_size = x_sparse.size(0)
    device = x_sparse.device
    theta_dim = param_cvae.decoder_net[0][-1].out_features
    
    x_repeated = x_sparse.repeat_interleave(num_chains, dim=0)
    theta_curr = torch.randn(batch_size * num_chains, theta_dim, device=device)

    y_samples_list, theta_samples_list = [], []

    for step in range(num_steps):
        z_A = torch.randn(batch_size * num_chains, hidden_cvae.latent_dim, device=device)
        y_mean = hidden_cvae.decode(z_A, x_repeated, theta_curr)
        y_curr = y_mean + torch.randn_like(y_mean) * infer_noise_y
        y_curr = torch.clamp(y_curr, min=-1.5, max=1.5)
        
        z_B = torch.randn(batch_size * num_chains, param_cvae.latent_dim, device=device)
        theta_mean = param_cvae.decode(z_B, x_repeated, y_curr)
        theta_curr = theta_mean + torch.randn_like(theta_mean) * infer_noise_p
        theta_curr = torch.clamp(theta_curr, min=-1.5, max=1.5)
        
        if step >= burn_in:
            y_samples_list.append(y_curr.view(batch_size, num_chains, -1).cpu().numpy())
            theta_samples_list.append(theta_curr.view(batch_size, num_chains, -1).cpu().numpy())

    final_y = np.concatenate(y_samples_list, axis=1)
    final_theta = np.concatenate(theta_samples_list, axis=1)
    
    theta_history = np.stack(theta_samples_list, axis=2) 
    
    return final_y, final_theta, theta_history


# =====================================================================
# 2. Visualization (기존 플롯 함수 유지 + Robustness 추가)
# =====================================================================
def plot_trajectory_coverage(y_true, y_samples, time_points, save_path=None):
    plt.figure(figsize=(10, 6))
    lower_bound = np.quantile(y_samples, 0.025, axis=0)
    upper_bound = np.quantile(y_samples, 0.975, axis=0)
    median_traj = np.median(y_samples, axis=0)

    plt.fill_between(time_points, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.plot(time_points, median_traj, 'b-', linewidth=2, label='Predicted Median')
    plt.plot(time_points, y_true, 'r--', linewidth=2, label='Ground Truth (Simulation)')
    
    plt.title("Latent Trajectory Inference with Uncertainty", fontsize=14, fontweight='bold')
    plt.xlabel("Time (min)", fontsize=12)
    plt.ylabel("Concentration", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_parameter_posterior(theta_true, theta_samples, param_names=['$S_I$', '$\sigma$'], save_path=None):
    plt.close('all') 
    g = sns.jointplot(x=theta_samples[:, 0], y=theta_samples[:, 1], 
                      kind="kde", fill=True, cmap="Blues", height=8, space=0)
    g.ax_joint.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='Ground Truth')
    g.set_axis_labels(param_names[0], param_names[1], fontsize=14)
    g.fig.suptitle("Parameter Joint Posterior", fontsize=16, fontweight='bold', y=1.02)
    g.fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.95) 
    
    if save_path: g.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()
    
def plot_mcmc_trace_and_acf(theta_history, param_names=['$S_I$', '$\sigma$'], chain_idx=0, save_path=None):
    steps = theta_history.shape[0]
    dims = theta_history.shape[1]
    
    fig, axes = plt.subplots(2, dims, figsize=(12, 8))
    fig.suptitle(f"MCMC Diagnostics (Chain #{chain_idx})", fontsize=16, fontweight='bold')
    
    def autocorr(x, lags):
        mean, var = np.mean(x), np.var(x)
        xp = x - mean
        corr = np.correlate(xp, xp, mode='full')[len(x)-1:] / (var * len(x) + 1e-8)
        return corr[:lags]

    for d in range(dims):
        trace_data = theta_history[:, d]
        axes[0, d].plot(range(steps), trace_data, color='steelblue', alpha=0.8)
        axes[0, d].set_title(f"Trace: {param_names[d]}", fontsize=12)
        axes[0, d].set_xlabel("Steps (After Burn-in)")
        axes[0, d].grid(True, alpha=0.3)
        
        lags = min(40, steps)
        acf_vals = autocorr(trace_data, lags)
        axes[1, d].bar(range(lags), acf_vals, width=0.3, color='darkorange')
        axes[1, d].axhline(0, color='black', linewidth=1)
        axes[1, d].set_title(f"Autocorrelation: {param_names[d]}", fontsize=12)
        axes[1, d].set_xlabel("Lags")
        axes[1, d].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_scatter(y_true_full, y_pred_full, param_names=['$S_I$', '$\sigma$'], save_path=None):
    dims = y_true_full.shape[1]
    y_pred_mean = np.mean(y_pred_full, axis=1) 
    residuals = y_true_full - y_pred_mean
    
    fig, axes = plt.subplots(1, dims, figsize=(6 * dims, 5))
    fig.suptitle("Residual Analysis (Sim-to-Real Bias Check)", fontsize=16, fontweight='bold')
    
    for d in range(dims):
        axes[d].scatter(y_true_full[:, d], residuals[:, d], alpha=0.5, color='purple', edgecolor='k')
        axes[d].axhline(0, color='black', linestyle='--', lw=2)
        axes[d].set_xlabel(f"True {param_names[d]}", fontsize=12)
        axes[d].set_ylabel("Residual (True - Predicted Mean)", fontsize=12)
        axes[d].set_title(f"{param_names[d]} Error Distribution")
        axes[d].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_symmetric_collapse_probabilistic(p_true, theta_samples, p_base, save_path=None):
    idx_si, idx_sigma = 0, 1 
    si_true, sigma_true = p_true[0, idx_si], p_true[0, idx_sigma]
    si_base, sigma_base = p_base[0, idx_si], p_base[0, idx_sigma]
    si_samples = theta_samples[0, :, idx_si]
    sigma_samples = theta_samples[0, :, idx_sigma]
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    sns.kdeplot(x=si_samples, y=sigma_samples, ax=ax1, cmap="Blues", fill=True, alpha=0.6, label='Ours (Posterior)')
    ax1.plot(si_true, sigma_true, 'r*', markersize=15, label='Ground Truth')
    ax1.plot(si_base, sigma_base, 'crimson', marker='X', markersize=12, label='Baseline (Collapsed)')
    
    k_true = si_true * sigma_true
    si_range = np.linspace(min(si_samples), max(si_samples), 100)
    ax1.plot(si_range, k_true / si_range, 'k--', alpha=0.5, label='Non-identifiable Valley ($S_I \\times \\sigma = c$)')
    
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel("$S_I$ [Log]", fontsize=14); ax1.set_ylabel("$\sigma$ [Log]", fontsize=14)
    ax1.set_title("Resolution of Symmetric Collapse via Probabilistic Inference", fontsize=16, fontweight='bold')
    ax1.legend()
    
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_ogtt_trajectories_probabilistic(system_ode_func, y0, p_true, theta_samples, p_base, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    t_span = (0, 120)
    t_eval = np.linspace(0, 120, 200)
    
    sol_true = solve_ivp(system_ode_func, t_span, y0, args=tuple(p_true), t_eval=t_eval)
    ax.plot(sol_true.t, sol_true.y[1], 'k-', lw=3, label='True Insulin (Hidden)')
    
    sol_base = solve_ivp(system_ode_func, t_span, y0, args=tuple(p_base), t_eval=t_eval)
    ax.plot(sol_base.t, sol_base.y[1], 'crimson', linestyle='--', lw=2, label='Baseline (Failed)')
    
    num_plot_samples = 50
    indices = np.random.choice(theta_samples.shape[0], num_plot_samples, replace=False)
    
    for i, idx in enumerate(indices):
        param_sample = theta_samples[idx]
        sol_sample = solve_ivp(system_ode_func, t_span, y0, args=tuple(param_sample), t_eval=t_eval)
        label = 'Ours (Posterior Samples)' if i == 0 else ""
        ax.plot(sol_sample.t, sol_sample.y[1], 'steelblue', alpha=0.1, lw=1.5, label=label)

    ax.set_title("Insulin Dynamics (Hidden State Distribution)", fontsize=15, fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

# =====================================================================
# 3. Robustness Evaluation Module (신규 추가)
# =====================================================================
def evaluate_robustness_probabilistic(hidden_cvae, param_cvae, baseline_model, test_loader, config, normalizer, target_dir):
    print("\n\033[1;36m=== Running Robustness Stress Test with Variance Analysis ===\033[0m")
    
    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0] 
    results = []
    
    # 1. 분산 및 표준편차 측정을 위해 모든 X_obs를 수집
    all_x = []
    for x_batch, _, _ in test_loader:
        all_x.append(x_batch.to(config.DEVICE))
    x_norm_full = torch.cat(all_x, dim=0)
    x_std = x_norm_full.std(dim=0, keepdim=True)
    
    # OOM 방지를 위한 데이터 캐싱
    batches = [(x.to(config.DEVICE), p.cpu().numpy()) for x, _, p in test_loader]

    def safe_pearsonr(x, y, eps=1e-12):
        var_x, var_y = np.var(x), np.var(y)
        if var_x < eps or var_y < eps:
            return 0.0
        r, _ = pearsonr(x, y)
        return r if np.isfinite(r) else 0.0

    for nl_val in noise_levels:
        nl = nl_val / 100.0
        print(f"[Testing] Noise Level: {nl_val:.1f}% ...", end="\r")
        
        torch.manual_seed(config.SEED)
        
        all_p_true_phys = []
        all_p_ours_phys = []
        all_p_base_phys = []
        
        # Batch 처리
        for x_batch, p_batch in batches:
            noise = torch.randn_like(x_batch) * (x_std * nl)
            x_noisy = x_batch + noise
            
            # Ground Truth 역정규화
            p_true_phys = normalizer.denormalize_params(torch.tensor(p_batch).to(config.DEVICE)).cpu().numpy()
            all_p_true_phys.append(p_true_phys)
            
            # Ours: Probabilistic Inference
            _, theta_samples_norm, _ = pseudo_gibbs_sampling(
                hidden_cvae, param_cvae, x_noisy, 
                infer_noise_y=config.INFER_NOISE_Y, infer_noise_p=config.INFER_NOISE_P,
                num_chains=config.INFERENCE_CHAINS, num_steps=config.INFERENCE_STEPS, burn_in=config.INFERENCE_BURN_IN
            )
            theta_mean_norm = torch.tensor(np.mean(theta_samples_norm, axis=1)).to(config.DEVICE)
            all_p_ours_phys.append(normalizer.denormalize_params(theta_mean_norm).cpu().numpy())
            
            # Baseline: Deterministic Inference
            if baseline_model:
                with torch.no_grad():
                    theta_base_norm = baseline_model(x_noisy)
                    all_p_base_phys.append(normalizer.denormalize_params(theta_base_norm).cpu().numpy())
            else:
                all_p_base_phys.append(np.zeros_like(p_true_phys))

        p_true_phys_full = np.concatenate(all_p_true_phys, axis=0)
        p_ours_phys_full = np.concatenate(all_p_ours_phys, axis=0)
        p_base_phys_full = np.concatenate(all_p_base_phys, axis=0)

        # 3. 메트릭 계산
        r_si_base = safe_pearsonr(p_true_phys_full[:, 0], p_base_phys_full[:, 0])
        r_sigma_base = safe_pearsonr(p_true_phys_full[:, 1], p_base_phys_full[:, 1])
        r_si_ours = safe_pearsonr(p_true_phys_full[:, 0], p_ours_phys_full[:, 0])
        r_sigma_ours = safe_pearsonr(p_true_phys_full[:, 1], p_ours_phys_full[:, 1])

        results.append({
            'noise_level': nl_val,
            'Base_SI_Pearson': r_si_base, 'Base_Sigma_Pearson': r_sigma_base,
            'Ours_SI_Pearson': r_si_ours, 'Ours_Sigma_Pearson': r_sigma_ours,
            'Var_SI_GT': np.var(p_true_phys_full[:, 0]), 'Var_Sigma_GT': np.var(p_true_phys_full[:, 1]),
            'Var_SI_Base': np.var(p_base_phys_full[:, 0]), 'Var_Sigma_Base': np.var(p_base_phys_full[:, 1]),
            'Var_SI_Ours': np.var(p_ours_phys_full[:, 0]), 'Var_Sigma_Ours': np.var(p_ours_phys_full[:, 1])
        })

    print("\nRobustness evaluation completed.")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(target_dir, "robustness_metrics_with_variance.csv"), index=False)

    # --- 시각화: Pearson R과 Variance 동시 비교 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, (param, label) in enumerate(zip(['SI', 'Sigma'], ['$S_I$', '$\sigma$'])):
        ax = axes[0, i]
        ax.plot(df['noise_level'], df[f'Base_{param}_Pearson'], 'ro--', label=f'Baseline ({label})')
        ax.plot(df['noise_level'], df[f'Ours_{param}_Pearson'], 'bo-', label=f'Ours ({label})')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{label} Prediction Robustness (Pearson $r$)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Noise Level (%)"); ax.set_ylabel("Correlation ($r$)")
        ax.grid(True, alpha=0.3); ax.legend()

    for i, (param, label) in enumerate(zip(['SI', 'Sigma'], ['$S_I$', '$\sigma$'])):
        ax = axes[1, i]
        gt_var = df[f'Var_{param}_GT'].iloc[0]
        base_var = df[f'Var_{param}_Base']
        ours_var = df[f'Var_{param}_Ours']

        limit = 0.2
        base_var_clipped = np.clip(base_var, None, limit + 0.1)
        ours_var_clipped = np.clip(ours_var, None, limit + 0.1)

        ax.axhline(y=gt_var, color='grey', linestyle='-', alpha=0.5, label='Ground Truth Var')
        ax.plot(df['noise_level'], base_var_clipped, 'ro--', label='Baseline Var')
        ax.plot(df['noise_level'], ours_var_clipped, 'bo-', label='Ours Var')

        ax.axhspan(limit, limit + 0.05, color='red', alpha=0.1)
        ax.text(df['noise_level'].mean(), limit + 0.025, "Divergence Zone", 
                color='red', fontsize=10, ha='center', fontweight='bold')

        ax.set_ylim(0.0, limit + 0.05)
        ax.set_title(f"{label} Variance: Stability vs. Explosion", fontsize=14, fontweight='bold')
        ax.set_xlabel("Noise Level (%)"); ax.set_ylabel("Variance")
        ax.grid(True, alpha=0.2); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, "7_robustness_and_variance_analysis.png"), dpi=300)
    plt.close()


# =====================================================================
# 4. Main Evaluation Pipeline
# =====================================================================
def run_full_analysis(hidden_cvae, param_cvae, baseline_model, test_loader, config, system, normalizer):
    print("\n\033[1;36m=== Running Probabilistic Analysis & Evaluation ===\033[0m")
    
    save_dir = os.path.join(config.RESULTS_DIR, config.EXPERIMENT_NAME, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    hidden_cvae.eval()
    param_cvae.eval()
    if baseline_model: baseline_model.eval()

    all_y_true, all_theta_true = [], []
    all_y_samples, all_theta_samples = [], []
    all_theta_base = []
    
    print("Evaluating entire test set and generating samples...")
    all_theta_history = [] 
    
    for x_batch, y_batch, theta_batch in tqdm(test_loader, desc="Gibbs Sampling Progress"):
        x_batch = x_batch.to(config.DEVICE)
        
        y_samples_norm, theta_samples_norm, theta_hist_norm = pseudo_gibbs_sampling(
            hidden_cvae, param_cvae, x_batch, 
            infer_noise_y=config.INFER_NOISE_Y,
            infer_noise_p=config.INFER_NOISE_P,
            num_chains=config.INFERENCE_CHAINS, 
            num_steps=config.INFERENCE_STEPS, 
            burn_in=config.INFERENCE_BURN_IN
        )
        
        if baseline_model:
            with torch.no_grad():
                theta_base_norm = baseline_model(x_batch)
                theta_base_phys = normalizer.denormalize_params(theta_base_norm).cpu().numpy()
        else:
            theta_base_phys = np.zeros_like(theta_batch.cpu().numpy())
            
        y_true_phys = normalizer.denormalize_inputs(y_batch, variable_type='hidden').cpu().numpy()
        y_samples_phys = np.zeros_like(y_samples_norm)
        for i in range(y_samples_norm.shape[1]):
            y_samples_phys[:, i, :] = normalizer.denormalize_inputs(
                torch.tensor(y_samples_norm[:, i, :]).to(config.DEVICE), variable_type='hidden'
            ).cpu().numpy()
            
        theta_true_phys = normalizer.denormalize_params(theta_batch).cpu().numpy()
        theta_samples_phys = np.zeros_like(theta_samples_norm)
        for i in range(theta_samples_norm.shape[1]):
            theta_samples_phys[:, i, :] = normalizer.denormalize_params(
                torch.tensor(theta_samples_norm[:, i, :]).to(config.DEVICE)
            ).cpu().numpy()
            
        if len(all_theta_history) == 0:
            hist_phys = np.zeros_like(theta_hist_norm)
            for c in range(theta_hist_norm.shape[1]): 
                for s in range(theta_hist_norm.shape[2]): 
                    hist_phys[:, c, s, :] = normalizer.denormalize_params(
                        torch.tensor(theta_hist_norm[:, c, s, :]).to(config.DEVICE)
                    ).cpu().numpy()
            all_theta_history.append(hist_phys)
                    
        all_y_true.append(y_true_phys)
        all_theta_true.append(theta_true_phys)
        all_y_samples.append(y_samples_phys)
        all_theta_samples.append(theta_samples_phys)
        all_theta_base.append(theta_base_phys)

    y_true_full = np.concatenate(all_y_true, axis=0)
    theta_true_full = np.concatenate(all_theta_true, axis=0)
    y_samples_full = np.concatenate(all_y_samples, axis=0)
    theta_samples_full = np.concatenate(all_theta_samples, axis=0)
    theta_base_full = np.concatenate(all_theta_base, axis=0)

    # -------------------------------------------------------------
    # [수정] 1. 평균 지표 대신 파라미터별(SI, Sigma)로 분리 계산 및 출력
    # -------------------------------------------------------------
    picp, mpiw = calculate_prediction_interval(theta_true_full, theta_samples_full)
    crps = calculate_crps(theta_true_full, theta_samples_full)
    
    # 확실한 독립 계산을 위해 scikit-learn과 scipy.stats 직접 호출
    theta_pred_mean = np.mean(theta_samples_full, axis=1)
    rmse_si = np.sqrt(mean_squared_error(theta_true_full[:, 0], theta_pred_mean[:, 0]))
    rmse_sigma = np.sqrt(mean_squared_error(theta_true_full[:, 1], theta_pred_mean[:, 1]))
    pearson_si, _ = pearsonr(theta_true_full[:, 0], theta_pred_mean[:, 0])
    pearson_sigma, _ = pearsonr(theta_true_full[:, 1], theta_pred_mean[:, 1])
    
    print("\n\033[1;32m[Parameter Estimation Metrics (Test Set)]\033[0m")
    print(f"PICP (95% Coverage) : {np.mean(picp)*100:.2f}%")
    print(f"MPIW (Width)        : {np.mean(mpiw):.4f}")
    print(f"CRPS                : {np.mean(crps):.4f}")
    
    print(f"\n[\033[1;33m$S_I$ Metrics\033[0m]")
    print(f"  RMSE      : {rmse_si:.4f}")
    print(f"  Pearson r : {pearson_si:.4f}")
    
    print(f"\n[\033[1;33m$\sigma$ Metrics\033[0m]")
    print(f"  RMSE      : {rmse_sigma:.4f}")
    print(f"  Pearson r : {pearson_sigma:.4f}")

    # -------------------------------------------------------------
    # [추가] 2. Robustness Test (노이즈 주입 평가) 호출
    # -------------------------------------------------------------
    evaluate_robustness_probabilistic(
        hidden_cvae, param_cvae, baseline_model, 
        test_loader, config, normalizer, save_dir
    )

    # -------------------------------------------------------------
    # 시각화 (기존)
    # -------------------------------------------------------------
    print("\nGenerating Visual Proofs...")
    sample_idx = 0
    time_points = np.linspace(0, 120, y_true_full.shape[1])
    
    plot_trajectory_coverage(
        y_true=y_true_full[sample_idx], y_samples=y_samples_full[sample_idx], 
        time_points=time_points, save_path=os.path.join(save_dir, "1_trajectory_coverage.png")
    )
    
    plot_parameter_posterior(
        theta_true_full[sample_idx], theta_samples_full[sample_idx], 
        save_path=os.path.join(save_dir, "2_parameter_posterior.png")
    )
    
    if baseline_model:
        plot_symmetric_collapse_probabilistic(
            theta_true_full, theta_samples_full, theta_base_full, 
            save_path=os.path.join(save_dir, "3_symmetric_collapse.png")
        )
        
        y0 = [val[0] if isinstance(val, list) else val for val in system.initial_conditions]
        def ode_func(t, y, *params): return system.ode_func(t, y, list(params))
        
        plot_ogtt_trajectories_probabilistic(
            ode_func, y0, theta_true_full[sample_idx], theta_samples_full[sample_idx], theta_base_full[sample_idx], 
            save_path=os.path.join(save_dir, "4_ode_trajectories.png")
        )
    
    target_history = all_theta_history[0][sample_idx, 0, :, :] 
    plot_mcmc_trace_and_acf(
        target_history, 
        save_path=os.path.join(save_dir, "5_mcmc_diagnostics.png")
    )
    
    plot_residual_scatter(
        theta_true_full, theta_samples_full,
        save_path=os.path.join(save_dir, "6_residual_scatter.png")
    )
    
    print(f"Analysis complete. All plots saved to: {save_dir}")