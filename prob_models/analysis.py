import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from prob_models.infer import pseudo_gibbs_sampling, single_cvae_sampling
from prob_models.metrics import (
    calculate_prediction_interval, 
    calculate_crps, 
    calculate_kde_nll
)

# =====================================================================
# 1. Visualization
# =====================================================================
def plot_parameter_posterior(theta_true, theta_samples, param_names=['$S_I$', '$\sigma$'], save_path=None):
    plt.close('all') 
    g = sns.jointplot(x=theta_samples[:, 0], y=theta_samples[:, 1], 
                      kind="kde", fill=True, cmap="Blues", height=8, space=0)
    g.ax_joint.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='Ground Truth')
    g.set_axis_labels(param_names[0], param_names[1], fontsize=14)
    
    # [수정됨] g.fig 대신 g.figure 사용 (Deprecation Warning 해결)
    g.figure.suptitle("Parameter Joint Posterior", fontsize=16, fontweight='bold', y=1.02)
    g.figure.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.95) 
    
    if save_path: g.figure.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()
    
def plot_mcmc_trace_and_acf(theta_history, param_names=['$S_I$', '$\sigma$'], chain_idx=0, save_path=None):
    steps, dims = theta_history.shape[0], theta_history.shape[1]
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
        axes[0, d].grid(True, alpha=0.3)
        
        lags = min(40, steps)
        axes[1, d].bar(range(lags), autocorr(trace_data, lags), width=0.3, color='darkorange')
        axes[1, d].axhline(0, color='black', linewidth=1)
        axes[1, d].set_title(f"Autocorrelation: {param_names[d]}", fontsize=12)
        axes[1, d].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 2. Robustness Evaluation Module
# =====================================================================
def evaluate_robustness_probabilistic(state_cvae, param_cvae, test_loader, config, normalizer, target_dir, is_baseline):
    print("\n\033[1;36m=== Running Robustness Stress Test ===\033[0m")
    
    # [수정됨] 생물학적 현실성을 반영하여 노이즈 상한을 20%로 제한
    noise_levels = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0] 
    results = []
    
    all_x = [x.to(config.DEVICE) for x, _, _ in test_loader]
    x_std = torch.cat(all_x, dim=0).std(dim=0, keepdim=True)
    batches = [(x.to(config.DEVICE), p.to(config.DEVICE)) for x, _, p in test_loader]

    def safe_pearsonr(x, y):
        if np.var(x) < 1e-12 or np.var(y) < 1e-12: return 0.0
        r, _ = pearsonr(x, y)
        return r if np.isfinite(r) else 0.0

    for nl_val in noise_levels:
        nl = nl_val / 100.0
        print(f"[Testing] Noise Level: {nl_val:.1f}% ...", end="\r")
        torch.manual_seed(config.SEED)
        
        all_p_true, all_p_pred = [], []
        
        for x_batch, p_batch in batches:
            x_noisy = x_batch + torch.randn_like(x_batch) * (x_std * nl)
            p_true_phys = normalizer.denormalize_params(p_batch).cpu().numpy()
            all_p_true.append(p_true_phys)
            
            if is_baseline:
                theta_samples_norm = single_cvae_sampling(param_cvae, x_noisy, num_samples=config.INFERENCE_CHAINS)
            else:
                _, theta_samples_norm, _ = pseudo_gibbs_sampling(
                    state_cvae, param_cvae, x_noisy, 
                    num_chains=config.INFERENCE_CHAINS, num_steps=config.INFERENCE_STEPS, burn_in=config.INFERENCE_BURN_IN
                )
            
            theta_mean_norm = theta_samples_norm.mean(dim=1) 
            all_p_pred.append(normalizer.denormalize_params(theta_mean_norm).cpu().numpy())

        p_true_full = np.concatenate(all_p_true, axis=0)
        p_pred_full = np.concatenate(all_p_pred, axis=0)

        results.append({
            'noise_level': nl_val,
            'SI_Pearson': safe_pearsonr(p_true_full[:, 0], p_pred_full[:, 0]),
            'Sigma_Pearson': safe_pearsonr(p_true_full[:, 1], p_pred_full[:, 1]),
            'Var_SI_Pred': np.var(p_pred_full[:, 0]),
            'Var_Sigma_Pred': np.var(p_pred_full[:, 1])
        })

    df = pd.DataFrame(results)
    csv_name = "robustness_baseline.csv" if is_baseline else "robustness_ours.csv"
    df.to_csv(os.path.join(target_dir, csv_name), index=False)
    print(f"\nRobustness data saved to {csv_name}")

# =====================================================================
# 3. Main Evaluation Pipeline 
# =====================================================================
# [수정됨] 사용하지 않는 파라미터(logger, system, history, real_test_loader, p_init) 모두 제거
def run_prob_evaluation_phase(
    run_config, state_estimator, param_estimator, 
    test_l, normalizer, device
):
    print("\n\033[1;36m=== Running Probabilistic Analysis & Evaluation ===\033[0m")
    
    is_baseline = getattr(run_config, 'RUN_BASELINE', False)
    save_dir = os.path.join(run_config.RESULTS_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    all_theta_true, all_theta_samples = [], []
    all_theta_history = [] 
    
    print("Evaluating test set and generating samples...")
    for x_batch, _, theta_batch in tqdm(test_l, desc="Inference Progress"):
        x_batch = x_batch.to(device)
        theta_batch = theta_batch.to(device)
        
        if is_baseline:
            theta_samples_norm = single_cvae_sampling(param_estimator, x_batch, num_samples=run_config.INFERENCE_CHAINS)
            theta_hist_norm = None
        else:
            _, theta_samples_norm, theta_hist_norm = pseudo_gibbs_sampling(
                state_estimator, param_estimator, x_batch, 
                num_chains=run_config.INFERENCE_CHAINS, num_steps=run_config.INFERENCE_STEPS, burn_in=run_config.INFERENCE_BURN_IN
            )
            
        B, S, D = theta_samples_norm.shape
        theta_samples_flat = theta_samples_norm.reshape(-1, D)
        theta_samples_phys = normalizer.denormalize_params(theta_samples_flat).reshape(B, S, D).cpu().numpy()
        theta_true_phys = normalizer.denormalize_params(theta_batch).cpu().numpy()
        
        all_theta_true.append(theta_true_phys)
        all_theta_samples.append(theta_samples_phys)
        
        if theta_hist_norm is not None and len(all_theta_history) == 0:
            B_h, C_h, S_h, D_h = theta_hist_norm.shape
            hist_flat = theta_hist_norm.reshape(-1, D_h).to(device)
            hist_phys = normalizer.denormalize_params(hist_flat).reshape(B_h, C_h, S_h, D_h).cpu().numpy()
            all_theta_history.append(hist_phys)

    theta_true_full = np.concatenate(all_theta_true, axis=0)
    theta_samples_full = np.concatenate(all_theta_samples, axis=0)

    # Metrics Calculation
    picp, mpiw = calculate_prediction_interval(theta_true_full, theta_samples_full)
    crps = calculate_crps(theta_true_full, theta_samples_full)
    nll = calculate_kde_nll(theta_true_full, theta_samples_full)
    
    theta_pred_mean = np.mean(theta_samples_full, axis=1)
    rmse_si = np.sqrt(mean_squared_error(theta_true_full[:, 0], theta_pred_mean[:, 0]))
    rmse_sigma = np.sqrt(mean_squared_error(theta_true_full[:, 1], theta_pred_mean[:, 1]))
    pearson_si, _ = pearsonr(theta_true_full[:, 0], theta_pred_mean[:, 0])
    pearson_sigma, _ = pearsonr(theta_true_full[:, 1], theta_pred_mean[:, 1])
    
    print("\n\033[1;32m[Parameter Estimation Metrics]\033[0m")
    print(f"PICP (95% Cov) : {np.mean(picp)*100:.2f}%")
    print(f"MPIW (Width)   : {np.mean(mpiw):.4f}")
    print(f"CRPS           : {np.mean(crps):.4f}")
    print(f"NLL            : {np.mean(nll):.4f}")
    print(f"[$S_I$]   RMSE: {rmse_si:.4f} | r: {pearson_si:.4f}")
    print(f"[$\sigma$] RMSE: {rmse_sigma:.4f} | r: {pearson_sigma:.4f}")

    print("\nGenerating Visualizations...")
    sample_idx = 0
    plot_parameter_posterior(
        theta_true_full[sample_idx], theta_samples_full[sample_idx], 
        save_path=os.path.join(save_dir, "parameter_posterior.png")
    )
    
    if not is_baseline and len(all_theta_history) > 0:
        target_history = all_theta_history[0][sample_idx, 0, :, :] 
        plot_mcmc_trace_and_acf(target_history, save_path=os.path.join(save_dir, "mcmc_diagnostics.png"))

    evaluate_robustness_probabilistic(
        state_estimator, param_estimator, test_l, run_config, normalizer, save_dir, is_baseline
    )

    print(f"Analysis complete. Outputs saved to: {save_dir}")

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