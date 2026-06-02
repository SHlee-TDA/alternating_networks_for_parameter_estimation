import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import sys
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prob_models.config import ProbConfig
from prob_models.models import SingleCVAE, HiddenStateCVAE, ParameterCVAE
from src.data_loader import DataGenerator, setup_dataloaders
from tools.exp_tools import get_system_class
from tools.interactive_file_selector import interactive_file_selector

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
# 4. Standalone Execution Pipeline (맨 아래에 추가)
# =====================================================================
def main():
    try:
        print("\n\033[1;33m=== [Analysis 평가를 위한 파일 선택] ===\033[0m")
        base_search_dir = "./results"
        
        # 1. 모델 및 설정 파일 인터랙티브 선택
        rel_config_path = interactive_file_selector("[1/4] 실험 설정 파일 (config.json) 선택:", start_dir=base_search_dir)
        config_path = os.path.join(base_search_dir, rel_config_path)
        
        rel_prob_config_path = interactive_file_selector("[2/4] 확률 모델 설정 파일 (prob_config.json) 선택:", start_dir=base_search_dir)
        prob_config_path = os.path.join(base_search_dir, rel_prob_config_path)
        
        rel_baseline_path = interactive_file_selector("[3/4] Single CVAE 가중치 선택:", start_dir=base_search_dir)
        baseline_path = os.path.join(base_search_dir, rel_baseline_path)
        
        rel_Hnet_path = interactive_file_selector("[4/4] Hidden CVAE 가중치 선택 (Param CVAE 자동로드):", start_dir=base_search_dir)
        Hnet_path = os.path.join(base_search_dir, rel_Hnet_path)
        Pnet_path = Hnet_path.replace('hidden_cvae.pth', 'param_cvae.pth').replace('Hnet.pth', 'Pnet.pth')

        # 2. Config 병합 로드
        config = ProbConfig()
        print(f"\n[Info] Loading Config from: {config_path}")
        with open(config_path, 'r') as f:
            for k, v in json.load(f).items():
                if not k.startswith('__'): setattr(config, k, v)
                
        print(f"[Info] Loading Prob Config from: {prob_config_path}")
        with open(prob_config_path, 'r') as f:
            for k, v in json.load(f).items():
                if not k.startswith('__'): setattr(config, k, v)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.DEVICE = device

        # 3. System 및 Test DataLoader 완벽 로드
        SystemClass = get_system_class(config.SYSTEM_NAME)
        system = SystemClass()

        print(f"[Info] Loading Data Cache...")
        generator = DataGenerator(system, config)
        sim_data_tuple = generator.generate_data()

        print(f"[Info] Setting up DataLoaders (Preserving exact Normalizer)...")
        loaders = setup_dataloaders(vars(config), sim_data_tuple, system, config)
        _, _, test_l, _, _, normalizer = loaders

        # 4. 차원 정보 추출 및 모델 초기화
        sample_x, sample_y, sample_p = next(iter(test_l))
        x_dim, y_dim, theta_dim = sample_x.shape[1], sample_y.shape[1], sample_p.shape[1]

        latent_dim_h = getattr(config, 'LATENT_DIM_HIDDEN', 4)
        latent_dim_p = getattr(config, 'LATENT_DIM_PARAM', 2)
        latent_dim_base = getattr(config, 'LATENT_DIM_BASELINE', 2)
        hidden_dims = getattr(config, 'HIDDEN_DIMS', [64, 64, 64, 64])

        print(f"[Info] Initializing Models...")
        baseline_cvae = SingleCVAE(x_dim, theta_dim, latent_dim=latent_dim_base, hidden_dims=hidden_dims).to(device)
        Hnet = HiddenStateCVAE(x_dim, theta_dim, y_dim, latent_dim=latent_dim_h, hidden_dims=hidden_dims).to(device)
        Pnet = ParameterCVAE(x_dim, y_dim, theta_dim, latent_dim=latent_dim_p, hidden_dims=hidden_dims).to(device)
        Pnet.theta_dim = theta_dim # 속성 주입 (안전장치)

        def load_weight_safe(model, path, possible_keys):
            ckpt = torch.load(path, map_location=device)
            if isinstance(ckpt, dict):
                for key in possible_keys:
                    if key in ckpt:
                        model.load_state_dict(ckpt[key])
                        return
                for k, v in ckpt.items():
                    if 'state_dict' in k and 'optimizer' not in k:
                        model.load_state_dict(v)
                        return
            model.load_state_dict(ckpt)

        print(f"[Info] Loading Checkpoints...")
        load_weight_safe(baseline_cvae, baseline_path, ['baseline_cvae_state_dict'])
        load_weight_safe(Hnet, Hnet_path, ['hidden_cvae_state_dict', 'Hnet_state_dict'])
        load_weight_safe(Pnet, Pnet_path, ['param_cvae_state_dict', 'Pnet_state_dict'])

        baseline_cvae.eval(); Hnet.eval(); Pnet.eval()
        
        # 5. Core Evaluation 실행 (Ours)
        print("\n" + "="*50)
        print("  Evaluating Iterative CVAEs (Proposed Method)")
        print("="*50)
        config.RUN_BASELINE = False
        run_prob_evaluation_phase(
            run_config=config,
            state_estimator=Hnet,
            param_estimator=Pnet,
            test_l=test_l,
            normalizer=normalizer,
            device=device
        )
        
        # 필요시 주석 해제하여 Baseline도 바로 이어서 평가 가능
        """
        print("\n" + "="*50)
        print("  Evaluating Single CVAE (Baseline Method)")
        print("="*50)
        config.RUN_BASELINE = True
        run_prob_evaluation_phase(
            run_config=config,
            state_estimator=None,
            param_estimator=baseline_cvae,
            test_l=test_l,
            normalizer=normalizer,
            device=device
        )
        """

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()