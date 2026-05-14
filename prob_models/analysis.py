import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

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
def pseudo_gibbs_sampling(hidden_cvae, param_cvae, x_sparse, num_chains=100, num_steps=50, burn_in=10):
    """실제 환자 데이터(X_sparse)를 바탕으로 Y와 theta의 사후 분포를 추출합니다."""
    hidden_cvae.eval()
    param_cvae.eval()
    
    batch_size = x_sparse.size(0)
    device = x_sparse.device
    theta_dim = param_cvae.decoder_net[-1].out_features
    
    # 병렬 체인을 위해 x_sparse 복제
    x_repeated = x_sparse.repeat_interleave(num_chains, dim=0)
    
    # 초기화: Prior N(0, I)에서 시작
    theta_curr = torch.randn(batch_size * num_chains, theta_dim, device=device)

    y_samples_list, theta_samples_list = [], []

    for step in range(num_steps):
        # Step A: 파라미터를 바탕으로 궤적(Y) 샘플링
        z_A = torch.randn(batch_size * num_chains, hidden_cvae.latent_dim, device=device)
        y_curr = hidden_cvae.decode(z_A, x_repeated, theta_curr)
        
        # Step B: 궤적(Y)을 바탕으로 파라미터(theta) 샘플링
        z_B = torch.randn(batch_size * num_chains, param_cvae.latent_dim, device=device)
        theta_curr = param_cvae.decode(z_B, x_repeated, y_curr)
        
        if step >= burn_in:
            y_samples_list.append(y_curr.view(batch_size, num_chains, -1).cpu().numpy())
            theta_samples_list.append(theta_curr.view(batch_size, num_chains, -1).cpu().numpy())

    # [batch, samples, dim] 형태로 결합
    final_y = np.concatenate(y_samples_list, axis=1)
    final_theta = np.concatenate(theta_samples_list, axis=1)
    
    return final_y, final_theta


# =====================================================================
# 2. Visualization (시각화 함수)
# =====================================================================
def plot_trajectory_coverage(x_sparse, y_true, y_samples, time_points, save_path=None):
    """인슐린 궤적의 95% 신뢰 구간과 정답을 함께 시각화합니다."""
    plt.figure(figsize=(10, 6))
    
    # 95% 신뢰 구간 (2.5% ~ 97.5%) 및 중앙값 계산
    lower_bound = np.quantile(y_samples, 0.025, axis=0)
    upper_bound = np.quantile(y_samples, 0.975, axis=0)
    median_traj = np.median(y_samples, axis=0)

    # 신뢰 구간 밴드 그리기
    plt.fill_between(time_points, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # 예측 중앙값과 실제 정답 그리기
    plt.plot(time_points, median_traj, 'b-', linewidth=2, label='Predicted Median')
    plt.plot(time_points, y_true, 'r--', linewidth=2, label='Ground Truth (Simulation)')
    
    # 관측된 희소 데이터 포인트 (X_sparse) 표시 (예시: 특정 시간에 측정되었다고 가정)
    # 실제 프로젝트의 time index에 맞게 수정 필요
    obs_times = np.linspace(time_points[0], time_points[-1], len(x_sparse))
    plt.scatter(obs_times, x_sparse, color='black', s=100, zorder=5, marker='X', label='Observed Blood Glucose ($X_{sparse}$)')

    plt.title("Latent Trajectory Inference with Uncertainty", fontsize=14, fontweight='bold')
    plt.xlabel("Time (min)", fontsize=12)
    plt.ylabel("Concentration", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_parameter_posterior(theta_true, theta_samples, param_names=['$S_I$', '$\sigma$'], save_path=None):
    """비식별성 계곡을 보여주는 파라미터의 결합 사후 분포(Joint Posterior) 시각화"""
    # Seaborn을 이용한 2D 밀도(Density) 플롯
    g = sns.jointplot(x=theta_samples[:, 0], y=theta_samples[:, 1], 
                      kind="kde", fill=True, cmap="Blues", 
                      height=8, space=0)
    
    # 정답(Ground Truth) 지점을 빨간색 별표로 표시
    g.ax_joint.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='Ground Truth')
    
    g.set_axis_labels(param_names[0], param_names[1], fontsize=14)
    g.fig.suptitle("Joint Posterior Distribution of Parameters", y=1.02, fontsize=16, fontweight='bold')
    g.ax_joint.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    
def plot_symmetric_collapse_probabilistic(p_true, theta_samples, p_base, save_path=None):
    """
    p_true: [batch, 2] (정답)
    theta_samples: [batch, num_samples, 2] (우리의 사후 분포 샘플들)
    p_base: [batch, 2] (Baseline의 결정론적 예측치)
    """
    idx_si, idx_sigma = 0, 1 
    
    # 예시로 첫 번째 환자(batch index 0)의 분포만 시각화
    si_true, sigma_true = p_true[0, idx_si], p_true[0, idx_sigma]
    si_base, sigma_base = p_base[0, idx_si], p_base[0, idx_sigma]
    
    # 우리의 샘플들 (1,000개)
    si_samples = theta_samples[0, :, idx_si]
    sigma_samples = theta_samples[0, :, idx_sigma]
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # 1. 우리의 확률 분포 (KDE 구름)
    sns.kdeplot(x=si_samples, y=sigma_samples, ax=ax1, cmap="Blues", fill=True, alpha=0.6, label='Ours (Posterior)')
    
    # 2. 정답 (Ground Truth)
    ax1.plot(si_true, sigma_true, 'r*', markersize=15, label='Ground Truth')
    
    # 3. Baseline의 오답 (Point)
    ax1.plot(si_base, sigma_base, 'crimson', marker='X', markersize=12, label='Baseline (Collapsed)')
    
    # 비식별성 계곡 라인 (S_I * sigma = K_true)
    k_true = si_true * sigma_true
    si_range = np.linspace(min(si_samples), max(si_samples), 100)
    ax1.plot(si_range, k_true / si_range, 'k--', alpha=0.5, label='Non-identifiable Valley ($S_I \\times \\sigma = c$)')
    
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel("$S_I$ [Log]", fontsize=14); ax1.set_ylabel("$\sigma$ [Log]", fontsize=14)
    ax1.set_title("Resolution of Symmetric Collapse via Probabilistic Inference", fontsize=16, fontweight='bold')
    ax1.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    
def plot_ogtt_trajectories_probabilistic(system_ode_func, y0, p_true, theta_samples, p_base, save_path=None):
    """
    샘플링된 파라미터들로 ODE를 수십 번 풀어 인슐린 궤적의 '불확실성 대역'을 그립니다.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    t_span = (0, 120)
    t_eval = np.linspace(0, 120, 200)
    
    # 1. 정답 궤적 (True)
    sol_true = solve_ivp(system_ode_func, t_span, y0, args=tuple(p_true), t_eval=t_eval)
    ax.plot(sol_true.t, sol_true.y[1], 'k-', lw=3, label='True Insulin (Hidden)')
    
    # 2. Baseline 궤적
    sol_base = solve_ivp(system_ode_func, t_span, y0, args=tuple(p_base), t_eval=t_eval)
    ax.plot(sol_base.t, sol_base.y[1], 'crimson', linestyle='--', lw=2, label='Baseline (Failed)')
    
    # 3. 우리의 분포 예측 (샘플 50개 정도만 무작위로 뽑아서 겹쳐 그리기)
    num_plot_samples = 50
    indices = np.random.choice(theta_samples.shape[0], num_plot_samples, replace=False)
    
    for i, idx in enumerate(indices):
        param_sample = theta_samples[idx]
        sol_sample = solve_ivp(system_ode_func, t_span, y0, args=tuple(param_sample), t_eval=t_eval)
        # 여러 선을 투명하게 겹쳐 그려서 '밴드'처럼 보이게 함
        label = 'Ours (Posterior Samples)' if i == 0 else ""
        ax.plot(sol_sample.t, sol_sample.y[1], 'steelblue', alpha=0.1, lw=1.5, label=label)

    ax.set_title("Insulin Dynamics (Hidden State Distribution)", fontsize=15, fontweight='bold')
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# =====================================================================
# 3. Main Evaluation Pipeline
# =====================================================================
def run_full_analysis(hidden_cvae, param_cvae, baseline_model, test_loader, config, system, normalizer):
    """
    테스트 셋 전체에 대한 정량적 평가 및 시각적 증명을 수행합니다.
    """
    print("\n\033[1;36m=== Running Probabilistic Analysis & Evaluation ===\033[0m")
    
    save_dir = os.path.join(config.RESULTS_DIR, config.EXPERIMENT_NAME + "_prob", "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    hidden_cvae.eval()
    param_cvae.eval()
    if baseline_model: baseline_model.eval()

    all_y_true, all_theta_true = [], []
    all_y_samples, all_theta_samples = [], []
    all_theta_base = []
    
    print("Evaluating entire test set and generating samples...")
    # 1. 전체 데이터셋 순회하며 샘플 수집
    for x_batch, y_batch, theta_batch in test_loader:
        x_batch = x_batch.to(config.DEVICE)
        
        # 확률 모델 추론 (Samples)
        y_samples, theta_samples = pseudo_gibbs_sampling(
            hidden_cvae, param_cvae, x_batch, num_chains=50, num_steps=60, burn_in=10
        )
        
        # Baseline 추론 (Point)
        if baseline_model:
            with torch.no_grad():
                theta_base_norm = baseline_model(x_batch)
                theta_base = normalizer.denormalize_params(theta_base_norm).cpu().numpy()
        else:
            theta_base = np.zeros_like(theta_batch.cpu().numpy())
            
        # Denormalization (물리적 스케일 복원)
        # (주의: Y 궤적도 정규화되어 있다면 normalizer.denormalize_trajectory 등이 필요합니다)
        theta_true = normalizer.denormalize_params(theta_batch).cpu().numpy()
        
        # 역정규화된 샘플들 저장
        # theta_samples 형태: [batch, num_chains, dim]
        theta_samples_phys = np.zeros_like(theta_samples)
        for i in range(theta_samples.shape[1]):
            theta_samples_phys[:, i, :] = normalizer.denormalize_params(
                torch.tensor(theta_samples[:, i, :]).to(config.DEVICE)
            ).cpu().numpy()

        all_y_true.append(y_batch.cpu().numpy())
        all_theta_true.append(theta_true)
        all_y_samples.append(y_samples)
        all_theta_samples.append(theta_samples_phys)
        all_theta_base.append(theta_base)

    # 데이터 병합
    y_true_full = np.concatenate(all_y_true, axis=0)
    theta_true_full = np.concatenate(all_theta_true, axis=0)
    y_samples_full = np.concatenate(all_y_samples, axis=0)
    theta_samples_full = np.concatenate(all_theta_samples, axis=0)
    theta_base_full = np.concatenate(all_theta_base, axis=0)

    # 2. 정량적 지표 계산 (전체 데이터셋 평균)
    rmse, pearson = calculate_point_metrics(theta_true_full, theta_samples_full)
    picp, mpiw = calculate_prediction_interval(theta_true_full, theta_samples_full)
    crps = calculate_crps(theta_true_full, theta_samples_full)
    
    print("\n\033[1;32m[Parameter Estimation Metrics (Test Set Average)]\033[0m")
    print(f"PICP (95% Coverage) : {np.mean(picp)*100:.2f}%")
    print(f"MPIW (Width)        : {np.mean(mpiw):.4f}")
    print(f"CRPS                : {np.mean(crps):.4f}")
    print(f"RMSE (Mean Est.)    : {np.mean(rmse):.4f}")
    print(f"Pearson r           : {np.mean(pearson):.4f}")
    
    # 3. 시각화 (인덱스 0번 샘플을 대표로 시각화)
    print("\nGenerating Visual Proofs...")
    sample_idx = 0
    time_points = np.linspace(0, 120, y_true_full.shape[1])
    
    # x_sparse도 필요하면 역정규화
    x_sample = normalizer.denormalize_x(test_loader.dataset[sample_idx][0].to(config.DEVICE)).cpu().numpy()
    
    # (1) 궤적 커버리지 플롯
    plot_trajectory_coverage(
        x_sample, y_true_full[sample_idx], y_samples_full[sample_idx], 
        time_points, save_path=os.path.join(save_dir, "1_trajectory_coverage.png")
    )
    
    # (2) 파라미터 결합 사후 분포
    plot_parameter_posterior(
        theta_true_full[sample_idx], theta_samples_full[sample_idx], 
        save_path=os.path.join(save_dir, "2_parameter_posterior.png")
    )
    
    if baseline_model:
        # (3) 대칭 붕괴 해결 증명 (Log-Log 공간)
        plot_symmetric_collapse_probabilistic(
            theta_true_full, theta_samples_full, theta_base_full, 
            save_path=os.path.join(save_dir, "3_symmetric_collapse.png")
        )
        
        # (4) ODE 기반 물리적 궤적 대역 증명
        y0 = [val[0] if isinstance(val, list) else val for val in system.initial_conditions]
        def ode_func(t, y, *params): return system.ode_func(t, y, params)
        
        plot_ogtt_trajectories_probabilistic(
            ode_func, y0, theta_true_full[sample_idx], theta_samples_full[sample_idx], theta_base_full[sample_idx], 
            save_path=os.path.join(save_dir, "4_ode_trajectories.png")
        )
    
    print(f"Analysis complete. All plots saved to: {save_dir}")