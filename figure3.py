import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prob_models.config import ProbConfig
from prob_models.models import SingleCVAE, HiddenStateCVAE, ParameterCVAE
from prob_models.infer import single_cvae_sampling, pseudo_gibbs_sampling
from src.data_loader import DataGenerator, setup_dataloaders
from tools.exp_tools import get_system_class
from tools.interactive_file_selector import interactive_file_selector

# =====================================================================
# 1. 완벽하게 동기화된 Context Loader
# =====================================================================
def load_prob_experiment_context(config_path, prob_config_path, baseline_path, Hnet_path, Pnet_path, device_override="cuda"):
    config = ProbConfig()
    
    print(f"[Info] Loading Global Config from: {config_path}")
    with open(config_path, 'r') as f:
        for k, v in json.load(f).items():
            if not k.startswith('__'): setattr(config, k, v)
                
    print(f"[Info] Loading Prob Config from: {prob_config_path}")
    with open(prob_config_path, 'r') as f:
        for k, v in json.load(f).items():
            if not k.startswith('__'): setattr(config, k, v)
                
    device = torch.device(device_override if torch.cuda.is_available() else "cpu")
    config.DEVICE = device
    
    SystemClass = get_system_class(config.SYSTEM_NAME)
    system = SystemClass()
    
    print(f"[Info] Generating/Loading Data Cache...")
    generator = DataGenerator(system, config)
    sim_data_tuple = generator.generate_data()
    
    print(f"[Info] Setting up DataLoaders...")
    loaders = setup_dataloaders(vars(config), sim_data_tuple, system, config)
    train_l, val_l, test_l, real_test_loader, p_init, normalizer = loaders
    
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
    Pnet.theta_dim = theta_dim

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
    target_dir = Path(Hnet_path).parent 
    
    return Hnet, Pnet, baseline_cvae, test_l, normalizer, target_dir

# =====================================================================
# 2. 동적 꼬리 탐색 및 필터링 함수
# =====================================================================
def find_best_mean_trap_sample(p_true_phys, p_single_global_phys, target_percentile=10):
    si_t, sig_t = p_true_phys[:, 0], p_true_phys[:, 1]
    si_s, sig_s = p_single_global_phys[:, 0], p_single_global_phys[:, 1]
    
    current_percentile = target_percentile
    best_idx = -1
    
    while current_percentile <= 50:
        si_threshold = np.percentile(si_t, 100 - current_percentile)
        sig_threshold = np.percentile(sig_t, current_percentile)
        mask = (si_t >= si_threshold) & (sig_t <= sig_threshold)
        
        if np.any(mask):
            dist = (si_t - si_s)**2 + (sig_t - sig_s)**2
            dist[~mask] = -1.0
            best_idx = np.argmax(dist)
            print(f"[*] Found optimal tail sample at top/bottom {current_percentile}% boundary.")
            break
        current_percentile += 5

    if best_idx == -1:
        dist = (si_t - si_s)**2 + (sig_t - sig_s)**2
        best_idx = np.argmax(dist)
    return best_idx

def filter_outliers(data, si_max=10.0, sig_max=5.0):
    mask = (data[:, 0] > 0) & (data[:, 0] < si_max) & (data[:, 1] > 0) & (data[:, 1] < sig_max)
    return data[mask, :]

# =====================================================================
# 3. 플로팅 함수 (이론적 궤적 추가)
# =====================================================================
def plot_figure3_A_global_1D(global_true, global_single, global_iter, save_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    gt_f, sg_f, it_f = filter_outliers(global_true), filter_outliers(global_single), filter_outliers(global_iter)

    ax = axes[0]
    sns.kdeplot(sg_f[:, 0], ax=ax, fill=True, color="royalblue", alpha=0.5, label="Single CVAE", clip=(0, 10))
    sns.kdeplot(it_f[:, 0], ax=ax, fill=True, color="forestgreen", alpha=0.5, label="A-DCVAE", clip=(0, 10))
    sns.kdeplot(gt_f[:, 0], ax=ax, color="black", linestyle="--", linewidth=2, label="True Prior", clip=(0, 10))
    ax.set(xlim=(0, 5), xlabel=r"Insulin Sensitivity ($S_I$)", ylabel="Density", title="A. Global 1D Marginals (Population)")
    ax.legend(loc="upper right", fontsize=11)

    ax = axes[1]
    sns.kdeplot(sg_f[:, 1], ax=ax, fill=True, color="royalblue", alpha=0.5, clip=(0, 5))
    sns.kdeplot(it_f[:, 1], ax=ax, fill=True, color="forestgreen", alpha=0.5, clip=(0, 5))
    sns.kdeplot(gt_f[:, 1], ax=ax, color="black", linestyle="--", linewidth=2, clip=(0, 5))
    ax.set(xlim=(0, 5), xlabel=r"Secretion Capacity ($\sigma$)", ylabel="Density")

    plt.tight_layout()
    plt.savefig(save_dir / "figure3_A_global_1D.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure3_B_global_2D(global_true, global_single, global_iter, save_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    gt_f, sg_f, it_f = filter_outliers(global_true), filter_outliers(global_single), filter_outliers(global_iter)

    # 1. Ground Truth 분포를 '지형(Landscape)'처럼 명확하게 렌더링
    # 1-1. 옅은 쉐이딩 (바닥면)
    sns.kdeplot(x=gt_f[:, 0], y=gt_f[:, 1], ax=ax, cmap="Greys", fill=True, thresh=0.05, levels=8, alpha=0.3)
    # 1-2. 진한 등고선 라인 (지형의 등고선을 명확히 묘사하여 시인성 극대화)
    sns.kdeplot(x=gt_f[:, 0], y=gt_f[:, 1], ax=ax, color="dimgrey", linewidths=1.5, thresh=0.05, levels=8, alpha=0.8)

    # 2. 각 모델의 1-shot 예측 산점도 (구름처럼 흩뿌리기)
    ax.scatter(sg_f[:, 0], sg_f[:, 1], color="royalblue", s=15, alpha=0.3)
    ax.scatter(it_f[:, 0], it_f[:, 1], color="forestgreen", s=15, alpha=0.3)
    
    ax.set(xlim=(0, 3), ylim=(0, 3), title=r"B. Global Joint $p(\theta)$", xlabel=r"Insulin Sensitivity ($S_I$)", ylabel=r"Secretion Capacity ($\sigma$)")
    
    # 3. 직관적인 커스텀 레전드(Legend) 생성
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        # Ground Truth를 면(Patch)으로 표시하여 Contour임을 명시
        Patch(facecolor='lightgrey', edgecolor='dimgrey', linewidth=1.5, label='True Prior Density'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=8, label='Single CVAE Samples'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', markersize=8, label='A-DCVAE Samples')
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "figure3_B_global_2D.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure3_C_local_2D(local_true, local_single, local_iter, save_dir):
    from matplotlib.patches import Circle, ConnectionPatch, Patch
    from matplotlib.lines import Line2D
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Main Plot 렌더링
    sns.kdeplot(x=local_single[:, 0], y=local_single[:, 1], ax=ax, cmap="Blues", fill=True, thresh=0.05, levels=6, alpha=0.6, clip=((0, 10), (0, 5)))
    sns.kdeplot(x=local_iter[:, 0], y=local_iter[:, 1], ax=ax, cmap="Greens", fill=True, thresh=0.05, levels=6, alpha=0.6, clip=((0, 10), (0, 5)))
    
    # 2. 이론적 비식별성 매니폴드 곡선
    C_val = local_true[0] * local_true[1]
    si_range = np.linspace(0.05, 10, 500)
    sigma_curve = C_val / si_range
    ax.plot(si_range, sigma_curve, color='purple', linestyle='--', linewidth=2.5, alpha=0.8, zorder=5)

    # 3. Ground Truth 정답 별표
    ax.scatter(local_true[0], local_true[1], color="firebrick", marker="*", s=500, edgecolor='black', linewidth=1.5, zorder=10)
    
    # ---------------------------------------------------------
    # 4. 현미경 뷰 (원형 돋보기) 생성 및 세팅
    # ---------------------------------------------------------
    single_mean_x = local_single[:, 0].mean()
    single_mean_y = local_single[:, 1].mean()
    single_std_x = local_single[:, 0].std()
    single_std_y = local_single[:, 1].std()

    # 원형이 찌그러지지 않도록 가로세로를 정사각형(square) 비율로 맞춤
    margin = max(single_std_x * 6, single_std_y * 6, 0.05) 

    # 범례 아래 공간(우측 상단)에 돋보기 축 생성
    axins = ax.inset_axes([0.5, 0.4, 0.2, 0.2])
    
    # [추가 요청] 확대 뷰 안에 모든 요소를 'KDE'와 선으로 동일하게 렌더링!
    sns.kdeplot(x=local_single[:, 0], y=local_single[:, 1], ax=axins, cmap="Blues", fill=True, thresh=0.05, levels=6, alpha=0.8)
    sns.kdeplot(x=local_iter[:, 0], y=local_iter[:, 1], ax=axins, cmap="Greens", fill=True, thresh=0.05, levels=6, alpha=0.4)
    axins.plot(si_range, sigma_curve, color='purple', linestyle='--', linewidth=3, alpha=0.8, zorder=5)
    axins.scatter(local_true[0], local_true[1], color="firebrick", marker="*", s=300, edgecolor='black', linewidth=1.5, zorder=10)
    
    # 돋보기 초점(영역) 맞추기
    axins.set_xlim(single_mean_x - margin, single_mean_x + margin)
    axins.set_ylim(single_mean_y - margin, single_mean_y + margin)
    
    # --- [매직] 사각형 축을 '원형(Circle)'으로 깎아내는 클리핑 기법 ---
    circle_patch = Circle((0.5, 0.5), 0.5, transform=axins.transAxes, fill=False, edgecolor='firebrick', linewidth=2.5, zorder=20)
    axins.add_patch(circle_patch)
    
    # 현미경 안의 모든 그래프 요소(KDE 음영, 선 등)를 원 모양에 맞춰 자릅니다.
    for c in axins.collections:
        c.set_clip_path(circle_patch)
    for l in axins.lines:
        l.set_clip_path(circle_patch)
        
    axins.axis('off') # 사각형 테두리와 눈금선 숨김
    #axins.set_title("Microscope View\n(Single vs Iter)", fontsize=11, color='firebrick', fontweight='bold', y=1.05)
    
    # ---------------------------------------------------------
    # 5. 본래 플롯에 타겟 원(과녁) 그리고 연결선 긋기
    # ---------------------------------------------------------
    # 메인 플롯 위에 돋보기가 보고 있는 영역을 붉은 점선 원으로 표시
    target_circle = Circle((single_mean_x, single_mean_y), margin, facecolor='none', edgecolor='firebrick', linewidth=1.5, linestyle='--', zorder=15)
    ax.add_patch(target_circle)
    
    # 타겟 원의 맨 위(Top)와 현미경 렌즈의 맨 아래(Bottom)를 점선으로 연결
    con = ConnectionPatch(xyA=(single_mean_x, single_mean_y + margin), xyB=(0.5, 0.0),
                          coordsA="data", coordsB="axes fraction",
                          axesA=ax, axesB=axins,
                          color="firebrick", linewidth=1.5, linestyle=":", alpha=0.8, zorder=15)
    ax.add_artist(con)

    ax.set(xlim=(0, 2), ylim=(0, 2), title=r"C. Local Posterior $q(\theta | x_{obs})$", xlabel=r"Insulin Sensitivity ($S_I$)", ylabel=r"Secretion Capacity ($\sigma$)")
    
    # 6. Custom Legend (loc="upper right" 유지)
    legend_elements = [
        Patch(facecolor='royalblue', alpha=0.6, label=r'Single $q(\theta|x_{obs})$'),
        Patch(facecolor='forestgreen', alpha=0.6, label=r'A-DCVAE $q(\theta|x_{obs})$'),
        Line2D([0], [0], color='purple', linestyle='--', linewidth=2.5, label=rf"Theoretical Valley ($C={C_val:.1f}$)"),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='firebrick', markersize=15, markeredgecolor='black', label='Ground Truth')
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / "figure3_C_local_2D.pdf", dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 4. 메인 실행부
# =====================================================================
def main():
    try:
        print("\n\033[1;33m=== [Figure 3 생성을 위한 파일 선택] ===\033[0m")
        base_search_dir = "./results"
        
        rel_config_path = interactive_file_selector("[1/4] 실험 설정 파일 (config.json) 선택:", start_dir=base_search_dir)
        config_path = os.path.join(base_search_dir, rel_config_path)
        
        rel_prob_config_path = interactive_file_selector("[2/4] 확률 모델 설정 파일 (prob_config.json) 선택:", start_dir=base_search_dir)
        prob_config_path = os.path.join(base_search_dir, rel_prob_config_path)
        
        rel_baseline_path = interactive_file_selector("[3/4] Single CVAE 가중치 선택:", start_dir=base_search_dir)
        baseline_path = os.path.join(base_search_dir, rel_baseline_path)
        
        rel_Hnet_path = interactive_file_selector("[4/4] Hidden CVAE 가중치 선택 (Param CVAE 자동로드):", start_dir=base_search_dir)
        Hnet_path = os.path.join(base_search_dir, rel_Hnet_path)
        Pnet_path = Hnet_path.replace('hidden_cvae.pth', 'param_cvae.pth').replace('Hnet.pth', 'Pnet.pth')

        # 1. 모델과 Test Loader 로드
        Hnet, Pnet, base_cvae, test_l, normalizer, target_dir = load_prob_experiment_context(
            config_path, prob_config_path, baseline_path, Hnet_path, Pnet_path
        )
        
        print("\n=== 1. Extracting Data from Test Loader ===")
        x_list, p_list = [], []
        total_samples = 0
        for x_batch, _, p_batch in test_l:
            x_list.append(x_batch)
            p_list.append(p_batch)
            total_samples += x_batch.size(0)
            if total_samples >= 1000:
                break
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_global = torch.cat(x_list, dim=0).to(device)
        p_true_global = torch.cat(p_list, dim=0).to(device)

        print("\n=== 2. Generating Global Distribution (Population Level) ===")
        with torch.no_grad():
            p_single_global_norm = single_cvae_sampling(base_cvae, x_global, num_samples=1).squeeze(1)
            _, p_iter_global_norm, _ = pseudo_gibbs_sampling(Hnet, Pnet, x_global, num_chains=1, num_steps=30, burn_in=20)
        
        p_iter_global_norm = p_iter_global_norm.squeeze(1)
        
        p_true_phys = normalizer.denormalize_params(p_true_global).cpu().numpy()
        p_single_global_phys = normalizer.denormalize_params(p_single_global_norm).cpu().numpy()
        p_iter_global_phys = normalizer.denormalize_params(p_iter_global_norm).cpu().numpy()

        print("\n=== 3. Finding Extreme Tail Sample ===")
        tail_idx = find_best_mean_trap_sample(p_true_phys, p_single_global_phys)
        x_local = x_global[tail_idx:tail_idx+1]
        p_true_local = p_true_phys[tail_idx]
        print(f"[*] Tail Sample Selected! True SI: {p_true_local[0]:.3f}, True Sigma: {p_true_local[1]:.3f}")

        print("\n=== 4. Generating Local Posterior (Specific Sample Level) ===")
        num_chains = 1000
        with torch.no_grad():
            p_single_local_norm = single_cvae_sampling(base_cvae, x_local, num_samples=num_chains).squeeze(0)
            _, p_iter_local_norm, _ = pseudo_gibbs_sampling(Hnet, Pnet, x_local, num_chains=num_chains, num_steps=50, burn_in=30)
            p_iter_local_norm = p_iter_local_norm.squeeze(0)

        p_single_local_phys = normalizer.denormalize_params(p_single_local_norm).cpu().numpy()
        p_iter_local_phys = normalizer.denormalize_params(p_iter_local_norm).cpu().numpy()

        print("\n=== 5. Plotting Figure 3 (A, B, C separately) ===")
        plot_figure3_A_global_1D(p_true_phys, p_single_global_phys, p_iter_global_phys, target_dir)
        plot_figure3_B_global_2D(p_true_phys, p_single_global_phys, p_iter_global_phys, target_dir)
        plot_figure3_C_local_2D(p_true_local, p_single_local_phys, p_iter_local_phys, target_dir)
        
        print(f"\n[Success] 3 separate PDF figures successfully saved to: {target_dir}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()