import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

# 프로젝트 루트 경로 추가 (main.py와 동일)
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# [연구자님의 코어 모듈 임포트]
from prob_models.config import ProbConfig
from prob_models.models import SingleCVAE, HiddenStateCVAE, ParameterCVAE
from prob_models.infer import single_cvae_sampling, pseudo_gibbs_sampling
from src.data_loader import DataGenerator, setup_dataloaders
from tools.exp_tools import get_system_class
from tools.interactive_file_selector import interactive_file_selector

# =====================================================================
# 1. 완벽하게 동기화된 Context Loader (main.py 구조 100% 반영)
# =====================================================================
def load_prob_experiment_context(config_path, prob_config_path, baseline_path, Hnet_path, Pnet_path, device_override="cuda"):
    # Config 객체 초기화
    config = ProbConfig()
    
    print(f"[Info] Loading Global Config from: {config_path}")
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
        for k, v in saved_config.items():
            if not k.startswith('__'):
                setattr(config, k, v)
                
    print(f"[Info] Loading Prob Config from: {prob_config_path}")
    with open(prob_config_path, 'r') as f:
        saved_prob_config = json.load(f)
        for k, v in saved_prob_config.items():
            if not k.startswith('__'):
                setattr(config, k, v) # 두 config를 하나로 병합하여 사용
                
    device = torch.device(device_override if torch.cuda.is_available() else "cpu")
    config.DEVICE = device
    
    # System 초기화
    SystemClass = get_system_class(config.SYSTEM_NAME)
    system = SystemClass()
    
    # 데이터 생성 및 Loader 셋업 (main.py와 완전히 동일한 로직)
    print(f"[Info] Generating/Loading Data Cache...")
    generator = DataGenerator(system, config)
    sim_data_tuple = generator.generate_data()
    
    print(f"[Info] Setting up DataLoaders...")
    # main.py에서 사용한 setup_dataloaders 호출
    loaders = setup_dataloaders(vars(config), sim_data_tuple, system, config)
    train_l, val_l, test_l, real_test_loader, p_init, normalizer = loaders
    
    # 모델 초기화를 위한 차원 추출
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
    Pnet.theta_dim = theta_dim # infer.py 안전장치

    # 가중치 로드 함수
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
# 2. 동적 꼬리 탐색 및 필터링 함수 (이전과 동일)
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
    """
    data의 shape이 (N, D)일 때, 첫 번째 파라미터(S_I)와 두 번째 파라미터(sigma)만 보고 필터링합니다.
    """
    # 0번 인덱스(SI)와 1번 인덱스(sigma)에 대해서만 조건을 겁니다.
    mask = (data[:, 0] > 0) & (data[:, 0] < si_max) & (data[:, 1] > 0) & (data[:, 1] < sig_max)
    
    # [수정된 부분] 전체 행을 살리되, 위 마스크를 만족하는 행만 뽑아냅니다.
    return data[mask, :]

# =====================================================================
# 3. 플로팅 함수 (이전과 동일하게 분리됨)
# =====================================================================
def plot_figure3_A_global_1D(global_true, global_single, global_iter, save_dir):
    print(f"Shape of global_true: {global_true.shape}, global_single: {global_single.shape}, global_iter: {global_iter.shape}")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    gt_f, sg_f, it_f = filter_outliers(global_true), filter_outliers(global_single), filter_outliers(global_iter)

    ax = axes[0]
    sns.kdeplot(sg_f[:, 0], ax=ax, fill=True, color="royalblue", alpha=0.5, label="Single CVAE", clip=(0, 10))
    sns.kdeplot(it_f[:, 0], ax=ax, fill=True, color="forestgreen", alpha=0.5, label="Iter CVAEs", clip=(0, 10))
    sns.kdeplot(gt_f[:, 0], ax=ax, color="black", linestyle="--", linewidth=2, label="True Prior", clip=(0, 10))
    ax.set(xlim=(0, 10), xlabel=r"Insulin Sensitivity ($S_I$)", ylabel="Density", title="A. Global 1D Marginals (Population)")
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

    sns.kdeplot(x=gt_f[:, 0], y=gt_f[:, 1], ax=ax, cmap="Greys", fill=True, thresh=0.05, levels=8, alpha=0.4)
    ax.scatter(sg_f[:, 0], sg_f[:, 1], color="royalblue", s=15, alpha=0.3, label="Single CVAE Samples")
    ax.scatter(it_f[:, 0], it_f[:, 1], color="forestgreen", s=15, alpha=0.3, label="Iter CVAEs Samples")
    
    ax.set(xlim=(0, 10), ylim=(0, 5), title=r"B. Global Joint $p(\theta)$", xlabel=r"Insulin Sensitivity ($S_I$)", ylabel=r"Secretion Capacity ($\sigma$)")
    ax.legend(loc="upper right", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / "figure3_B_global_2D.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure3_C_local_2D(local_true, local_single, local_iter, save_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sns.kdeplot(x=local_single[:, 0], y=local_single[:, 1], ax=ax, cmap="Blues", fill=True, thresh=0.05, levels=6, alpha=0.6, label="Single $q(\theta|x_{obs})$", clip=((0, 10), (0, 5)))
    sns.kdeplot(x=local_iter[:, 0], y=local_iter[:, 1], ax=ax, cmap="Greens", fill=True, thresh=0.05, levels=6, alpha=0.6, label="Iter $q(\theta|x_{obs})$", clip=((0, 10), (0, 5)))
    ax.scatter(local_true[0], local_true[1], color="firebrick", marker="*", s=500, edgecolor='black', linewidth=1.5, label="Ground Truth", zorder=10)
    
    ax.set(xlim=(0, 10), ylim=(0, 5), title=r"C. Local Posterior $q(\theta | x_{obs})$", xlabel=r"Insulin Sensitivity ($S_I$)", ylabel=r"Secretion Capacity ($\sigma$)")
    ax.legend(loc="upper right", fontsize=11)
    
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

        # 1. 모델과 Test Loader 완벽 로드
        Hnet, Pnet, base_cvae, test_l, normalizer, target_dir = load_prob_experiment_context(
            config_path, prob_config_path, baseline_path, Hnet_path, Pnet_path
        )
        
        print("\n=== 1. Extracting Data from Test Loader ===")
        x_list, p_list = [], []
        # Test 데이터 중 연산 속도를 위해 앞부분 일부 배치(약 1000개 수준)만 추출
        total_samples = 0
        for x_batch, _, p_batch in test_l:
            x_list.append(x_batch)
            p_list.append(p_batch)
            total_samples += x_batch.size(0)
            if total_samples >= 1000:
                break
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_global = torch.cat(x_list, dim=0).to(device)
        p_true_global = torch.cat(p_list, dim=0).to(device)

        print("\n=== 2. Generating Global Distribution (Population Level) ===")
        with torch.no_grad():
            p_single_global_norm = single_cvae_sampling(base_cvae, x_global, num_samples=1).squeeze(1)
            _, p_iter_global_norm, _ = pseudo_gibbs_sampling(Hnet, Pnet, x_global, num_chains=1, num_steps=30, burn_in=20)
        
        # 훈련 시점의 완벽한 Normalizer로 물리적 값 복원
        
        p_true_phys = normalizer.denormalize_params(p_true_global).cpu().numpy()
        p_single_global_phys = normalizer.denormalize_params(p_single_global_norm).cpu().numpy()
        p_iter_global_phys = normalizer.denormalize_params(p_iter_global_norm[:,-1, :]).cpu().numpy()

        print(f"Global distribution shapes - True: {p_true_phys.shape}, Single CVAE: {p_single_global_phys.shape}, Iter CVAEs: {p_iter_global_phys.shape}")

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