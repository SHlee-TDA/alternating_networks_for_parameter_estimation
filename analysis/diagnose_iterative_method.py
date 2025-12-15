import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from scipy.stats import wasserstein_distance

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import Config
from models import ParameterEstimator, HiddenVarPredictor
from systems.ogtt_simul import OgttSimul
from utils import Normalizer
from data_loader import DataGenerator

def load_experiment_context(experiment_name_query, device_override=None):
    """
    실험 설정을 로드하고 모델(f_theta, g_phi)과 데이터를 준비합니다.
    """
    config = Config() 
    results_root = Path(config.RESULTS_DIR) / config.SYSTEM_NAME
    
    # 1. 경로 탐색 (Smart Search)
    target_dir = None
    exact_path = results_root / experiment_name_query
    
    if exact_path.exists():
        target_dir = exact_path
    else:
        candidates = []
        if not results_root.exists():
             raise FileNotFoundError(f"Results root not found: {results_root}")
             
        for p in results_root.iterdir():
            if not p.is_dir(): continue
            if p.name.endswith(f"_{experiment_name_query}") or experiment_name_query in p.name:
                candidates.append(p)
        
        if not candidates:
            raise FileNotFoundError(f"No experiment found for query: '{experiment_name_query}'")
        
        candidates.sort(key=lambda x: x.name)
        target_dir = candidates[-1]

    print(f"[Info] Target Experiment: {target_dir.name}")

    # 2. Config 로드
    config_path = target_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            for k, v in saved_config.items():
                if not k.startswith('__'):
                    setattr(config, k, v)
    
    if device_override:
        config.DEVICE = device_override
        
    # 3. 데이터 로드 (Validation)
    system = OgttSimul()
    print(f"[Info] Loading Validation Data...")
    data_gen = DataGenerator(system, config)
    obs_data, hid_data, params_data, _ = data_gen.generate_data()
    
    # Normalizer 설정 (기존 설정 반영)
    scale_obs = np.percentile(np.abs(obs_data), 99.9)
    scale_hid = np.percentile(np.abs(hid_data), 99.9)
    calc_scales = [scale_obs * 1.2, scale_hid * 1.2]
    
    p_min = np.min(params_data, axis=0)
    p_max = np.max(params_data, axis=0)
    p_bounds = (p_min / 1.2, p_max * 1.2)
    
    use_log = getattr(config, 'USE_LOG_PARAMS', True)
    normalizer = Normalizer(system, config.DEVICE, state_scales=calc_scales, param_bounds=p_bounds, use_log_params=use_log)
    
    # Validation Split
    num_samples = len(obs_data)
    test_size = int(num_samples * config.TEST_SPLIT)
    
    val_x = obs_data[-test_size:]
    val_y = hid_data[-test_size:] # Hidden GT (비교용)
    val_p = params_data[-test_size:] # Param GT
    
    # Tensor 변환
    val_x_tensor = torch.tensor(val_x, dtype=torch.float32).view(test_size, -1).to(config.DEVICE)
    val_p_tensor = torch.tensor(val_p, dtype=torch.float32).to(config.DEVICE)
    
    val_x_norm = normalizer.normalize_inputs(val_x_tensor, 'observed')
    
    # 4. 모델 로드 (f_theta, g_phi 둘 다 필요)
    flat_x_dim = val_x_tensor.shape[1]
    # flat_y_dim 계산 (Hidden dimension)
    flat_y_dim = val_y.shape[1] * val_y.shape[2] 
    num_params = val_p_tensor.shape[1]

    # f_theta 로드
    f_theta = HiddenVarPredictor(
        flat_x_dim=flat_x_dim,
        flat_y_dim=flat_y_dim,
        num_params=num_params,
        model_config=config.MODEL_CONFIG['f_theta'],
        use_spectral_norm=config.USE_SPECTRAL_NORM
    ).to(config.DEVICE)
    
    # g_phi 로드
    g_phi = ParameterEstimator(
        flat_x_dim=flat_x_dim,
        flat_y_dim=flat_y_dim,
        num_params=num_params,
        model_config=config.MODEL_CONFIG['g_phi'],
        use_spectral_norm=config.USE_SPECTRAL_NORM
    ).to(config.DEVICE)
    
    # 가중치 찾기 및 로드
    def load_weights(model, filename):
        candidates = list(target_dir.rglob(filename))
        if not candidates:
            # best_model.pth fallback
            candidates = list(target_dir.rglob('best_model.pth'))
            if not candidates: return False
            ckpt = torch.load(candidates[0], map_location=config.DEVICE)
            key = f"{filename.split('.')[0]}_state_dict"
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                return True
            return False
        
        model.load_state_dict(torch.load(candidates[0], map_location=config.DEVICE))
        return True

    if not load_weights(f_theta, 'f_theta.pth'): raise FileNotFoundError("f_theta weights not found")
    if not load_weights(g_phi, 'g_phi.pth'): raise FileNotFoundError("g_phi weights not found")
    
    f_theta.eval()
    g_phi.eval()
    
    return f_theta, g_phi, (val_x_norm, val_p_tensor), normalizer, config, target_dir

def analyze_iterative_process(f_theta, g_phi, val_data, normalizer, config, save_dir):
    """
    Iterative Method를 실행하며 스텝별 분포 변화를 추적하고 시각화합니다.
    """
    x_norm, p_true_phys_tensor = val_data
    batch_size = x_norm.size(0)
    iterations = 10  # 추적할 반복 횟수
    
    # 1. 초기 추정값 설정 (P_init)
    # 보통 학습 데이터의 평균이나 임의의 값에서 시작
    # 여기서는 P_true의 평균값을 초기값으로 가정 (Uninformative Prior)
    p_mean = p_true_phys_tensor.mean(dim=0, keepdim=True) # (1, ParamDim)
    p_curr_phys = p_mean.repeat(batch_size, 1) # 모든 샘플 동일 시작
    p_curr_norm = normalizer.normalize_params(p_curr_phys)
    
    # 기록 저장소
    history_phys = [p_curr_phys.cpu().numpy()] # Step 0
    
    print(f"[Info] Running Iterative Inference for {iterations} steps...")
    
    # 2. Iteration Loop
    with torch.no_grad():
        for k in range(iterations):
            # Step A: P -> Y
            y_hat = f_theta(x_norm, p_curr_norm)
            
            # Step B: Y -> P
            p_next_norm = g_phi(x_norm, y_hat)
            
            # Record Physical Values
            p_next_phys = normalizer.denormalize_params(p_next_norm)
            history_phys.append(p_next_phys.cpu().numpy())
            
            # Update
            p_curr_norm = p_next_norm
            
    # Ground Truth
    p_true = p_true_phys_tensor.cpu().numpy()
    
    # --- Visualization ---
    plot_distribution_evolution(history_phys, p_true, save_dir)
    plot_joint_distribution_evolution(history_phys, p_true, save_dir)
    calculate_metrics_evolution(history_phys, p_true, save_dir)

def plot_distribution_evolution(history, p_true, save_dir):
    """
    각 파라미터별 Marginal Distribution의 변화를 KDE Plot으로 시각화
    """
    num_steps = len(history)
    num_params = p_true.shape[1]
    param_names = ['Si (Sensitivity)', 'Sigma (Secretion)']
    
    # 보고 싶은 스텝 선정 (너무 많으면 복잡하므로)
    steps_to_show = [0, 1, 3, 5, num_steps-1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_to_show)))
    
    fig, axes = plt.subplots(1, num_params, figsize=(16, 6))
    if num_params == 1: axes = [axes]
    
    for i in range(num_params):
        ax = axes[i]
        
        # Ground Truth 분포 (배경)
        sns.kdeplot(p_true[:, i], ax=ax, color='black', linestyle='--', linewidth=2, fill=True, alpha=0.1, label='Ground Truth')
        
        # Iteration별 분포
        for idx, step in enumerate(steps_to_show):
            data = history[step][:, i]
            label = f'Iter {step}' if step > 0 else 'Init'
            sns.kdeplot(data, ax=ax, color=colors[idx], label=label, linewidth=1.5)
            
        ax.set_title(f"Evolution of {param_names[i]}")
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_dir / "iter_marginal_evolution.png", dpi=150)
    print(f"[Plot] Marginal evolution saved.")

def plot_joint_distribution_evolution(history, p_true, save_dir):
    """
    두 파라미터 간의 Joint Distribution (Scatter/Contour) 변화 시각화
    """
    if p_true.shape[1] < 2: return # 1D면 생략
    
    steps_to_show = [0, 1, 5, len(history)-1]
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(5 * len(steps_to_show), 5), sharex=True, sharey=True)
    
    # Ground Truth 범위 계산 (축 고정)
    x_min, x_max = p_true[:, 0].min(), p_true[:, 0].max()
    y_min, y_max = p_true[:, 1].min(), p_true[:, 1].max()
    margin = 0.2
    
    for idx, step in enumerate(steps_to_show):
        ax = axes[idx]
        data = history[step]
        
        # Ground Truth (Grey Contour)
        sns.kdeplot(x=p_true[:, 0], y=p_true[:, 1], ax=ax, color='grey', alpha=0.3, levels=5, fill=True)
        
        # Current Step Distribution (Scatter or KDE)
        # 샘플이 많으면 Scatter가 지저분하므로 KDE 권장, 여기서는 Scatter+Alpha 사용
        ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.4, c='blue', label=f'Pred (Iter {step})')
        
        # 중심 이동 경로 표시 (Mean Trajectory)
        mean_curr = np.mean(data, axis=0)
        mean_true = np.mean(p_true, axis=0)
        ax.plot(mean_curr[0], mean_curr[1], 'r*', markersize=12, label='Mean Pred')
        ax.plot(mean_true[0], mean_true[1], 'kX', markersize=12, label='Mean GT')
        
        ax.set_title(f"Step {step}")
        ax.set_xlabel("Si")
        if idx == 0: ax.set_ylabel("Sigma")
        
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "iter_joint_evolution.png", dpi=150)
    print(f"[Plot] Joint distribution evolution saved.")

def calculate_metrics_evolution(history, p_true, save_dir):
    """
    각 스텝별로 Ground Truth 분포와의 거리(Wasserstein Distance)를 계산하여 그래프로 표시
    """
    num_steps = len(history)
    num_params = p_true.shape[1]
    
    w_distances = np.zeros((num_steps, num_params))
    
    for k in range(num_steps):
        for i in range(num_params):
            # 1D Wasserstein Distance (Earth Mover's Distance)
            w_dist = wasserstein_distance(history[k][:, i], p_true[:, i])
            w_distances[k, i] = w_dist
            
    # Plotting
    plt.figure(figsize=(8, 5))
    param_names = ['Si', 'Sigma']
    colors = ['tab:blue', 'tab:orange']
    
    for i in range(num_params):
        plt.plot(range(num_steps), w_distances[:, i], marker='o', label=f'WD({param_names[i]})', color=colors[i])
        
    plt.title("Distribution Distance from Ground Truth over Iterations")
    plt.xlabel("Iteration Step")
    plt.ylabel("Wasserstein Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_dir / "iter_metric_convergence.png", dpi=150)
    print(f"[Plot] Metric convergence saved.")
    
    # 텍스트 로그 출력
    print("\n[Convergence Metrics]")
    print(f"{'Step':<5} | {'WD(Si)':<10} | {'WD(Sigma)':<10}")
    print("-" * 35)
    for k in range(num_steps):
        print(f"{k:<5} | {w_distances[k, 0]:.4f}     | {w_distances[k, 1]:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name query")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    try:
        f_theta, g_phi, val_data, normalizer, config, target_dir = load_experiment_context(args.exp_name, args.device)
        analyze_iterative_process(f_theta, g_phi, val_data, normalizer, config, target_dir)
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()