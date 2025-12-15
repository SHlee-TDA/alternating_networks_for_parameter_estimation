import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import Config
from models import ParameterEstimator, ExcludeLambda
from systems.ogtt_simul import OgttSimul
from utils import Normalizer
from data_loader import DataGenerator

# ... (load_experiment_context 함수는 이전과 동일하므로 생략) ...
# ... (load_experiment_context는 그대로 유지) ...
def load_experiment_context(experiment_name_query, device_override=None):
    """
    (이전 코드와 동일: Config 복원 시 hasattr 체크 제거한 버전 사용)
    """
    config = Config() 
    results_root = Path(config.RESULTS_DIR) / config.SYSTEM_NAME
    
    if not results_root.exists():
        raise FileNotFoundError(f"Results root directory not found: {results_root}")

    # --- 1. Smart Path Search ---
    target_dir = None
    exact_path = results_root / experiment_name_query
    
    if exact_path.exists():
        target_dir = exact_path
    else:
        candidates = []
        for p in results_root.iterdir():
            if not p.is_dir(): continue
            if p.name.endswith(f"_{experiment_name_query}") or experiment_name_query in p.name:
                candidates.append(p)
        
        if not candidates:
            raise FileNotFoundError(f"No experiment folder found matching '{experiment_name_query}'")
        
        candidates.sort(key=lambda x: x.name)
        target_dir = candidates[-1]

    print(f"[Info] Target Experiment Dir: {target_dir}")

    # --- 2. Load Config ---
    config_path = target_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            for k, v in saved_config.items():
                # [수정] hasattr 체크 제거하여 모든 설정 로드
                if not k.startswith('__'):
                    setattr(config, k, v)
    
    if device_override:
        config.DEVICE = device_override
    
    # --- 3. System & Data Load (Full Validation) ---
    system = OgttSimul()
    
    print(f"[Info] Loading FULL validation data (Scenario: {'SDE' if getattr(config, 'USE_SDE', False) else 'ODE'})...")
    data_gen = DataGenerator(system, config)
    obs_data, hid_data, params_data, t_points = data_gen.generate_data()
    
    # Normalizer 재현 (User 요청 반영: Log Parameter & Margin)
    scale_obs = np.percentile(np.abs(obs_data), 99.9)
    scale_hid = np.percentile(np.abs(hid_data), 99.9)
    calc_scales = [scale_obs * 1.2, scale_hid * 1.2]
    
    p_min = np.min(params_data, axis=0)
    p_max = np.max(params_data, axis=0)
    p_bounds = (p_min / 1.2, p_max * 1.2)
    
    # config에 USE_LOG_PARAMS가 있는지 확인 후 사용, 없으면 기본값 True 가정
    use_log = getattr(config, 'USE_LOG_PARAMS', True)
    normalizer = Normalizer(system, config.DEVICE, state_scales=calc_scales, param_bounds=p_bounds, use_log_params=use_log)
    
    # Validation Split
    num_samples = len(obs_data)
    test_size = int(num_samples * config.TEST_SPLIT)
    
    val_x = obs_data[-test_size:]
    val_y = hid_data[-test_size:]
    val_p = params_data[-test_size:]
    
    val_x_tensor = torch.tensor(val_x, dtype=torch.float32).view(test_size, -1).to(config.DEVICE)
    val_y_tensor = torch.tensor(val_y, dtype=torch.float32).view(test_size, -1).to(config.DEVICE)
    val_p_tensor = torch.tensor(val_p, dtype=torch.float32).to(config.DEVICE)
    
    val_x_norm = normalizer.normalize_inputs(val_x_tensor, 'observed')
    val_y_norm = normalizer.normalize_inputs(val_y_tensor, 'hidden')
    val_p_norm = normalizer.normalize_params(val_p_tensor)
    
    # --- 4. Load Model (g_phi) ---
    flat_x_dim = val_x_tensor.shape[1]
    flat_y_dim = val_y_tensor.shape[1]
    num_params = val_p_tensor.shape[1]

    g_phi = ParameterEstimator(
        flat_x_dim=flat_x_dim,
        flat_y_dim=flat_y_dim,
        num_params=num_params,
        model_config=config.MODEL_CONFIG['g_phi'],
        use_spectral_norm=config.USE_SPECTRAL_NORM
    ).to(config.DEVICE)
    
    weight_candidates = list(target_dir.rglob('g_phi.pth'))
    if not weight_candidates:
        weight_candidates = list(target_dir.rglob('best_model.pth'))
    
    if not weight_candidates:
         raise FileNotFoundError(f"No model weights found in {target_dir}")
    
    weight_path = weight_candidates[0]
    print(f"[Info] Loading weights from: {weight_path}")
    
    checkpoint = torch.load(weight_path, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'g_phi_state_dict' in checkpoint:
        g_phi.load_state_dict(checkpoint['g_phi_state_dict'])
    else:
        g_phi.load_state_dict(checkpoint)
        
    g_phi.eval()
    
    return g_phi, (val_x_norm, val_y_norm, val_p_norm), normalizer, config, target_dir

def diagnose_activations(g_phi, val_data, normalizer, save_dir):
    """
    Parameter-wise 진단: 각 파라미터(Si, Sigma)별로 통계와 분포를 분리하여 출력
    """
    x_batch, y_batch, p_batch = val_data
    
    activations = {}
    def get_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    network = g_phi.network
    hooks = []
    
    # Hook Register
    found_layers = {'linear': False, 'lambda': False, 'tanh': False}
    
    for i in reversed(range(len(network))):
        layer = network[i]
        if isinstance(layer, torch.nn.Tanh) and not found_layers['tanh']:
            hooks.append(layer.register_forward_hook(get_hook('3_Post_Tanh')))
            found_layers['tanh'] = True
        elif isinstance(layer, ExcludeLambda) and not found_layers['lambda']:
            hooks.append(layer.register_forward_hook(get_hook('2_Post_Lambda')))
            found_layers['lambda'] = True
        elif isinstance(layer, torch.nn.Linear) and not found_layers['linear']:
            hooks.append(layer.register_forward_hook(get_hook('1_Post_Linear_Raw')))
            found_layers['linear'] = True
            break
            
    # Forward Pass
    print(f"[Info] Running inference on {len(x_batch)} samples...")
    with torch.no_grad():
        _ = g_phi(x_batch, y_batch)
        
    for h in hooks: h.remove()
    
    # Handle Missing Lambda
    if '2_Post_Lambda' not in activations:
        activations['2_Post_Lambda'] = activations['1_Post_Linear_Raw']
        
    # Data Processing
    tanh_out_tensor = torch.tensor(activations['3_Post_Tanh'], device=x_batch.device)
    phys_out_np = normalizer.denormalize_params(tanh_out_tensor).cpu().numpy()
    activations['4_Physical_Scale'] = phys_out_np
    
    p_true_phys = normalizer.denormalize_params(p_batch).cpu().numpy()
    p_true_norm = p_batch.cpu().numpy()

    # --- 1. Print Statistics (Parameter-wise) ---
    param_names = ['P0 (Si)', 'P1 (Sigma)']
    num_params = p_true_phys.shape[1]
    
    steps = [
        ('1_Post_Linear_Raw', 'Pre-Activation (Linear)'),
        ('2_Post_Lambda', 'Post-Lambda (Pre-Tanh)'),
        ('3_Post_Tanh', 'Post-Tanh (Normalized)'),
        ('4_Physical_Scale', 'Final Output (Physical)')
    ]

    for p_idx in range(num_params):
        print(f"\n >>> Analysis for Parameter: {param_names[p_idx]} <<<")
        print("="*80)
        print(f"{'Layer / Stage':<30} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
        print("-" * 80)
        
        for key, name in steps:
            # 해당 파라미터(컬럼)만 슬라이싱
            data = activations[key][:, p_idx]
            print(f"{name:<30} | {np.mean(data):8.3f} | {np.std(data):8.3f} | {np.min(data):8.3f} | {np.max(data):8.3f}")
            
        print("-" * 80)
        # Ground Truth for this param
        gt_norm = p_true_norm[:, p_idx]
        gt_phys = p_true_phys[:, p_idx]
        print(f"{'Ground Truth (Normalized)':<30} | {np.mean(gt_norm):8.3f} | {np.std(gt_norm):8.3f} | {np.min(gt_norm):8.3f} | {np.max(gt_norm):8.3f}")
        print(f"{'Ground Truth (Physical)':<30} | {np.mean(gt_phys):8.3f} | {np.std(gt_phys):8.3f} | {np.min(gt_phys):8.3f} | {np.max(gt_phys):8.3f}")
        print("="*80)

    # --- 2. Plotting (2 Rows: One per Parameter) ---
    fig, axes = plt.subplots(num_params, 4, figsize=(24, 6 * num_params))
    colors = ['grey', 'orange', 'green', 'blue']
    
    for p_idx in range(num_params):
        for i, (key, title) in enumerate(steps):
            ax = axes[p_idx][i] if num_params > 1 else axes[i]
            data = activations[key][:, p_idx]
            
            sns.histplot(data, ax=ax, kde=True, color=colors[i], stat='density', alpha=0.4, bins=50)
            
            stats_txt = f"Mean: {np.mean(data):.2f}\nStd: {np.std(data):.2f}\nMin: {np.min(data):.2f}\nMax: {np.max(data):.2f}"
            ax.text(0.95, 0.95, stats_txt, transform=ax.transAxes, 
                         va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_title(f"[{param_names[p_idx]}] {title}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if 'Tanh' in title:
                ax.axvline(-1, c='r', ls='--'); ax.axvline(1, c='r', ls='--')
                
            if i == 3: # Compare with GT
                sns.kdeplot(p_true_phys[:, p_idx], ax=ax, color='red', linestyle='--', linewidth=2, label='GT')
                ax.legend()
                
    plt.tight_layout()
    save_path = save_dir / "diagnosis_plot_param_wise.png"
    plt.savefig(save_path, dpi=150)
    print(f"[Done] Diagnosis plot saved to: {save_path}")

def diagnose_weights_and_inputs(g_phi, val_data):
    """
    g_phi의 마지막 Linear Layer에 들어가는 입력(Input)과 
    그 레이어의 가중치(Weight/Bias) 상태를 정밀 분석합니다.
    """
    print("\n" + "="*30 + " [Deep Diagnosis: Weights & Inputs] " + "="*30)
    
    x_batch, y_batch, _ = val_data
    device = x_batch.device
    
    # 1. 마지막 Linear Layer 찾기
    final_linear = None
    for layer in reversed(g_phi.network):
        if isinstance(layer, torch.nn.Linear):
            final_linear = layer
            break
            
    if final_linear is None:
        print("[Error] Cannot find final Linear layer.")
        return

    # 2. 입력값 가로채기 (Hook)
    layer_inputs = []
    def hook(module, input, output):
        # input은 tuple 형태임
        layer_inputs.append(input[0].detach().cpu().numpy())
    
    handle = final_linear.register_forward_hook(hook)
    
    # Forward Pass (Sample)
    # 메모리 절약을 위해 2000개만
    idx = torch.randperm(len(x_batch))[:]
    with torch.no_grad():
        _ = g_phi(x_batch[idx], y_batch[idx])
        
    handle.remove()
    
    # 데이터 정리
    inputs = np.concatenate(layer_inputs, axis=0) # Shape: (N, In_Features)
    weights = final_linear.weight.detach().cpu().numpy() # Shape: (Out_Features, In_Features)
    bias = final_linear.bias.detach().cpu().numpy() if final_linear.bias is not None else np.zeros(weights.shape[0])
    
    # 3. 분석 및 시각화
    print(f"Final Linear Layer Shape: Weights {weights.shape}, Bias {bias.shape}")
    print(f"Captured Inputs Shape: {inputs.shape}")
    
    # (1) 입력값(x)의 부호 확인
    pos_input_ratio = np.mean(inputs > 0) * 100
    print(f"Positive Input Ratio: {pos_input_ratio:.1f}% (Expected: ~100% if hypothesis is true)")
    
    # (2) 가중치(W)의 분포 확인
    # Output Node별로 가중치 분포를 봅니다. (P0: Si, P1: Sigma)
    num_params = weights.shape[0]
    param_names = ['P0 (Si)', 'P1 (Sigma)']
    
    fig, axes = plt.subplots(num_params, 3, figsize=(18, 5 * num_params))
    
    for i in range(num_params):
        # Row selection
        ax_row = axes[i] if num_params > 1 else axes
        
        # A. Input Distribution (x)
        # 입력은 모든 파라미터가 공유하므로 전체 분포를 봅니다.
        sns.histplot(inputs.flatten(), ax=ax_row[0], color='skyblue', kde=True, stat='density')
        ax_row[0].set_title(f"Inputs to Final Layer (x)")
        ax_row[0].set_xlabel("Value")
        
        # B. Weight Distribution (W_i)
        # i번째 파라미터를 만들기 위한 가중치 벡터
        w_i = weights[i, :]
        sns.histplot(w_i, ax=ax_row[1], color='salmon', kde=True, stat='density')
        ax_row[1].set_title(f"Weights for {param_names[i]} ($W_{i}$)")
        ax_row[1].axvline(0, color='k', linestyle='--')
        
        # 통계 표시
        w_mean, w_std = np.mean(w_i), np.std(w_i)
        ax_row[1].text(0.95, 0.95, f"Mean: {w_mean:.4f}\nStd: {w_std:.4f}\nBias: {bias[i]:.4f}", 
                       transform=ax_row[1].transAxes, ha='right', va='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # C. Dot Product Contribution (W_i * x)
        # 가중치와 입력이 만나서 만드는 값(Pre-Bias)의 분포
        # 이것이 음수가 되어야 최종 출력이 음수가 됨
        dot_products = inputs @ w_i
        sns.histplot(dot_products, ax=ax_row[2], color='purple', kde=True, stat='density')
        ax_row[2].set_title(f"Linear Output ($W_{i}x$) contribution")
        ax_row[2].axvline(0, color='k', linestyle='--')
        
        # 실제 Bias가 더해진 최종 값 위치 표시
        final_mean = np.mean(dot_products + bias[i])
        ax_row[2].axvline(final_mean, color='r', linestyle='-', label=f'Mean w/ Bias: {final_mean:.2f}')
        ax_row[2].legend()

    plt.tight_layout()
    plt.show() # 혹은 plt.savefig(...)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name query")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    try:
        g_phi, val_data, normalizer, config, target_dir = load_experiment_context(args.exp_name, args.device)
        diagnose_activations(g_phi, val_data, normalizer, target_dir)
        diagnose_weights_and_inputs(g_phi, val_data)
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()