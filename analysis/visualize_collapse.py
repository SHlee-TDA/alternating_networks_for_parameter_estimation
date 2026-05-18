import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

# 프로젝트 루트 경로 추가 (사용자 환경에 맞춤)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import Config
from src.models import ParameterEstimator, HiddenVarPredictor
from systems.ogtt_simul import OgttSimul
from src.utils import Normalizer
from src.data_loader import DataGenerator

# --- 1. 설정 및 모델 로딩 (기존 코드 유지) ---
def load_experiment_context(experiment_name_query, device_override=None):
    """
    실험 설정을 로드하고 모델(f_theta, g_phi)과 데이터를 준비합니다.
    """
    config = Config()
    results_root = Path(config.RESULTS_DIR) / config.SYSTEM_NAME
    
    # 경로 탐색
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
            if experiment_name_query in p.name:
                candidates.append(p)
        
        if not candidates:
            raise FileNotFoundError(f"No experiment found for query: '{experiment_name_query}'")
        candidates.sort(key=lambda x: x.name)
        target_dir = candidates[-1]

    print(f"[Info] Target Experiment: {target_dir.name}")

    # Config 로드
    config_path = target_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            for k, v in saved_config.items():
                if not k.startswith('__'):
                    setattr(config, k, v)
    
    if device_override:
        config.DEVICE = device_override
        
    # 데이터 로드 (Validation)
    system = OgttSimul()
    print(f"[Info] Loading Validation Data...")
    data_gen = DataGenerator(system, config)
    obs_data, hid_data, params_data, _ = data_gen.generate_data()
    
    # Normalizer 설정
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
    val_p = params_data[-test_size:] # Param GT
    
    val_x_tensor = torch.tensor(val_x, dtype=torch.float32).view(test_size, -1).to(config.DEVICE)
    val_p_tensor = torch.tensor(val_p, dtype=torch.float32).to(config.DEVICE)
    val_x_norm = normalizer.normalize_inputs(val_x_tensor, 'observed')
    
    # 모델 로드
    flat_x_dim = val_x_tensor.shape[1]
    # hidden dimension 계산 (hid_data shape: N, T, D)
    flat_y_dim = hid_data.shape[1] * hid_data.shape[2]
    num_params = val_p_tensor.shape[1]

    f_theta = HiddenVarPredictor(
        flat_x_dim=flat_x_dim, flat_y_dim=flat_y_dim, num_params=num_params,
        model_config=config.MODEL_CONFIG['f_theta'], use_spectral_norm=config.USE_SPECTRAL_NORM
    ).to(config.DEVICE)
    
    g_phi = ParameterEstimator(
        flat_x_dim=flat_x_dim, flat_y_dim=flat_y_dim, num_params=num_params,
        model_config=config.MODEL_CONFIG['g_phi'], use_spectral_norm=config.USE_SPECTRAL_NORM
    ).to(config.DEVICE)
    
    # 가중치 로드 함수
    def load_weights(model, filename):
        candidates = list(target_dir.rglob(filename))
        if not candidates:
            # fallback
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

    if not load_weights(f_theta, 'f_theta.pth'): print("Warning: f_theta weights not found")
    if not load_weights(g_phi, 'g_phi.pth'): print("Warning: g_phi weights not found")
    
    f_theta.eval()
    g_phi.eval()
    
    return f_theta, g_phi, (val_x_norm, val_p_tensor), normalizer, config, target_dir

# --- 2. 추론 실행 함수 (Inference Loop) ---
def run_inference(f_theta, g_phi, val_data, normalizer):
    """
    Iterative Method를 통해 최종 예측값(p_pred)을 산출합니다.
    """
    x_norm, p_true_phys_tensor = val_data
    batch_size = x_norm.size(0)
    iterations = 10 # 수렴할 때까지 반복
    
    # 초기값: GT 평균에서 시작 (Uninformative)
    p_mean = p_true_phys_tensor.mean(dim=0, keepdim=True)
    p_curr_phys = p_mean.repeat(batch_size, 1)
    p_curr_norm = normalizer.normalize_params(p_curr_phys)
    
    print(f"[Info] Running Inference for {iterations} steps...")
    
    with torch.no_grad():
        for _ in range(iterations):
            # Step A: P -> Y (Hidden variable prediction)
            y_hat = f_theta(x_norm, p_curr_norm)
            # Step B: Y -> P (Parameter update)
            p_next_norm = g_phi(x_norm, y_hat)
            p_curr_norm = p_next_norm
            
    # 최종 결과 Denormalize
    p_pred_phys = normalizer.denormalize_params(p_curr_norm)
    return p_pred_phys.cpu().numpy(), p_true_phys_tensor.cpu().numpy()

# --- 3. 핵심 시각화 함수: Manifold Collapse ---
def plot_manifold_collapse(p_pred, p_true, save_dir):
    """
    SI * Sigma = Constant 곡선으로의 붕괴 현상을 진단하는 3-Panel Plot
    """
    print(f"[Info] Generating Manifold Collapse Visualization at {save_dir}")
    
    # 파라미터 인덱스 가정 (0: SI, 1: Sigma)
    # 만약 순서가 다르다면 여기서 수정하세요.
    idx_si, idx_sigma = 0, 1
    
    si_true, sigma_true = p_true[:, idx_si], p_true[:, idx_si+1]
    si_pred, sigma_pred = p_pred[:, idx_si], p_pred[:, idx_si+1]
    
    # 곱(Product) 계산: K = SI * Sigma
    k_true = si_true * sigma_true
    k_pred = si_pred * sigma_pred
    
    # 스타일 설정
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 6))
    
    # --- Panel 1: Log-Log Joint Distribution ---
    # 목적: 쌍곡선(SI*Sigma=C)이 로그 스케일에서 직선(기울기 -1)으로 나타남을 이용
    ax1 = fig.add_subplot(131)
    
    # Ground Truth (Grey Background)
    ax1.scatter(si_true, sigma_true, c='grey', alpha=0.15, s=15, label='Ground Truth (Distribution)')
    
    # Prediction (Red Points)
    # 붕괴가 일어났다면 이 점들이 얇은 선 형태로 나타날 것임
    ax1.scatter(si_pred, sigma_pred, c='crimson', alpha=0.6, s=15, label='Prediction (Collapsed)')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 참조선 (Reference Line for SI*Sigma = Constant)
    # 데이터의 Median Product 값을 기준으로 기울기 -1인 선 그리기
    ref_k = np.median(k_true)
    x_range = np.logspace(np.log10(si_true.min()), np.log10(si_true.max()), 100)
    y_ref = ref_k / x_range
    ax1.plot(x_range, y_ref, 'k--', linewidth=1.5, alpha=0.7, label=f'Ref: $SI \\cdot \\sigma \\approx$ Const')
    
    #ax1.set_title("1. Manifold Collapse (Log-Log Scale)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("$S_I$ [Log]", fontsize=12)
    ax1.set_ylabel("$\sigma$ [Log]", fontsize=12)
    ax1.legend(loc='upper right')
    
    # --- Panel 2: Product (Stiff Direction) Distribution ---
    # 목적: 개별 값은 못 맞춰도, 곱(K)은 정확히 맞추고 있음을 증명
    ax2 = fig.add_subplot(132)
    
    sns.kdeplot(k_true, ax=ax2, color='grey', fill=True, alpha=0.3, linewidth=2, label='Ground Truth ($K_{true}$)')
    sns.kdeplot(k_pred, ax=ax2, color='crimson', linewidth=2, label='Prediction ($K_{pred}$)')
    
    # 에러율 계산 (MAPE)
    mape_k = np.mean(np.abs((k_pred - k_true) / k_true)) * 100
    
    #ax2.set_title(f"2. Product Consistency ($K = S_I \\cdot \\sigma$)\nMAPE: {mape_k:.2f}% (Very Low Error)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("$mDI$", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.legend()
    
    # --- Panel 3: Correlation Comparison ---
    # 목적: Sloppy Direction(개별) vs Stiff Direction(곱)의 예측 성능 대조
    ax3 = fig.add_subplot(133)
    
    # 0~1로 정규화하여 한 그래프에 그리기 위함
    def normalize_minmax(v): return (v - v.min()) / (v.max() - v.min())
    
    # Individual Param (SI) - Sloppy
    ax3.scatter(normalize_minmax(si_true), normalize_minmax(si_pred), 
                c='steelblue', alpha=0.3, s=10, label='$S_I$')
    
    # Product Param (K) - Stiff
    ax3.scatter(normalize_minmax(k_true), normalize_minmax(k_pred), 
                c='crimson', alpha=0.3, s=10, label='$mDI$')
    
    # y=x 대각선
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    #ax3.set_title("3. Correlation Analysis", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Ground Truth (Normalized)", fontsize=12)
    ax3.set_ylabel("Prediction (Normalized)", fontsize=12)
    ax3.legend()
    
    plt.tight_layout()
    save_path = save_dir / "manifold_collapse_evidence.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Plot saved to: {save_path}")

def plot_symmetric_collapse(p_pred, p_true, save_dir):
    """
    [수정됨] Symmetric Collapse (대칭적 붕괴) 및 Log Correlation 시각화
    1. Log-Log Joint Plot: 기울기 1인 직선(y=x) 위로 붕괴되는지 확인
    2. Product Distribution: 곱(K)의 일치도 확인
    3. Log Correlation: 개별 변수 vs 곱 변수의 로그 스케일 상관관계 비교
    """
    print(f"[Info] Generating Symmetric Collapse Visualization (V2) at {save_dir}")
    
    # 인덱스 설정 (0: SI, 1: Sigma)
    idx_si, idx_sigma = 0, 1
    
    si_true, sigma_true = p_true[:, idx_si], p_true[:, idx_si+1]
    si_pred, sigma_pred = p_pred[:, idx_si], p_pred[:, idx_si+1]
    
    # 곱(Product) 계산
    k_true = si_true * sigma_true
    k_pred = si_pred * sigma_pred
    
    sns.set_style("ticks")
    fig = plt.figure(figsize=(20, 6))
    
    # --- Panel 1: Log-Log Joint Distribution (Symmetric Collapse Check) ---
    ax1 = fig.add_subplot(131)
    
    # GT (Grey)
    ax1.scatter(si_true, sigma_true, c='grey', alpha=0.2, s=20, label='Ground Truth')
    
    # Pred (Red) - 기울기 1 확인용
    ax1.scatter(si_pred, sigma_pred, c='crimson', alpha=0.6, s=15, label='Prediction')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 범위 설정 (GT와 Pred를 모두 포함하도록)
    all_vals = np.concatenate([si_true, sigma_true, si_pred, sigma_pred])
    vmin, vmax = all_vals.min(), all_vals.max()
    
    # 가이드라인 1: y = x (Symmetric Line, 기울기 1) -> 모델이 여기로 붕괴되었는지 확인
    ax1.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='y=x (Symmetric Mean)')
    
    # 가이드라인 2: SI * Sigma = Median (Manifold Line, 기울기 -1) -> 데이터가 퍼져있는 방향
    # 시각적 참조용으로 하나만 그림
    ref_k = np.median(k_true)
    x_range = np.logspace(np.log10(vmin), np.log10(vmax), 100)
    y_ref = ref_k / x_range
    ax1.plot(x_range, y_ref, 'k:', linewidth=1, alpha=0.5, label='SI*Sigma=Const')
    
    #ax1.set_title("1. Symmetric Collapse (Log-Log)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("\$S_I$ [Log]", fontsize=12)
    ax1.set_ylabel("$\sigma$ [Log]", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, which="major", ls="-", alpha=0.2)
    
    # --- Panel 2: Product Distribution (Stiff Direction) ---
    ax2 = fig.add_subplot(132)
    
    # Log 스케일로 분포 비교 (Log-Normal이므로 Log 취하면 정규분포처럼 보임)
    sns.kdeplot(np.log10(k_true), ax=ax2, color='grey', fill=True, alpha=0.3, linewidth=2, label='Log($K_{true}$)')
    sns.kdeplot(np.log10(k_pred), ax=ax2, color='crimson', linewidth=2, label='Log($K_{pred}$)')
    
    mape_k = np.mean(np.abs((k_pred - k_true) / k_true)) * 100
    
    #ax2.set_title(f"2. Product Consistency (Log Scale)\nMAPE: {mape_k:.2f}%", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Log Product ($\log_{10} K$)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    # --- Panel 3: Log-Correlation Analysis (Linearity Check) ---
    ax3 = fig.add_subplot(133)
    
    # Log를 취해서 Scatter를 그림 -> 곡선이 펴지는지 확인
    log_si_true = np.log10(si_true)
    log_si_pred = np.log10(si_pred)
    
    log_k_true = np.log10(k_true)
    log_k_pred = np.log10(k_pred)
    
    # 1. Individual (SI) - Sloppy
    ax3.scatter(log_si_true, log_si_pred, c='steelblue', alpha=0.2, s=15, label='$S_I$')
    
    # 2. Product (K) - Stiff
    ax3.scatter(log_k_true, log_k_pred, c='crimson', alpha=0.2, s=15, label='$mDI$')
    
    # y=x Reference
    min_val = min(log_si_true.min(), log_k_true.min())
    max_val = max(log_si_true.max(), log_k_true.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    
    #ax3.set_title("3. Log-Correlation Analysis", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Log Ground Truth ($\log_{10}$ GT)", fontsize=12)
    ax3.set_ylabel("Log Prediction ($\log_{10}$ Pred)", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = save_dir / "symmetric_collapse_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Plot saved to: {save_path}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name query")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    try:
        # 1. Load Context
        f_theta, g_phi, val_data, normalizer, config, target_dir = load_experiment_context(args.exp_name, args.device)
        
        # 2. Run Inference
        p_pred, p_true = run_inference(f_theta, g_phi, val_data, normalizer)
        
        # 3. Visualize
        plot_manifold_collapse(p_pred, p_true, target_dir)
        plot_symmetric_collapse(p_pred, p_true, target_dir)
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()