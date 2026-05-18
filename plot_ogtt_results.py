import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from scipy.stats import pearsonr
from scipy.integrate import solve_ivp
import pandas as pd

from config import Config
from src.models import ParameterEstimator, HiddenVarPredictor, SingleNetworkBaseline
from systems.ogtt_simul import OgttSimul
from src.utils import Normalizer
from src.data_loader import DataGenerator, RealOGTTDataLoader
from tools.interactive_file_selector import interactive_file_selector


def load_experiment_context_from_paths(config_path, baseline_path, Hnet_path, Pnet_path, device_override="cuda"):
    """사용자가 명시적으로 선택한 파일 경로들을 바탕으로 실험 컨텍스트를 로드합니다."""
    config = Config()
    
    # --- 1. Config 로드 ---
    print(f"[Info] Loading Config from: {config_path}")
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
        for k, v in saved_config.items():
            if not k.startswith('__'):
                setattr(config, k, v)
    
    if device_override: 
        config.DEVICE = device_override if torch.cuda.is_available() else "cpu"
        
    system = OgttSimul()
    
    # --- 2. Simulation 데이터 로드 ---
    print(f"[Info] Generating/Loading Simulation Data...")
    data_gen = DataGenerator(system, config)
    obs_data, hid_data, params_data, _ = data_gen.generate_data()
    
    scale_obs = np.percentile(np.abs(obs_data), 99.9)
    scale_hid = np.percentile(np.abs(hid_data), 99.9)
    p_min, p_max = np.min(params_data, axis=0), np.max(params_data, axis=0)
    
    normalizer = Normalizer(system, config.DEVICE, state_scales=[scale_obs*1.2, scale_hid*1.2], 
                            param_bounds=(p_min/1.2, p_max*1.2), use_log_params=getattr(config, 'USE_LOG_PARAMS', True))
    
    test_size = int(len(obs_data) * config.TEST_SPLIT)
    sim_x = obs_data[-test_size:]
    sim_p = params_data[-test_size:]
    sim_x_norm = normalizer.normalize_inputs(torch.tensor(sim_x, dtype=torch.float32).view(test_size, -1).to(config.DEVICE), 'observed')
    sim_p_tensor = torch.tensor(sim_p, dtype=torch.float32).to(config.DEVICE)
    
    # --- 3. Real 데이터 로드 ---
    print(f"[Info] Loading Real Clinical Data...")
    try:
        real_loader = RealOGTTDataLoader(file_path='data/clean_sumner_n_612.xlsx', config=config, split_file='data/data_split_indices.json')
        obs_real, _, params_real, _ = real_loader.load_data()
        with open('data/data_split_indices.json', 'r') as f:
            test_indices = json.load(f)['test_indices']
        real_x = obs_real[test_indices]
        real_p = params_real[test_indices]
        real_x_norm = normalizer.normalize_inputs(torch.tensor(real_x, dtype=torch.float32).view(len(real_x), -1).to(config.DEVICE), 'observed')
        real_p_tensor = torch.tensor(real_p, dtype=torch.float32).to(config.DEVICE)
        has_real_data = True
    except Exception as e:
        print(f"[Warning] Real data loading failed: {e}")
        has_real_data, real_x_norm, real_p_tensor = False, None, None

    # --- 4. 모델 초기화 및 가중치 로드 ---
    print(f"[Info] Initializing Models and Loading Weights...")
    flat_x_dim = sim_x_norm.shape[1]
    flat_y_dim = hid_data.shape[1] * hid_data.shape[2]
    num_params = sim_p_tensor.shape[1]

    # Baseline은 Spectral Norm 없이, Ours는 적용해서 껍데기 생성
    baseline_model = SingleNetworkBaseline(flat_x_dim, num_params, model_config=config.MODEL_CONFIG['param_net'], use_spectral_norm=False).to(config.DEVICE)
    Hnet = HiddenVarPredictor(flat_x_dim, flat_y_dim, num_params, model_config=config.MODEL_CONFIG['hidden_net'], use_spectral_norm=True).to(config.DEVICE)
    Pnet = ParameterEstimator(flat_x_dim, flat_y_dim, num_params, model_config=config.MODEL_CONFIG['param_net'], use_spectral_norm=True).to(config.DEVICE)

    # 명시된 경로에서 정확하게 가중치 탑재
    def load_weight_safe(model, path, possible_keys):
        ckpt = torch.load(path, map_location=config.DEVICE)
        
        # 만약 ckpt가 epoch, optimizer 등이 포함된 '보따리(Dict)'라면
        if isinstance(ckpt, dict):
            for key in possible_keys:
                if key in ckpt:
                    model.load_state_dict(ckpt[key])
                    return
            
            # 보따리인데 예상한 키가 없다면, 혹시 이름이 바뀐 state_dict가 있는지 탐색
            for k, v in ckpt.items():
                if 'state_dict' in k and 'optimizer' not in k:
                    model.load_state_dict(v)
                    return
                    
        # 순수한 가중치 파일이거나, 위의 조건에 걸리지 않았다면 통째로 로드 시도
        model.load_state_dict(ckpt)
    
    load_weight_safe(baseline_model, baseline_path, ['baseline_state_dict'])
    load_weight_safe(Hnet, Hnet_path, ['Hnet_state_dict'])
    load_weight_safe(Pnet, Pnet_path, ['Pnet_state_dict'])

    Hnet.eval(); Pnet.eval(); baseline_model.eval()
    
    # 결과를 저장할 타겟 폴더는 Hnet.pth가 위치한 폴더로 지정
    target_dir = Path(Hnet_path).parent 
    
    return Hnet, Pnet, baseline_model, (sim_x_norm, sim_p_tensor), (real_x_norm, real_p_tensor), normalizer, config, target_dir, system

def run_inference(Hnet, Pnet, baseline_model, x_norm, p_true_tensor, normalizer, iterations=10):
    """Baseline 및 Ours 추론 수행 (1초 내외 소요)"""
    batch_size = x_norm.size(0)
    p_mean = p_true_tensor.mean(dim=0, keepdim=True)
    p_curr_norm = normalizer.normalize_params(p_mean.repeat(batch_size, 1))
    
    with torch.no_grad():
        # Baseline Inference
        p_base_norm = baseline_model(x_norm)
        # Ours Iterative Inference
        for _ in range(iterations):
            y_hat = Hnet(x_norm, p_curr_norm)
            p_curr_norm = Pnet(x_norm, y_hat)
            
    p_ours_phys = normalizer.denormalize_params(p_curr_norm).cpu().numpy()
    p_base_phys = normalizer.denormalize_params(p_base_norm).cpu().numpy()
    return p_true_tensor.cpu().numpy(), p_ours_phys, p_base_phys

def plot_symmetric_collapse(p_true, p_ours, p_base, save_dir, prefix="sim"):
    print(f"[Info] Generating Symmetric Collapse Visualization ({prefix})")
    idx_si, idx_sigma = 0, 1
    si_true, sigma_true = p_true[:, idx_si], p_true[:, idx_sigma]
    si_ours, sigma_ours = p_ours[:, idx_si], p_ours[:, idx_sigma]
    si_base, sigma_base = p_base[:, idx_si], p_base[:, idx_sigma]
    k_true, k_ours, k_base = si_true * sigma_true, si_ours * sigma_ours, si_base * sigma_base
    
    sns.set_style("ticks")
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Log-Log Joint Distribution
    ax1 = fig.add_subplot(121)
    ax1.scatter(si_true, sigma_true, c='grey', alpha=0.2, s=20, label='Ground Truth')
    ax1.scatter(si_base, sigma_base, c='crimson', alpha=0.4, s=15, marker='x', label='Baseline')
    ax1.scatter(si_ours, sigma_ours, c='steelblue', alpha=0.4, s=15, edgecolors='white', lw=0.5, label='Ours')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    vmin, vmax = min(si_true.min(), sigma_true.min()), max(si_true.max(), sigma_true.max())
    ax1.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='y=x (Symmetric Line)')
    ax1.set_xlabel("$S_I$ [Log]", fontsize=12); ax1.set_ylabel("$\sigma$ [Log]", fontsize=12)
    ax1.set_title("Log-Log Joint Distribution", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.2)
    
    # 2. Product Distribution
    ax2 = fig.add_subplot(122)
    sns.kdeplot(np.log10(k_true), ax=ax2, color='grey', fill=True, alpha=0.3, linewidth=2, label='Log($K_{true}$)')
    sns.kdeplot(np.log10(k_base), ax=ax2, color='crimson', linestyle='--', linewidth=2, label='Log($K_{base}$)')
    sns.kdeplot(np.log10(k_ours), ax=ax2, color='steelblue', linewidth=2, label='Log($K_{ours}$)')
    ax2.set_xlabel("Log Product ($\log_{10} K$)", fontsize=12); ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("mDI_woI ($S_I \cdot \sigma$)", fontsize=14, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.2)
    
    
    plt.tight_layout()
    save_path = save_dir / f"{prefix}_symmetric_collapse.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Success] Plot saved to: {save_path}")

def plot_ogtt_trajectories(p_true, p_ours, p_base, system, save_dir, prefix="sim"):
    print(f"[Info] Generating OGTT Trajectories ({prefix})")
    
    # Baseline이 가장 크게 붕괴된(오차가 큰) 샘플 인덱스 추출
    idx = np.argmax(np.abs(p_true[:, 0] - p_base[:, 0])) 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t_span = (0, 120)
    t_eval = np.linspace(0, 120, 200)
    
    # ---------------------------------------------------------
    # [수정된 부분] 4D 초기값 (G, I, N5, N6) 동적 생성 로직
    # ---------------------------------------------------------
    # 공복 상태의 기준 혈당(G)과 인슐린(I) 설정 (단위: mg/dL, uU/mL 등)
    G0_nominal = 90.0 
    I0_nominal = 15.0 

    def get_steady_state_y0(params):
        """파라미터에 맞는 N5, N6의 정상 상태(Steady-state)를 계산하여 4D 초기값을 반환"""
        system.model.theta['si'] = params[0]
        system.model.theta['sigma'] = params[1]
        
        # G0를 기준으로 내부 지연 변수 N5, N6의 초기 평형점 계산
        n5_ss, n6_ss = system.model.find_steady_state_N(G0_nominal)
        return [G0_nominal, I0_nominal, n5_ss, n6_ss]

    # 각 모델이 예측한 파라미터를 기반으로 생리학적으로 올바른 초기값 세팅
    y0_true = get_steady_state_y0(p_true[idx])
    y0_base = get_steady_state_y0(p_base[idx])
    y0_ours = get_steady_state_y0(p_ours[idx])

    def ogtt_ode(t, y, *params_tuple):
        return system.ode_func(t, y, list(params_tuple))

    # ODE 풀이 (Stiff 문제이므로 BDF나 LSODA, 또는 RK45 사용)
    sol_true = solve_ivp(ogtt_ode, t_span, y0_true, args=tuple(p_true[idx]), t_eval=t_eval, method='BDF')
    sol_base = solve_ivp(ogtt_ode, t_span, y0_base, args=tuple(p_base[idx]), t_eval=t_eval, method='BDF')
    sol_ours = solve_ivp(ogtt_ode, t_span, y0_ours, args=tuple(p_ours[idx]), t_eval=t_eval, method='BDF')

    ax = axes[0]
    ax.plot(sol_true.t, sol_true.y[0], 'k-', lw=3, label='True Glucose')
    ax.plot(sol_base.t, sol_base.y[0], 'indianred', linestyle='--', lw=2, label='Baseline')
    ax.plot(sol_ours.t, sol_ours.y[0], 'steelblue', linestyle='-.', lw=2, label='Ours')
    ax.set_title("Glucose Dynamics (Observed State)", fontsize=14)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Concentration")
    ax.legend(); ax.grid(True, alpha=0.3)

  
    ax = axes[1]
    ax.plot(sol_true.t, sol_true.y[1], 'k-', lw=3, label='True Insulin (Hidden)')
    ax.plot(sol_base.t, sol_base.y[1], 'indianred', linestyle='--', lw=2, label='Baseline', alpha=0.5)
    ax.plot(sol_ours.t, sol_ours.y[1], 'steelblue', linestyle='-.', lw=2, label='Ours', alpha=0.5)
    ax.set_title("Insulin Dynamics (Hidden State)", fontsize=14)
    ax.set_xlabel("Time (min)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / f"{prefix}_trajectory_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Success] Plot saved to: {save_path}")

def evaluate_robustness(Hnet, Pnet, base_model, x_norm, p_true_tensor, normalizer, target_dir):
    print("\n\033[1;36m=== Running Robustness Stress Test with Variance Analysis ===\033[0m")
    
    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0] 
    results = []
    x_std = x_norm.std(dim=0, keepdim=True)
    
    for nl_val in noise_levels:
        nl = nl_val / 100.0
        print(f"[Testing] Noise Level: {nl_val:.1f}% ...", end="\r")
        
        torch.manual_seed(42)
        noise = torch.randn_like(x_norm) * (x_std * nl)
        x_noisy = x_norm + noise
        
        p_true_phys, p_ours_phys, p_base_phys = run_inference(Hnet, Pnet, base_model, x_noisy, p_true_tensor, normalizer)
        
        def safe_pearsonr(x, y, eps=1e-12):
            """
            부동 소수점 오차를 고려하여 분산이 거의 0인 경우를 안전하게 처리합니다.
            """
            var_x = np.var(x)
            var_y = np.var(y)
            
            # 분산이 시스템 허용 오차(eps)보다 작으면 상관관계가 없다고 판단(0.0 반환)
            if var_x < eps or var_y < eps:
                # 이 경우가 발생한다면 '매니폴드 붕괴'의 확실한 증거가 됩니다.
                return 0.0
            
            r, _ = pearsonr(x, y)
            return r if np.isfinite(r) else 0.0

        r_si_base = safe_pearsonr(p_true_phys[:, 0], p_base_phys[:, 0])
        r_sigma_base = safe_pearsonr(p_true_phys[:, 1], p_base_phys[:, 1])
        r_si_ours = safe_pearsonr(p_true_phys[:, 0], p_ours_phys[:, 0])
        r_sigma_ours = safe_pearsonr(p_true_phys[:, 1], p_ours_phys[:, 1])

        results.append({
            'noise_level': nl_val,
            'Base_SI_Pearson': r_si_base, 'Base_Sigma_Pearson': r_sigma_base,
            'Ours_SI_Pearson': r_si_ours, 'Ours_Sigma_Pearson': r_sigma_ours,
            'Var_SI_GT': np.var(p_true_phys[:, 0]), 'Var_Sigma_GT': np.var(p_true_phys[:, 1]),
            'Var_SI_Base': np.var(p_base_phys[:, 0]), 'Var_Sigma_Base': np.var(p_base_phys[:, 1]),
            'Var_SI_Ours': np.var(p_ours_phys[:, 0]), 'Var_Sigma_Ours': np.var(p_ours_phys[:, 1])
        })

    df = pd.DataFrame(results)
    df.to_csv(target_dir / "robustness_metrics_with_variance.csv", index=False)

    # --- 시각화: Pearson R(상관성)과 Variance(표현력) 동시 비교 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1 & 2: Pearson Correlation (기존과 동일)
    for i, (param, label) in enumerate(zip(['SI', 'Sigma'], ['$S_I$', '$\sigma$'])):
        ax = axes[0, i]
        ax.plot(df['noise_level'], df[f'Base_{param}_Pearson'], 'ro--', label=f'Baseline ({label})')
        ax.plot(df['noise_level'], df[f'Ours_{param}_Pearson'], 'bo-', label=f'Ours ({label})')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{label} Prediction Robustness (Pearson $r$)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Noise Level (%)"); ax.set_ylabel("Correlation ($r$)")
        ax.grid(True, alpha=0.3); ax.legend()

    # Panel 3 & 4: Variance Comparison (붕괴 여부 진단)
    for i, (param, label) in enumerate(zip(['SI', 'Sigma'], ['$S_I$', '$\sigma$'])):
        ax = axes[1, i]
        
        # 1. 시각적 가독성을 위해 GT 분산을 기준으로 데이터를 정규화하거나 로그 스케일 검토
        gt_var = df[f'Var_{param}_GT'].iloc[0]
        base_var = df[f'Var_{param}_Base']
        ours_var = df[f'Var_{param}_Ours']

        limit = 0.2
        base_var_clipped = np.clip(base_var, None, limit + 0.1)
        ours_var_clipped = np.clip(ours_var, None, limit + 0.1)

        # 3. 로그 스케일 적용 (분산이 수십 배 차이 날 때 유용)
        # 만약 로그 스케일을 쓰고 싶다면 ax.set_yscale('log')를 사용하세요.
        # 여기서는 연구자님의 요청대로 1.0 임계값 클리핑 방식을 적용합니다.
        
        ax.axhline(y=gt_var, color='grey', linestyle='-', alpha=0.5, label='Ground Truth Var')
        ax.plot(df['noise_level'], base_var_clipped, 'ro--', label='Baseline Var')
        ax.plot(df['noise_level'], ours_var_clipped, 'bo-', label='Ours Var')

        # 4. '폭발 구역' 시각화
        ax.axhspan(limit, limit + 0.05, color='red', alpha=0.1)
        ax.text(df['noise_level'].mean(), limit + 0.025, "Divergence Zone", 
                color='red', fontsize=10, ha='center', fontweight='bold')

        ax.set_ylim(0.0, limit + 0.05) # 상단을 고정하여 바닥의 Ours 분산을 보호
        ax.set_title(f"{label} Variance: Stability vs. Explosion", fontsize=14, fontweight='bold')
        ax.set_xlabel("Noise Level (%)")
        ax.set_ylabel("Variance")
        ax.grid(True, alpha=0.2)
        ax.legend()

    plt.tight_layout()
    plt.savefig(target_dir / "robustness_and_variance_analysis_final.png", dpi=300)
    plt.show()

def main():
    try:
        print("\n\033[1;33m=== [OGTT 분석기] 파일을 3번 선택해 주세요 ===\033[0m")
        base_search_dir = "./results"
        
        # 1. Config 선택
        rel_config_path = interactive_file_selector(
            prompt_msg="[1/3] 실험 설정 파일 (config.json)을 선택하세요:", 
            start_dir=base_search_dir
        )
        config_path = os.path.join(base_search_dir, rel_config_path)
        
        # 2. Baseline 선택
        rel_baseline_path = interactive_file_selector(
            prompt_msg="[2/3] 비교할 베이스라인 모델 가중치 (예: baseline_net.pth)를 선택하세요:", 
            start_dir=base_search_dir
        )
        baseline_path = os.path.join(base_search_dir, rel_baseline_path)
        
        # 3. Hnet 선택 (Pnet는 문자열 치환으로 자동 유추)
        rel_Hnet_path = interactive_file_selector(
            prompt_msg="[3/3] 제안 모델 가중치 (Hnet.pth)를 선택하세요 (Pnet.pth는 자동 로드됨):", 
            start_dir=base_search_dir
        )
        Hnet_path = os.path.join(base_search_dir, rel_Hnet_path)
        Pnet_path = Hnet_path.replace('Hnet.pth', 'Pnet.pth')
        
        
        print("\n[선택된 파일 목록]")
        print(f" 1. Config  : {config_path}")
        print(f" 2. Baseline: {baseline_path}")
        print(f" 3. HiddenNet : {Hnet_path}")
        print(f" 4. ParamNet   : {Pnet_path} (자동추론)\n")

        # 혹시 모를 파일 부재 예외처리
        for path in [config_path, baseline_path, Hnet_path, Pnet_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        # 명시적인 경로들로 실험 컨텍스트 로드
        Hnet, Pnet, base_model, sim_data, real_data, normalizer, config, target_dir, system = \
            load_experiment_context_from_paths(config_path, baseline_path, Hnet_path, Pnet_path)
        
        # --- 1. Simulation Data 분석 ---
        print("\n=== Processing Simulation Data ===")
        p_true_sim, p_ours_sim, p_base_sim = run_inference(Hnet, Pnet, base_model, sim_data[0], sim_data[1], normalizer)
        plot_symmetric_collapse(p_true_sim, p_ours_sim, p_base_sim, target_dir, prefix="sim")
        plot_ogtt_trajectories(p_true_sim, p_ours_sim, p_base_sim, system, target_dir, prefix="sim")
        
        # --- 2. Real Data 분석 ---
        if real_data[0] is not None:
            print("\n=== Processing Real Clinical Data ===")
            p_true_real, p_ours_real, p_base_real = run_inference(Hnet, Pnet, base_model, real_data[0], real_data[1], normalizer)
            plot_symmetric_collapse(p_true_real, p_ours_real, p_base_real, target_dir, prefix="real")
            plot_ogtt_trajectories(p_true_real, p_ours_real, p_base_real, system, target_dir, prefix="real")
        
        # --- 3. Robustness Stress Test ---
        evaluate_robustness(Hnet, Pnet, base_model, sim_data[0], sim_data[1], normalizer, target_dir)
        
        print(f"\n\033[1;32m[완료] 모든 분석 플롯이 '{target_dir}' 폴더에 저장되었습니다!\033[0m")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

