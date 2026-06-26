import sys
import os
import time
import json
import warnings
import numpy as np

# 경로 주입 블록 (절대 경로 설정)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scipy.integrate import solve_ivp
from systems.sir import Sir
from systems.lotka_volterra import LotkaVolterra

# 실험 코어 로직 및 Estimator 임포트
from experiments.compare_baseline.core import (
    perturb_initial_values, 
    calculate_relative_error, 
    evaluate_success
)
from experiments.compare_baseline.estimators.nlls import NLLSEstimator
from experiments.compare_baseline.estimators.mcmc import MCMCEstimator
from experiments.compare_baseline.estimators.adjoint import AdjointLBFGSEstimator

# --- 저장 유틸리티 함수 ---
def save_results_to_json(filepath, sys_name, regime, model_name, results):
    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass 

    if sys_name not in data:
        data[sys_name] = {}
    if regime not in data[sys_name]:
        data[sys_name][regime] = {}
    
    data[sys_name][regime][model_name] = results
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def append_to_markdown(filepath, sys_name, regime, model_name, results_summary, sigmas):
    with open(filepath, 'a', encoding='utf-8') as f:
        if os.stat(filepath).st_size == 0 or not f.tell():
             f.write(f"# Baseline Experiment Results (5% Noise Injected)\n")
             f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"### Results Table: {sys_name.upper()} | Regime: {regime.upper()} | Model: {model_name}\n")
        header = "| Method | " + " | ".join([f"Sigma = {sigma}" for sigma in sigmas]) + " |"
        divider = "|--------|" + "|".join(["------------" for _ in sigmas]) + "|"
        row = f"| {model_name} | "
        
        for sigma in sigmas:
            # dict key가 string으로 저장될 수도 있으므로 float 처리
            key = sigma if sigma in results_summary else str(sigma)
            if key in results_summary:
                sr, err = results_summary[key]['success_rate'], results_summary[key]['mean_error']
                cell = f"[{err:.3f} ({sr:.0f}%)]" if sr > 0 else "[Diverge (0%)]"
            else:
                cell = "[Incomplete]" 
            row += f"{cell} | "
            
        f.write(header + "\n")
        f.write(divider + "\n")
        f.write(row + "\n\n")


def load_ground_truth_data(sys_obj, regime="easy"):
    """
    논문의 Narrative 강화를 위해 3가지 난이도(Regime)의 Ground Truth를 제공합니다.
    """
    t_span = sys_obj.t_span
    t_eval = sys_obj.t_points # 이미 [0, 30, 60, 90] 등으로 극도로 희소함 (Sparse)
    
    if sys_obj.name.lower() == 'sir':
        if regime == "easy":
            # [Regime 1: Peak Aligned] 완만한 확산으로 30~60 구간에서 관측 변화가 뚜렷함
            theta_true = np.array([0.25, 0.1])
            x_hid_true_0 = np.array([10.0])  # I(0) = 2
            y0 = [40.0, 10.0, 0.0]           # S(0) = 48, R(0) = 0
            
        elif regime == "hard":
            # [Regime 2: Hidden Peak] 빠른 전파와 회복. t=30 이전에 Peak가 지나가버려 정보 손실 발생
            theta_true = np.array([0.45, 0.25])
            x_hid_true_0 = np.array([10.0])  # 빠른 확산을 위해 I(0) = 5
            y0 = [40.0, 10.0, 0.0]
            
        elif regime == "ill-posed":
            # [Regime 3: Slow Decay] R0가 1 근처여서 곡선이 평탄함. 파라미터 간 구별이 수학적으로 어려움
            theta_true = np.array([0.15, 0.14])
            x_hid_true_0 = np.array([10.0])
            y0 = [40.0, 10.0, 0.0]
            
        else:
            raise ValueError(f"Unknown regime: {regime}")
            
    elif sys_obj.name.lower() == 'lotka_volterra':
        theta_true = np.array([0.8, 0.5, 0.4, 0.8])
        x_hid_true_0 = np.array([3.0])
        y0 = [10.0, 3.0]
    else:
        raise ValueError(f"Unknown system: {sys_obj.name}")
    
    sol = solve_ivp(
        fun=sys_obj.ode_func, 
        t_span=t_span, 
        y0=y0, 
        t_eval=t_eval, 
        args=(theta_true,)
    )
    x_obs = sol.y[sys_obj.observed_var_idx].reshape(-1, 1)
    
    return t_eval, x_obs, theta_true, x_hid_true_0


def run_monte_carlo_evaluation(
    estimator, 
    sys_obj, 
    test_radii, 
    regime="easy", 
    n_trials=50, 
    save_paths=None):
    print(f"\n[{sys_obj.name.upper()} | REGIME: {regime.upper()}] Evaluating {estimator.__class__.__name__} (N={n_trials})")
    
    t_eval, x_obs_clean, theta_true, x_hid_true_0 = load_ground_truth_data(sys_obj, regime=regime)
    
    # 공정한 평가를 위한 5% 가우시안 노이즈 주입
    noise_rng = np.random.default_rng(2024)
    noise_level = 0.05
    noise = noise_rng.normal(0, noise_level * np.abs(x_obs_clean), x_obs_clean.shape)
    x_obs_noisy = x_obs_clean + noise
    
    # 물리적 제약 강제: 음수 및 총인구수 초과 방지
    if sys_obj.name.lower() == 'sir':
        N_val = 50 # 총 인구수
        x_obs_noisy = np.clip(x_obs_noisy, 0.0, N_val)
    else:
        x_obs_noisy = np.maximum(0.0, x_obs_noisy)
    
    results_summary = {}
    model_name = estimator.__class__.__name__.replace("Estimator", "")
    
    for R in test_radii:
        rng = np.random.default_rng(42)
        success_count = 0
        errors = []
        exec_times = []
        all_trajectories = [] 
        
        print(f"  Radius R={R:3.1f} | ", end="", flush=True)
        
        for i in range(n_trials):
            # 1. 파라미터(theta)는 균등 상대오차 R을 주어 섭동시킴
            theta_init = perturb_initial_values(theta_true, R, rng=rng)
            
            # 🚨 2. 은닉 상태(x_hid)는 관측치로부터 '결정론적'으로 유도함 (물리적 제약 완벽 반영)
            if sys_obj.name.lower() == 'sir':
                S0_obs = float(x_obs_noisy[0, 0])
                # I0 = N - S0 (R0 = 0 가정)
                I0_inferred = max(0.1, N_val - S0_obs)
                x_hid_init_0 = np.array([I0_inferred])
            else:
                # 다른 시스템의 경우 기존 방식 유지
                x_hid_init_0 = perturb_initial_values(x_hid_true_0, R, rng=rng)
            
            try:
                theta_hat, _, exec_time, p_history = estimator.fit(t_eval, x_obs_noisy, theta_init, x_hid_init_0)
                exec_times.append(exec_time)
                
                error = calculate_relative_error(theta_hat, theta_true)
                errors.append(error) 
                
                # evaluate_success는 이제 core.py의 기본값(0.25)을 따름
                if evaluate_success(error):
                    success_count += 1
                
                all_trajectories.append([p.tolist() for p in p_history])
                        
            except Exception as e:
                all_trajectories.append([theta_init.tolist()])
                pass
                
        success_rate = (success_count / n_trials) * 100
        mean_error = float(np.mean(errors)) if len(errors) > 0 else float('inf')
        std_error = float(np.std(errors)) if len(errors) > 0 else float('inf')
        mean_time = np.mean(exec_times) if len(exec_times) > 0 else 0.0
        
        results_summary[R] = {
            "success_rate": success_rate, 
            "mean_error": mean_error,
            "std_error": std_error,
            "mean_exec_time_sec": mean_time,
            "all_trajectories": all_trajectories
        }
        
        print(f"Success: {success_rate:5.1f}% | Mean Rel_Err: {mean_error:.4f} ($\pm$ {std_error:.4f}) | Avg Time: {mean_time:.3f}s")
        
        if save_paths and 'json' in save_paths:
            save_results_to_json(save_paths['json'], sys_obj.name, regime, model_name, results_summary)
            
    if save_paths and 'md' in save_paths:
        append_to_markdown(save_paths['md'], sys_obj.name, regime, model_name, results_summary, test_radii)
        
    return results_summary

def run_baseline_experiment():
    # R(radius) 대신 Log-Normal의 Sigma 스케일 사용
    test_radii = [0.1, 0.5, 1.0, 2.0]
    regimes = ["easy", "hard", "ill-posed"]
    systems = [Sir()]
    
    results_dir = os.path.join(project_root, 'results', 'baselines')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_paths = {
        'json': os.path.join(results_dir, f'traditional_baselines_NOISY_{timestamp}.json'),
        'md': os.path.join(results_dir, f'traditional_baselines_NOISY_{timestamp}.md')
    }
    
    print(f"Results will be live-saved to: \n - {save_paths['json']}\n - {save_paths['md']}\n")

    for sys_obj in systems:
        print(f"\n{'='*60}\nSYSTEM: {sys_obj.name.upper()} (WITH 5% NOISE)\n{'='*60}")
        
        estimators = {
            "NLLS (LM)": NLLSEstimator(sys_obj),
            "Adjoint (L-BFGS)": AdjointLBFGSEstimator(sys_obj),
        }
        
        # 각 Regime(난이도)별로 평가 반복
        for regime in regimes:
            for name, model in estimators.items():
                results = run_monte_carlo_evaluation(
                    estimator=model, 
                    sys_obj=sys_obj, 
                    test_radii=test_radii, 
                    regime=regime, 
                    n_trials=200, 
                    save_paths=save_paths
                )
            
if __name__ == "__main__":
    run_baseline_experiment()