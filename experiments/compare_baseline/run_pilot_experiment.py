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
def save_results_to_json(filepath, sys_name, model_name, results):
    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass 

    if sys_name not in data:
        data[sys_name] = {}
    
    data[sys_name][model_name] = results
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def append_to_markdown(filepath, sys_name, model_name, results_summary, radii):
    with open(filepath, 'a', encoding='utf-8') as f:
        if os.stat(filepath).st_size == 0 or not f.tell():
             f.write(f"# Baseline Experiment Results (5% Noise Injected)\n")
             f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"### Results Table: {sys_name.upper()} - {model_name}\n")
        header = "| Method | " + " | ".join([f"Radius R={R}" for R in radii]) + " |"
        divider = "|--------|" + "|".join(["------------" for _ in radii]) + "|"
        row = f"| {model_name} | "
        
        for R in radii:
            if R in results_summary:
                sr, err = results_summary[R]['success_rate'], results_summary[R]['mean_error']
                cell = f"[{err:.3f} ({sr:.0f}%)]" if sr > 0 else "[Diverge (0%)]"
            else:
                cell = "[Incomplete]" 
            row += f"{cell} | "
            
        f.write(header + "\n")
        f.write(divider + "\n")
        f.write(row + "\n\n")


def load_ground_truth_data(sys_obj):
    t_span = sys_obj.t_span
    t_eval = sys_obj.t_points
    
    if sys_obj.name.lower() == 'sir':
        # 🚨 [수정 1] N=50 유지 & R_0=2.5 세팅 (뚜렷한 감염 폭발)
        theta_true = np.array([0.25, 0.1])
        x_hid_true_0 = np.array([2.0])  # 초기 감염자 2명 확보
        y0 = [48.0, 2.0, 0.0]           # S_0 = 48
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


def run_monte_carlo_evaluation(estimator, sys_obj, radii, n_trials=50, save_paths=None):
    print(f"\n[{sys_obj.name.upper()}] Evaluating {estimator.__class__.__name__} (N={n_trials})")
    
    t_eval, x_obs_clean, theta_true, x_hid_true_0 = load_ground_truth_data(sys_obj)
    
    # 🚨 [수정 2] 공정한 평가를 위한 2% 가우시안 노이즈 주입
    noise_rng = np.random.default_rng(2024)
    noise_level = 0.05
    noise = noise_rng.normal(0, noise_level * np.abs(x_obs_clean), x_obs_clean.shape)
    x_obs_noisy = x_obs_clean + noise
    
    # 🚨 [수정 3] 물리적 제약 강제: 음수 및 총인구수 초과 방지
    if sys_obj.name.lower() == 'sir':
        N_val = sum([val[0] for val in sys_obj.initial_conditions]) # 50.0
        x_obs_noisy = np.clip(x_obs_noisy, 0.0, N_val)
    else:
        x_obs_noisy = np.maximum(0.0, x_obs_noisy)
    
    results_summary = {}
    model_name = estimator.__class__.__name__.replace("Estimator", "")
    
    for R in radii:
        rng = np.random.default_rng(42)
        success_count = 0
        errors = []
        exec_times = []
        all_trajectories = [] 
        
        print(f"  R={R:3.1f} | ", end="", flush=True)
        
        for i in range(n_trials):
            theta_init = perturb_initial_values(theta_true, R, rng=rng)
            x_hid_init_0 = perturb_initial_values(x_hid_true_0, R, rng=rng)
            
            try:
                # 노이즈가 낀 관측치로 피팅을 수행합니다.
                theta_hat, _, exec_time, p_history = estimator.fit(t_eval, x_obs_noisy, theta_init, x_hid_init_0)
                exec_times.append(exec_time)
                
                error = calculate_relative_error(theta_hat, theta_true)
                
                # 🚨 [수정 4] 치명적 버그 해결: 성공 여부와 무관하게 모든 오차를 기록
                errors.append(error) 
                
                if evaluate_success(error, threshold=0.05):
                    success_count += 1
                
                all_trajectories.append([p.tolist() for p in p_history])
                        
            except Exception as e:
                # 에러로 터진 경우 시작점만 궤적으로 남겨서 플롯 누락 방지
                all_trajectories.append([theta_init.tolist()])
                pass
                
        success_rate = (success_count / n_trials) * 100
        # 에러 리스트가 정상적으로 채워졌으므로 이제 정확한 평균 에러가 도출됩니다.
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
        
        print(f"Success: {success_rate:5.1f}% | Mean Err: {mean_error:.4f} ($\pm$ {std_error:.4f}) | Avg Time: {mean_time:.3f}s")
        
        if save_paths and 'json' in save_paths:
            save_results_to_json(save_paths['json'], sys_obj.name, model_name, results_summary)
            
    if save_paths and 'md' in save_paths:
        append_to_markdown(save_paths['md'], sys_obj.name, model_name, results_summary, radii)
        
    return results_summary

def run_baseline_experiment():
    test_radii = [0.1, 0.5, 1.0, 2.0]
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
        print(f"\n{'='*50}\nSYSTEM: {sys_obj.name.upper()} (WITH 5% NOISE)\n{'='*50}")
        
        estimators = {
            "NLLS (LM)": NLLSEstimator(sys_obj),
            "Adjoint (L-BFGS)": AdjointLBFGSEstimator(sys_obj),
            #"MCMC (MH)": MCMCEstimator(sys_obj, n_iters=1000)
        }
        
        for name, model in estimators.items():
            results = run_monte_carlo_evaluation(model, sys_obj, test_radii, n_trials=200, save_paths=save_paths)
            
if __name__ == "__main__":
    run_baseline_experiment()