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
    """결과를 JSON 파일에 누적(업데이트)하여 저장합니다."""
    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # 파일이 비어있거나 깨졌으면 새로 시작

    if sys_name not in data:
        data[sys_name] = {}
    
    data[sys_name][model_name] = results
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def append_to_markdown(filepath, sys_name, model_name, results_summary, radii):
    """결과를 Markdown 테이블 형태로 파일에 이어 씁니다."""
    with open(filepath, 'a', encoding='utf-8') as f:
        # 파일이 비어있으면 시스템 헤더 작성
        if os.stat(filepath).st_size == 0 or not f.tell():
             f.write(f"# Baseline Experiment Results\n")
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
                cell = "[Incomplete]" # 중간에 중단된 경우 표시
            row += f"{cell} | "
            
        f.write(header + "\n")
        f.write(divider + "\n")
        f.write(row + "\n\n")


def load_ground_truth_data(sys_obj):
    t_span = sys_obj.t_span
    t_eval = sys_obj.t_points
    if sys_obj.name.lower() == 'sir':
        theta_true = np.array([0.15, 0.1])
        x_hid_true_0 = np.array([1.0])
        y0 = [49.0, 1.0, 0.0]
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
    t_eval, x_obs, theta_true, x_hid_true_0 = load_ground_truth_data(sys_obj)
    
    # 전역 seed 오염 방지를 위해 local generator 사용
    rng = np.random.default_rng(42) 
    results_summary = {}
    model_name = estimator.__class__.__name__.replace("Estimator", "")
    
    for R in radii:
        success_count = 0
        errors = []
        print(f"  R={R:3.1f} | ", end="", flush=True)
        start_time = time.time()
        
        for i in range(n_trials):
            # Local rng를 사용하여 다른 모델 평가에 영향을 주지 않음
            theta_init = perturb_initial_values(theta_true, R, rng=rng)
            x_hid_init_0 = perturb_initial_values(x_hid_true_0, R, rng=rng)
            
            try:
                theta_hat, _, _ = estimator.fit(t_eval, x_obs, theta_init, x_hid_init_0)
                error = calculate_relative_error(theta_hat, theta_true)
                
                if evaluate_success(error, threshold=0.05):
                    success_count += 1
                    errors.append(error)
            except Exception as e:
                # [디버깅] 여기서 조용히 넘어가지 말고 무슨 에러인지 출력하도록 수정!
                print(f"[{estimator.__class__.__name__} Failed]: {e}")
                
        elapsed_time = time.time() - start_time
        success_rate = (success_count / n_trials) * 100
        mean_error = np.mean(errors) if len(errors) > 0 else float('inf')
        
        results_summary[R] = {"success_rate": success_rate, "mean_error": mean_error}
        print(f"Success: {success_rate:5.1f}% | Mean Err: {mean_error:.4f} | Time: {elapsed_time:.1f}s")
        
        # [핵심] 하나의 Radius 처리가 끝날 때마다 JSON 파일 업데이트
        if save_paths and 'json' in save_paths:
            save_results_to_json(save_paths['json'], sys_obj.name, model_name, results_summary)
            
    # 전체 Radius 평가가 끝나면 Markdown 테이블에 추가
    if save_paths and 'md' in save_paths:
        append_to_markdown(save_paths['md'], sys_obj.name, model_name, results_summary, radii)
        
    return results_summary

def run_baseline_experiment():
    test_radii = [0.1, 0.5, 1.0, 2.0]
    systems = [
        Sir(), 
        LotkaVolterra()
        ]
    
    # 저장 경로 설정 (프로젝트 루트의 results 폴더 내)
    results_dir = os.path.join(project_root, 'results', 'baselines')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_paths = {
        'json': os.path.join(results_dir, f'traditional_baselines_{timestamp}.json'),
        'md': os.path.join(results_dir, f'traditional_baselines_{timestamp}.md')
    }
    
    print(f"Results will be live-saved to: \n - {save_paths['json']}\n - {save_paths['md']}\n")

    for sys_obj in systems:
        print(f"\n{'='*50}\nSYSTEM: {sys_obj.name.upper()}\n{'='*50}")
        
        estimators = {
            "NLLS (LM)": NLLSEstimator(sys_obj),
            "Adjoint (L-BFGS)": AdjointLBFGSEstimator(sys_obj),
            "MCMC (MH)": MCMCEstimator(sys_obj, n_iters=1000)
        }
        
        for name, model in estimators.items():
            results = run_monte_carlo_evaluation(model, sys_obj, test_radii, n_trials=50, save_paths=save_paths)
            
if __name__ == "__main__":
    run_baseline_experiment()
    