import sys
import os
import time
import json
import numpy as np
import torch
from scipy.integrate import solve_ivp

# 경로 주입
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 모델 및 시스템 임포트
from systems.sir import Sir
from experiments.compare_baseline.core import calculate_relative_error, evaluate_success
from experiments.compare_baseline.estimators.nlls import NLLSEstimator
from experiments.compare_baseline.estimators.mcmc import MCMCEstimator
from experiments.compare_baseline.estimators.adjoint import AdjointLBFGSEstimator
from experiments.compare_baseline.estimators.pinn import PINNEstimator
from experiments.compare_baseline.estimators.mlp import DirectMLEstimator
from experiments.compare_baseline.estimators.ours import ProposedEstimator

def perturb_on_sphere(true_values, radius, rng):
    """
    [시각화 전용 함수] N차원 구의 '표면(Boundary)'에서만 타원형으로 샘플링합니다.
    (core.py를 오염시키지 않기 위해 스크립트 내부에 독립적으로 정의)
    """
    direction = rng.normal(size=true_values.shape)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm  # 단위 벡터화
        
    noise = radius * direction
    perturbed = true_values * (1 + noise)
    
    # 물리적 하한선 강제 (음수 파라미터 방지)
    perturbed = np.maximum(1e-4, perturbed)
    return perturbed

def load_sir_ground_truth(sys_obj):
    """Spider Web 실험을 위한 SIR 전용 Ground Truth 로더"""
    t_eval = sys_obj.t_points
    theta_true = np.array([0.25, 0.1])
    x_hid_true_0 = np.array([10.0])
    y0 = [40.0, 10.0, 0.0]
    
    sol = solve_ivp(sys_obj.ode_func, [0, 110], y0, t_eval=t_eval, args=(theta_true,))
    x_obs = sol.y[sys_obj.observed_var_idx].reshape(-1, 1)
    return t_eval, x_obs, theta_true, x_hid_true_0

def run_spider_evaluation(estimator, sys_obj, radii=[1.0, 2.0], n_trials=30):
    """여러 반경(Multi-ring)에 대해 궤적을 수집하는 전용 평가 함수"""
    model_name = estimator.__class__.__name__.replace("Estimator", "")
    print(f"\n[{sys_obj.name.upper()}] Collecting Spider Web Data for {model_name} (Radii={radii}, N={n_trials})")
    
    t_eval, x_obs_clean, theta_true, x_hid_true_0 = load_sir_ground_truth(sys_obj)
    
    # 🚨 가장 이상적인 지형도를 위한 2% 노이즈 주입
    noise_rng = np.random.default_rng(2024)
    noise_level = 0.05
    noise = noise_rng.normal(0, noise_level * np.abs(x_obs_clean), x_obs_clean.shape)
    x_obs_noisy = x_obs_clean + noise
    
    # 인구수 범위 클리핑
    if sys_obj.name.lower() == 'sir':
        N_val = sum([val[0] for val in sys_obj.initial_conditions])
        x_obs_noisy = np.clip(x_obs_noisy, 0.0, N_val)
    else:
        x_obs_noisy = np.maximum(0.0, x_obs_noisy)
        
    all_results = {}
    
    for R in radii:
        # 🚨 [핵심 기법] R이 바뀔 때마다 동일한 시드로 리셋!
        # 이렇게 하면 R=1.0의 시작점과 R=2.0의 시작점이 동일한 직선(Ray) 상에 놓이게 되어
        # 시각적으로 완벽한 동심원(Concentric rings) 패턴이 형성됩니다.
        rng = np.random.default_rng(42) 
        
        success_count = 0
        errors = []
        all_trajectories = [] 
        
        for i in range(n_trials):
            # 시각화 전용 구면(Sphere) 섭동 함수 사용
            theta_init = perturb_on_sphere(theta_true, R, rng)
            x_hid_init_0 = perturb_on_sphere(x_hid_true_0, R, rng)
            
            try:
                theta_hat, _, _, p_history = estimator.fit(t_eval, x_obs_noisy, theta_init, x_hid_init_0)
                
                error = calculate_relative_error(theta_hat, theta_true)
                errors.append(error)
                
                if evaluate_success(error, threshold=0.05):
                    success_count += 1
                    
                all_trajectories.append([p.tolist() for p in p_history])
                    
            except Exception as e:
                all_trajectories.append([theta_init.tolist()])
                pass
                
        success_rate = (success_count / n_trials) * 100
        mean_error = float(np.mean(errors)) if len(errors) > 0 else float('inf')
        
        print(f" -> R={R} | Success: {success_rate:.1f}% | Mean Err: {mean_error:.4f}")
        
        all_results[str(R)] = {
            "success_rate": success_rate,
            "mean_error": mean_error,
            "all_trajectories": all_trajectories
        }
        
    return all_results

def run_spider_web_experiment():
    sys_obj = Sir()
    
    results_dir = os.path.join(project_root, 'results', 'final_benchmark')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f'spider_web_multi_ring_{timestamp}.json')
    
    print("="*60)
    print("🕸️ Starting Multi-Ring Spider Web Data Collection 🕸️")
    print("="*60)
    
    models = {
        "NLLS (LM)": NLLSEstimator(sys_obj),
        "Adjoint (L-BFGS)": AdjointLBFGSEstimator(sys_obj),
        #"MCMC (MH)": MCMCEstimator(sys_obj, n_iters=1000),
        #"PINN (Soft Physics)": PINNEstimator(sys_obj),
        "Direct Network (Naive ML)": DirectMLEstimator(sys_obj),
        "Proposed (Iterative)": ProposedEstimator(sys_obj)
    }
    
    for name, model in models.items():
        if hasattr(model, 'train_offline'):
            model.train_offline()
            
    final_results = {sys_obj.name: {}}
    for name, model in models.items():
        # n_trials를 30으로 세팅하여 선이 너무 뭉개지지 않도록 조절 (총 60가닥)
        model_result = run_spider_evaluation(model, sys_obj, radii=[1.0, 2.0], n_trials=30)
        final_results[sys_obj.name][name] = model_result
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n🎉 Multi-Ring Spider Web Data successfully saved to:\n -> {json_path}")

if __name__ == "__main__":
    run_spider_web_experiment()