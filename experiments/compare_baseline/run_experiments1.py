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

# 통합 벤치마크에서 사용하는 Ground Truth 로더 임포트
from experiments.compare_baseline.run_pilot_experiment import load_ground_truth_data

from experiments.compare_baseline.estimators.nlls import NLLSEstimator
from experiments.compare_baseline.estimators.mcmc import MCMCEstimator
from experiments.compare_baseline.estimators.adjoint import AdjointLBFGSEstimator
from experiments.compare_baseline.estimators.pinn import PINNEstimator
from experiments.compare_baseline.estimators.mlp import DirectMLEstimator
from experiments.compare_baseline.estimators.ours import ProposedEstimator

def perturb_on_sphere(true_values, radius, rng):
    """
    [시각화 전용 함수] 파라미터 공간에서 N차원 구의 표면을 샘플링합니다.
    🚨 [수정됨] 지수배(Log)를 버리고, 상대오차 기반 선형 섭동으로 복귀합니다.
    반경 R이 곧 최대 상대오차율(예: R=1.0 이면 100% 오차)을 의미합니다.
    """
    direction = rng.normal(size=true_values.shape)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm  # 단위 벡터화
        
    noise = radius * direction
    perturbed = true_values * (1 + noise)
    
    # 물리적 하한선 강제 (0 이하 방지)
    perturbed = np.clip(perturbed, 1e-4, 10.0)
    return perturbed

def run_spider_evaluation(estimator, sys_obj, radii=[1.0, 2.0], regime="easy", n_trials=30):
    model_name = estimator.__class__.__name__.replace("Estimator", "")
    print(f"[{sys_obj.name.upper()} | {regime.upper()}] Collecting Spider Web Data for {model_name}...")
    
    # 통합 벤치마크와 동일한 난이도별 GT 로드
    t_eval, x_obs_clean, theta_true, x_hid_true_0 = load_ground_truth_data(sys_obj, regime=regime)
    
    # 가장 이상적인 지형도를 위한 5% 노이즈 주입
    noise_rng = np.random.default_rng(2024)
    noise_level = 0.05
    noise = noise_rng.normal(0, noise_level * np.abs(x_obs_clean), x_obs_clean.shape)
    x_obs_noisy = x_obs_clean + noise
    
    # 인구수 범위 클리핑
    N_val = 50.0 # 기본값
    if sys_obj.name.lower() == 'sir':
        N_val = sum([val[0] for val in sys_obj.initial_conditions])
        x_obs_noisy = np.clip(x_obs_noisy, 0.0, N_val)
    else:
        x_obs_noisy = np.maximum(0.0, x_obs_noisy)
        
    all_results = {}
    
    for R in radii:
        rng = np.random.default_rng(42) 
        
        success_count = 0
        errors = []
        all_trajectories = [] 
        
        for i in range(n_trials):
            # 1. 파라미터는 상대오차(R)만큼 섭동
            theta_init = perturb_on_sphere(theta_true, R, rng)
            
            # 🚨 2. [핵심 수정] 은닉 상태(I_0)는 관측치로부터 물리적으로 유도 (결정론적)
            if sys_obj.name.lower() == 'sir':
                S0_obs = float(x_obs_noisy[0, 0])
                I0_inferred = max(0.1, N_val - S0_obs)
                x_hid_init_0 = np.array([I0_inferred])
            else:
                x_hid_init_0 = perturb_on_sphere(x_hid_true_0, R, rng)
            
            try:
                theta_hat, _, _, p_history = estimator.fit(t_eval, x_obs_noisy, theta_init, x_hid_init_0)
                
                error = calculate_relative_error(theta_hat, theta_true)
                errors.append(error)
                
                # 🚨 3. [수정됨] 하드코딩된 threshold=0.05 제거. core.py의 기준을 따름
                if evaluate_success(error):
                    success_count += 1
                    
                all_trajectories.append([p.tolist() for p in p_history])
                    
            except Exception as e:
                # 에러로 터지면 시작점만 점으로 남김
                all_trajectories.append([theta_init.tolist()])
                pass
                
        success_rate = (success_count / n_trials) * 100
        mean_error = float(np.mean(errors)) if len(errors) > 0 else float('inf')
        
        print(f" -> Radius R={R} | Success: {success_rate:5.1f}% | Mean Err: {mean_error:.4f}")
        
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
        "PINN (Soft Physics)": PINNEstimator(sys_obj),
        "Direct Network (Naive ML)": DirectMLEstimator(sys_obj),
        "Proposed (Iterative)": ProposedEstimator(sys_obj)
    }
    
    # 1. 오프라인 학습 (ML 모델들)
    for name, model in models.items():
        if hasattr(model, 'train_offline'):
            model.train_offline()
            
    final_results = {sys_obj.name: {}}
    
    # 2. 두 가지 난이도(Regime)에 대해 모두 시각화 궤적 추출
    regimes_to_visualize = ["easy", "hard"]
    
    for regime in regimes_to_visualize:
        print(f"\n" + "-"*50)
        print(f"--- Processing Regime: {regime.upper()} ---")
        print("-"*50)
        
        final_results[sys_obj.name][regime] = {}
        
        for name, model in models.items():
            # n_trials를 30으로 세팅하여 선이 뭉개지지 않도록 조절
            model_result = run_spider_evaluation(model, sys_obj, radii=[1.0, 2.0], regime=regime, n_trials=30)
            final_results[sys_obj.name][regime][name] = model_result
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    # 3. JSON 저장
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n🎉 Multi-Ring Spider Web Data successfully saved to:\n -> {json_path}")

if __name__ == "__main__":
    run_spider_web_experiment()