import sys
import os
import time
import numpy as np
import torch

# 경로 주입
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from systems.sir import Sir
from systems.lotka_volterra import LotkaVolterra
from experiments.compare_baseline.core import calculate_relative_error # 오차 계산 함수 임포트 추가
from experiments.compare_baseline.run_pilot_experiment import (
    run_monte_carlo_evaluation, save_results_to_json, append_to_markdown
)

# ML Estimators
from experiments.compare_baseline.estimators.pinn import PINNEstimator
from experiments.compare_baseline.estimators.mlp import DirectMLEstimator
from experiments.compare_baseline.estimators.ours import ProposedEstimator

# --- 새롭게 추가된 Sanity Check 함수 ---
def check_in_distribution_performance(direct_model, proposed_model):
    print("\n--- In-Distribution Sanity Check (N=100) ---")
    
    # 두 모델 모두 동일한 데이터를 로드했으므로 proposed_model의 데이터를 활용
    obs_sim = proposed_model.obs_sim
    hid_sim = proposed_model.hid_sim
    params_sim = proposed_model.params_sim
    
    # 평가 시점 (t_eval) 가져오기
    sys_obj = proposed_model.sys
    t_eval = sys_obj.t_points
    
    # 원본 관측 변수의 개수 파악 (미분 피처를 제외한 날것의 차원)
    orig_obs_dim = 1 if isinstance(sys_obj.observed_var_idx, int) else len(sys_obj.observed_var_idx)
    
    direct_errors = []
    proposed_errors = []
    
    # 훈련 데이터 중 처음 100개 샘플 추출
    num_tests = min(100, len(obs_sim))
    for i in range(num_tests):
        x_obs_full = obs_sim[i]
        # [핵심] DataGenerator가 붙여놓은 미분값을 떼어내고 '순수 상태 변수'만 추출
        x_obs_raw = x_obs_full[:, :orig_obs_dim] 
        
        theta_true = params_sim[i]
        
        # 섭동 없는 정확한 초기값(R=0)
        theta_init = theta_true.copy() 
        x_hid_init = hid_sim[i][0].copy()
        
        # 1. Direct Model 평가 (t_eval 전달 및 순수 데이터 입력)
        theta_hat_dir, _, _ = direct_model.fit(t_eval, x_obs_raw, theta_init, x_hid_init)
        direct_errors.append(calculate_relative_error(theta_hat_dir, theta_true))
        
        # 2. Proposed Model 평가
        theta_hat_prop, _, _ = proposed_model.fit(t_eval, x_obs_raw, theta_init, x_hid_init)
        proposed_errors.append(calculate_relative_error(theta_hat_prop, theta_true))
        
    print(f"Direct ML Mean Error:   {np.mean(direct_errors):.4f}")
    print(f"Proposed ML Mean Error: {np.mean(proposed_errors):.4f}")
    print("-" * 44 + "\n")

# --- 메인 실행 함수 ---
def run_ml_experiment():
    test_radii = [0.1, 0.5, 1.0, 2.0]
    systems = [Sir(), LotkaVolterra()]
    
    results_dir = os.path.join(project_root, 'results', 'baselines')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_paths = {
        'json': os.path.join(results_dir, f'ml_baselines_{timestamp}.json'),
        'md': os.path.join(results_dir, f'ml_baselines_{timestamp}.md')
    }

    print(f"ML Results will be saved to:\n - {save_paths['json']}\n - {save_paths['md']}")

    for sys_obj in systems:
        print(f"\n{'='*50}\nML SYSTEM: {sys_obj.name.upper()}\n{'='*50}")
        
        # 1. 모델 인스턴스화
        models = {
            "Direct Network (Naive ML)": DirectMLEstimator(sys_obj),
            "Proposed (Iterative)": ProposedEstimator(sys_obj),
            "PINN (Soft Physics)": PINNEstimator(sys_obj)
        }
        
        # 2. 오프라인 학습 (지원하는 모델만)
        for name, model in models.items():
            if hasattr(model, 'train_offline'):
                model.train_offline()
                
        # 3. [핵심] OOD 문제 진단을 위한 In-Distribution Sanity Check 실행
        check_in_distribution_performance(
            models["Direct Network (Naive ML)"], 
            models["Proposed (Iterative)"]
        )
                
        # 4. 몬테카를로 평가 진행 (기존 OOD 검증 시나리오)
        for name, model in models.items():
            run_monte_carlo_evaluation(model, sys_obj, test_radii, n_trials=50, save_paths=save_paths)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    run_ml_experiment()