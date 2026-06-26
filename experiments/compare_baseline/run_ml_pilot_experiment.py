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
from experiments.compare_baseline.core import calculate_relative_error

# 🚨 [수정됨] 직전에 수정하여 완성된 run_pilot_experiment.py의 함수들 임포트
from experiments.compare_baseline.run_pilot_experiment import (
    run_monte_carlo_evaluation, save_results_to_json, append_to_markdown
)

# 1. Classical Estimators
from experiments.compare_baseline.estimators.nlls import NLLSEstimator
from experiments.compare_baseline.estimators.mcmc import MCMCEstimator
from experiments.compare_baseline.estimators.adjoint import AdjointLBFGSEstimator

# 2. ML Estimators
from experiments.compare_baseline.estimators.pinn import PINNEstimator
from experiments.compare_baseline.estimators.mlp import DirectMLEstimator
from experiments.compare_baseline.estimators.ours import ProposedEstimator

def check_in_distribution_performance(direct_model, proposed_model):
    """ML 모델들이 Training 분포 내에서는 제대로 예측하는지 확인하는 Sanity Check (with 5% Noise)"""
    print("\n--- ML Models In-Distribution Sanity Check (N=100) ---")
    
    obs_sim, hid_sim, params_sim = proposed_model.obs_sim, proposed_model.hid_sim, proposed_model.params_sim
    t_eval = proposed_model.sys.t_points
    orig_obs_dim = 1 if isinstance(proposed_model.sys.observed_var_idx, int) else len(proposed_model.sys.observed_var_idx)
    
    direct_errors, proposed_errors = [], []
    num_tests = min(100, len(obs_sim))
    
    # 평가의 일관성을 위한 고정 시드 노이즈 생성기
    noise_rng = np.random.default_rng(999)
    
    for i in range(num_tests):
        x_obs_raw = obs_sim[i][:, :orig_obs_dim] 
        theta_true = params_sim[i]
        
        # 5% 노이즈 주입
        noise_level = 0.05
        noise = noise_rng.normal(0, noise_level * np.abs(x_obs_raw), x_obs_raw.shape)
        x_obs_noisy = x_obs_raw + noise
        
        # 물리적 클리핑
        if proposed_model.sys.name.lower() == 'sir':
            N = sum([val[0] for val in proposed_model.sys.initial_conditions])
            x_obs_noisy = np.clip(x_obs_noisy, 0.0, N - 0.1)
        else:
            x_obs_noisy = np.maximum(0.0, x_obs_noisy)

        theta_init = theta_true.copy() 
        x_hid_init = hid_sim[i][0].copy()
        
        # 노이즈가 섞인 데이터를 각 모델에 제공
        theta_hat_dir, _, _, _ = direct_model.fit(t_eval, x_obs_noisy, theta_init, x_hid_init)
        direct_errors.append(calculate_relative_error(theta_hat_dir, theta_true))
        
        theta_hat_prop, _, _, _ = proposed_model.fit(t_eval, x_obs_noisy, theta_init, x_hid_init)
        proposed_errors.append(calculate_relative_error(theta_hat_prop, theta_true))
        
    print(f"Direct ML Mean Error:   {np.mean(direct_errors):.4f}")
    print(f"Proposed ML Mean Error: {np.mean(proposed_errors):.4f}")
    print("-" * 54 + "\n")


def run_integrated_experiment():
    # 🚨 [수정 1] radius R 대신 Log-Normal의 sigma 스케일 적용
    test_radii = [0.1, 0.5, 1.0, 2.0]
    
    # 🚨 [수정 2] 난이도(Regime) 시나리오 도입
    regimes = ["easy", "hard", "ill-posed"]
    
    systems = [Sir()] # 논문용 최종 실험
    
    results_dir = os.path.join(project_root, 'results', 'numerical_comparison')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    save_paths = {
        'json': os.path.join(results_dir, f'integrated_results_{timestamp}.json'),
        'md': os.path.join(results_dir, f'integrated_results_{timestamp}.md')
    }

    print(f"Grand Unified Benchmark Starting!\nResults will be saved to:\n - {save_paths['json']}\n - {save_paths['md']}")

    for sys_obj in systems:
        print(f"\n{'='*60}\nSYSTEM: {sys_obj.name.upper()} (WITH 5% NOISE STRESS TEST)\n{'='*60}")
        
        # 1. 모든 경쟁 모델 선언
        models = {
            "NLLS (LM)": NLLSEstimator(sys_obj),
            "Adjoint (L-BFGS)": AdjointLBFGSEstimator(sys_obj),
            # "MCMC (MH)": MCMCEstimator(sys_obj, n_iters=1000), # 시간 관계상 주석 유지
            "PINN (Soft Physics)": PINNEstimator(sys_obj),
            "Direct Network (Naive ML)": DirectMLEstimator(sys_obj),
            "Proposed (Iterative)": ProposedEstimator(sys_obj)
        }
        
        # 2. 오프라인 학습 수행 (ML 모델들만 해당됨)
        # ML 모델은 특정 Regime이 아니라 전체 파라미터 분포를 한 번에 학습합니다.
        print("\n[Phase 1] Offline Training for ML Models...")
        for name, model in models.items():
            if hasattr(model, 'train_offline'):
                model.train_offline()
                
        # 3. ML 모델 Sanity Check
        check_in_distribution_performance(
            models["Direct Network (Naive ML)"], 
            models["Proposed (Iterative)"]
        )
                
        # 4. 공정한 몬테카를로 OOD 평가 (난이도별 순회)
        print("\n[Phase 2] Monte Carlo Evaluation (Robustness Test Across Regimes)...")
        
        # 🚨 [수정 3] 각 난이도(Regime)별로 모든 모델들을 평가
        for regime in regimes:
            print(f"\n" + "+" * 50)
            print(f"+++ Evaluating Regime: {regime.upper()} +++")
            print("+" * 50)
            
            for name, model in models.items():
                run_monte_carlo_evaluation(
                    estimator=model, 
                    sys_obj=sys_obj, 
                    test_radii=test_radii, 
                    regime=regime, 
                    n_trials=200, 
                    save_paths=save_paths
                )
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    run_integrated_experiment()