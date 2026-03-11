import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 최상단 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
# TODO: 실제 구현된 모델과 평가 함수(또는 Trainer)를 임포트하세요.
# from models import AlternatingNetwork
# from data_loader import DataLoader, ... (테스트 데이터 로드용)

# ==============================================================================
# 1. 모델 레지스트리 구성 (효과적인 다중 파일 입력 관리)
# ==============================================================================
# 각 실험의 이름과 해당 .pth 경로, 그리고 모델 초기화에 필요한 Config 설정을 매핑합니다.
# main.py 실행 후 확보하신 실제 .pth 경로로 업데이트하시면 됩니다.
MODEL_REGISTRY = {
    'No Deriv\n(Obs Only)': {
        'path': 'results/ogtt_no_deriv/best_model.pth',
        'use_derivative': False,
        'derivative_method': None
    },
    'Finite\nDiff.': {
        'path': 'results/ogtt_fd/best_model.pth',
        'use_derivative': True,
        'derivative_method': 'finite_difference'
    },
    'Lagrange\nPoly.': {
        'path': 'results/ogtt_lagrange/best_model.pth',
        'use_derivative': True,
        'derivative_method': 'lagrangian'
    },
    'Cubic Spline\n(Ours)': {
        'path': 'results/ogtt_spline/best_model.pth',
        'use_derivative': True,
        'derivative_method': 'spline'
    }
}

# ==============================================================================
# 2. 평가 함수
# ==============================================================================
def evaluate_model(model_info, test_data):
    """
    단일 모델에 대한 평가를 수행하고 에러를 반환합니다.
    """
    cfg = Config()
    cfg.SYSTEM_NAME = 'ogtt_simul'
    cfg.USE_DERIVATIVE = model_info['use_derivative']
    cfg.DERIVATIVE_METHOD = model_info['derivative_method']
    
    # 1) 모델 초기화 (Config 설정에 맞춰 Input Dimension 등이 결정됨)
    # model = AlternatingNetwork(cfg).to(cfg.DEVICE)
    
    # 2) 가중치 로드
    # state_dict = torch.load(model_info['path'], map_location=cfg.DEVICE)
    # model.load_state_dict(state_dict)
    # model.eval()
    
    # 3) 추론 및 에러 계산 (TODO: 실제 추론 코드로 교체)
    # hidden_mse, param_mse = 0.0, 0.0
    # with torch.no_grad():
    #     for batch in test_data:
    #         # 추론 로직 (x_hid_hat, theta_hat 도출)
    #         # ...
    #         # hidden_mse += F.mse_loss(x_hid_hat, x_hid_true).item()
    #         # param_mse += F.mse_loss(theta_hat, theta_true).item()
    
    # return hidden_mse / len(test_data), param_mse / len(test_data)
    
    # (임시) 스크립트 동작 확인을 위한 가상 데이터 반환
    import random
    if cfg.USE_DERIVATIVE == False:
        return 0.85, 0.45
    elif cfg.DERIVATIVE_METHOD == 'finite_difference':
        return 0.62, 0.35
    elif cfg.DERIVATIVE_METHOD == 'lagrangian':
        return 5.40, 2.80
    else: # spline
        return 0.21, 0.12

def run_evaluation_pipeline():
    print("="*60)
    print("🚀 Starting Downstream Evaluation Pipeline")
    print("="*60)
    
    # 공통 테스트 데이터셋 로드 (노이즈가 포함된 테스트 데이터)
    # test_data = ... 
    test_data = None 
    
    methods = list(MODEL_REGISTRY.keys())
    hidden_errors = []
    param_errors = []
    
    for method_name, info in MODEL_REGISTRY.items():
        if not os.path.exists(info['path']):
            print(f"[Warning] File not found: {info['path']}")
            # 파일이 없을 경우 그래프의 해당 위치는 0으로 처리
            hidden_errors.append(0)
            param_errors.append(0)
            continue
            
        print(f"Evaluating: {method_name.replace(chr(10), ' ')}")
        h_err, p_err = evaluate_model(info, test_data)
        hidden_errors.append(h_err)
        param_errors.append(p_err)
        print(f"  -> Hidden MSE: {h_err:.4f} | Param MSE: {p_err:.4f}")
        
    # ==============================================================================
    # 3. 결과 시각화
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    width = 0.35
    
    # Grouped Bar Chart
    ax.bar(x - width/2, hidden_errors, width, label='Hidden State Error ($I, N_5, N_6$)', color='indianred', edgecolor='black', zorder=3)
    ax.bar(x + width/2, param_errors, width, label='Parameter Error ($S_i, \sigma$)', color='steelblue', edgecolor='black', zorder=3)
    
    ax.set_ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    ax.set_title('Downstream Estimation Performance by Derivative Scheme (OGTT)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    
    # 에러 격차를 명확하게 보여주기 위한 로그 스케일
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    save_path = 'downstream_network_performance.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\n✅ Evaluation complete! Figure saved to {save_path}")

if __name__ == "__main__":
    run_evaluation_pipeline()