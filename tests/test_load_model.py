import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models import SingleNetworkBaseline


def test_single_network_baseline():
    print("--- SingleNetworkBaseline Sanity Check ---")
    
    # 1. 가상의 설정(Config) 정의
    # 논문에 명시된 대로 64 차원의 은닉층 3개를 가정합니다.
    model_config = {
        'hidden_dims': [64, 64, 64],
        'activation': 'silu' # 또는 프로젝트에서 사용하는 활성화 함수 문자열
    }
    
    # 2. 입출력 차원 정의 (예: Batch = 32, Observation time steps = 10, d_obs = 1, params = 2)
    batch_size = 32
    time_steps = 10
    d_obs = 1
    # u 벡터는 x_obs와 그 미분값을 포함하므로 차원이 2배가 될 수 있습니다 (예: 10 * 1 * 2 = 20)
    flat_x_dim = time_steps * d_obs * 2 
    # 출력은 파라미터(theta)의 개수 (예: SIR 모델의 beta, gamma)
    flat_y_dim = 2 
    
    print(f"Input dimension (flat_x_dim): {flat_x_dim}")
    print(f"Output dimension (flat_y_dim): {flat_y_dim}")

    # 3. 모델 인스턴스 생성
    try:
        model = SingleNetworkBaseline(
            flat_x_dim=flat_x_dim, 
            flat_y_dim=flat_y_dim, 
            model_config=model_config, 
            use_spectral_norm=True
        )
        print("\nModel Architecture:")
        print(model)
        print("-> Model instantiation successful!")
    except Exception as e:
        print(f"-> Model instantiation failed: {e}")
        return

    # 4. 더미 입력 데이터 생성 (Random Tensor)
    dummy_input = torch.randn(batch_size, flat_x_dim)
    print(f"\nDummy input shape: {dummy_input.shape}")

    # 5. 순전파(Forward pass) 실행 및 출력 차원 확인
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # 출력 차원이 기대한 바와 같은지 검증
        assert output.shape == (batch_size, flat_y_dim), f"Expected {(batch_size, flat_y_dim)}, got {output.shape}"
        print("-> Forward pass and shape check successful!")
        
    except Exception as e:
        print(f"-> Forward pass failed: {e}")

if __name__ == "__main__":
    test_single_network_baseline()