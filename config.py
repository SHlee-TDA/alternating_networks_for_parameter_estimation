# config.py
import torch

class Config:
    """실험 전체의 하이퍼파라미터와 설정을 관리하는 중앙 통제실"""

    # = an======================================================================
    # 1. 실험 선택 (Experiment Selection)
    # =========================================================================
    # 여기서 시스템 이름을 변경하여 Lotka-Volterra와 SIR 실험을 전환합니다.
    # 사용 가능한 옵션: 'lotka_volterra', 'sir', 'nc_sir'
    SYSTEM_NAME = 'lotka_volterra'
    
    # =========================================================================
    # 2. 실행 관리 (Execution Management)
    # =========================================================================
    # main.py에서 순차적으로 실행할 실험 시나리오 목록을 정의합니다.
    EXPERIMENTS = [
        {'name': 'vanila', 'use_spectral_norm': False, 'use_consistency_loss': False},
        {'name': 'spectral_norm', 'use_spectral_norm': True, 'use_consistency_loss': False},
        {'name': 'consistency_loss', 'use_spectral_norm': True, 'use_consistency_loss': True},
        {'name': 'consistency_loss_only', 'use_spectral_norm': False, 'use_consistency_loss': True},
        {'name': 'lagrangian', 'use_spectral_norm': False, 'use_consistency_loss': False, 'use_lagrangian': True},
        {'name': 'lagrangian_spectral_norm', 'use_spectral_norm': True, 'use_consistency_loss': False, 'use_lagrangian': True},
        {'name': 'lagrangian_consistency_loss', 'use_spectral_norm': True, 'use_consistency_loss': True, 'use_lagrangian': True},
        {'name': 'lagrangian_consistency_loss_only', 'use_spectral_norm': False, 'use_consistency_loss': True, 'use_lagrangian': True}
    ]
    
    SEED = 42  # 재현성을 위한 랜덤 시드
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = 'results'  # 결과물이 저장될 최상위 폴더 (실제 경로는 'results/SYSTEM_NAME/EXPERIMENT_NAME' 구조가 됩니다)

    # =========================================================================
    # 3. 데이터 및 학습 하이퍼파라미터 (Data & Training Hyperparameters)
    # =========================================================================
    NUM_SAMPLES = 100      # 생성할 데이터 샘플의 총 개수
    TEST_SPLIT = 0.2        # 테스트 데이터셋의 비율
    BATCH_SIZE = 128        # 학습 시 미니배치 크기
    EPOCHS = 10          # 총 학습 에포크 수
    LEARNING_RATE = 1e-6    # 옵티마이저의 학습률
    WEIGHT_DECAY = 1e-5     # L2 정규화 (Weight Decay) 계수
    
    # =========================================================================
    # 4. 방법론 관련 하이퍼파라미터 (Methodology Hyperparameters)
    # =========================================================================
    # Consistency Loss의 가중치 람다 값
    CONSISTENCY_LOSS_LAMBDA = 1.0
    
    # 테스트 시 고정점 반복법의 반복 횟수
    ITERATIONS = 10
    
    USE_LAGRANGIAN = True  # 라그랑지안 방법론 사용 여부 (main.py에서 동적으로 설정됨)
    
    # =========================================================================
    # 5. 모델 구조 설정 (Model Architecture)
    # =========================================================================
    # 각 네트워크의 구조를 여기서 정의하여 유연하게 변경할 수 있습니다.
    MODEL_CONFIG = {
        'f_theta': {
            'hidden_dims': [128, 128, 128, 128],  # 은닉층의 뉴런 수 리스트
            'activation': 'Tanh'      # 활성화 함수 (예: 'ReLU', 'Tanh')
        },
        'g_phi': {
            'hidden_dims': [128, 128, 128, 128],
            'activation': 'Tanh'
        },
        'initialization': {
            'type': 'xavier',  # 가중치 초기화 방법 ('xavier' or 'kaming')
            'distribution': 'normal'  # 분포 유형 ('uniform' 또는 'normal')
        }
    }