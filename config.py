# config.py
import torch

class Config:
    """실험 전체의 하이퍼파라미터와 설정을 관리하는 중앙 통제실"""

    # =========================================================================
    # 1. 실험 선택 (Experiment Selection)
    # =========================================================================
    SYSTEM_NAME = 'ogtt_simul'
    
    # =========================================================================
    # 2. 실행 관리 (Execution Management) - [Phase 5 업데이트]
    # =========================================================================
    # main.py의 get_experiment_dataloaders 함수가 이 설정을 읽어 동작합니다.
    #
    # 키 설명:
    # - use_sde (bool): True면 SDE(Euler-Maruyama) 데이터, False면 ODE 데이터 사용
    # - scenario (str): 
    #     'sim_only': 시뮬레이션 데이터만 학습 (Baseline / Domain Randomization)
    #     'hybrid': 시뮬레이션 + 실제 데이터 혼합 학습 (Weighted Random Sampling 적용)
    # - real_ratio (float): Hybrid 모드일 때 배치 내 실제 데이터의 비율 (0.0 ~ 1.0)
    # - val_source (str):
    #     'sim': 시뮬레이션 데이터의 일부를 검증(Validation)에 사용 (기존 방식)
    #     'real': 실제 데이터의 20%를 검증에 사용 (Real World 성능 직접 측정)
    # 1) 기본 데이터 시나리오 정의
    #    각 시나리오별로 데이터 구성과 검증 방식이 다릅니다.
    BASE_SCENARIOS = [
        # [시나리오 1] ODE Pure: ODE 데이터만 사용 (Baseline)
        {'name_prefix': 'ode_pure', 'use_sde': False, 'scenario': 'sim_only', 'val_source': 'sim'},
        
        # [시나리오 2] SDE Pure: SDE 데이터만 사용 (Domain Randomization 효과 확인)
        {'name_prefix': 'sde_pure', 'use_sde': True, 'scenario': 'sim_only', 'val_source': 'sim'},
        
        # [시나리오 3] ODE Hybrid: ODE 데이터 + Real 데이터 혼합 학습
        #{'name_prefix': 'ode_hybrid', 'use_sde': False, 'scenario': 'hybrid', 'real_ratio': 0.3, 'val_source': 'sim'},

        # [시나리오 4] SDE Hybrid: SDE 데이터 + Real 데이터 혼합 학습 (Main Method)
        #{'name_prefix': 'sde_hybrid', 'use_sde': True, 'scenario': 'hybrid', 'real_ratio': 0.3, 'val_source': 'sim'},
    
        # [시나리오 5] Mix Pure: (ODE + SDE) 학습 -> (ODE+SDE) 검증 -> 평가
        #{'name_prefix': 'mix_pure', 'use_sde': 'mixed', 'scenario': 'sim_only', 'val_source': 'sim'},
        
        # [시나리오 6] Mix Hybrid: (ODE + SDE + Real) 학습 -> (ODE+SDE) 검증 -> 평가
        #{'name_prefix': 'mix_hybrid', 'use_sde': 'mixed', 'scenario': 'hybrid', 'real_ratio': 0.3, 'val_source': 'sim'}
    ]
    
    # 2) Loss/Model 조합 정의 (4가지 경우의 수)
    LOSS_VARIANTS = [
        #{'sn': False, 'cl': False, 'suffix': 'vanilla'},          # 기본 모델
        #{'sn': True,  'cl': False, 'suffix': 'spectral_norm'},    # SN만 적용
        #{'sn': False, 'cl': True,  'suffix': 'consistency_loss'}, # CL만 적용
        {'sn': True,  'cl': True,  'suffix': 'full_method'}       # 둘 다 적용 (제안 방법)
    ]
    
    # 3) 실험 리스트 자동 생성 (Cartesian Product)
    EXPERIMENTS = []
    for scen in BASE_SCENARIOS:
        for loss_setting in LOSS_VARIANTS:
            # 기본 설정 복사
            exp = scen.copy()
            
            # Loss/Model 설정 주입
            exp['use_spectral_norm'] = loss_setting['sn']
            exp['use_consistency_loss'] = loss_setting['cl']
            
            # 실험 이름 생성 (예: hybrid_sim_val_full_method)
            # name_prefix는 삭제 (깔끔하게)
            prefix = exp.pop('name_prefix')
            exp['name'] = f"{prefix}_{loss_setting['suffix']}"
            
            EXPERIMENTS.append(exp)

    
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = 'results'

    # =========================================================================
    # 3. 데이터 및 학습 하이퍼파라미터
    # =========================================================================
    NUM_SAMPLES = 10000     # 생성할 증강 데이터 샘플 수
    TEST_SPLIT = 0.2
    BATCH_SIZE = 128
    EPOCHS = 20000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 200
    EARLY_STOPPING_MIN_DELTA = 1e-6
    
    # =========================================================================
    # 4. 방법론 관련 하이퍼파라미터 (Methodology Hyperparameters)
    # =========================================================================
    CONSISTENCY_LOSS_LAMBDA = 1.0
    ITERATIONS = 10
    
    # [Phase 5 설정] Lagrangian 및 미분 방법
    USE_LAGRANGIAN = True         # 미분값(Derivative)을 Feature로 사용
    DERIVATIVE_METHOD = 'spline'  # 실제 데이터 미분 시 Spline Smoothing 사용
    
    AUGMENTATION_FACTOR = 30
    SDE_SCALE_FACTORS = {
        'bias_scale': 1.0,
        'diffusion_scale': 1.27
    }
    
    
    # =========================================================================
    # 5. 모델 구조 설정 (Model Architecture)
    # =========================================================================
    MODEL_CONFIG = {
        'f_theta': {
            'hidden_dims': [256, 256, 256, 256, 256, 256],
            'activation': 'SiLU'
        },
        'g_phi': {
            'hidden_dims': [256, 256, 256, 256, 256, 256],
            'activation': 'SiLU'
        },
        'initialization': {
            'type': 'xavier',
            'distribution': 'normal'
        }
    }