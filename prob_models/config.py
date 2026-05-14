import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ProbConfig:
    """
    확률론적 생성 모델(CVAE + Pseudo-Gibbs Sampling) 전용 Configuration
    """
    # --- 1. System & Environment ---
    SYSTEM_NAME: str = 'ogtt_simul'
    EXPERIMENT_NAME: str = 'probabilistic_ogtt_v1'
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR: str = 'results/probabilistic_runs'
    
    # --- 2. Data Generator Compatibility Flags (data_loader.py 호환용) ---
    NUM_SAMPLES: int = 50000
    USE_SDE: bool = False
    AUGMENTATION_FACTOR: int = 1
    SDE_SCALE_FACTORS: Dict[str, float] = field(default_factory=lambda: {
        'bias_scale': 1.0,
        'diffusion_scale': 1.0
    })
    USE_DERIVATIVE: bool = True       # OGTT 데이터로더 호환을 위해 유지
    DERIVATIVE_METHOD: str = 'spline' # OGTT 데이터로더 호환을 위해 유지
    USE_LOG_PARAMS: bool = True       # OGTT 파라미터 스케일 정규화
    TEST_SPLIT: float = 0.2
    BATCH_SIZE: int = 256
    
    # --- 3. Training & Optimization ---
    EPOCHS: int = 3000
    LEARNING_RATE: float = 1e-3
    KL_WARMUP_EPOCHS: int = 150       # CVAE 붕괴 방지를 위한 긴 웜업
    EARLY_STOPPING_PATIENCE: int = 200
    EARLY_STOPPING_MIN_DELTA: float = 1e-5
    
    # --- 4. CVAE Architecture ---
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    LATENT_DIM_HIDDEN: int = 16  # Network A의 Z 공간
    LATENT_DIM_PARAM: int = 16    # Network B의 Z 공간
    
    # --- 5. Inference (Pseudo-Gibbs Sampling) ---
    INFERENCE_CHAINS: int = 50   # MCMC 체인 수 (그려질 밴드의 두께감 결정)
    INFERENCE_STEPS: int = 60    # 핑퐁 횟수
    INFERENCE_BURN_IN: int = 15  # 예열 기간 폐기 샘플 수

    def __post_init__(self):
        os.makedirs(os.path.join(self.RESULTS_DIR, self.EXPERIMENT_NAME), exist_ok=True)