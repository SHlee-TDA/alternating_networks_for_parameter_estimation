import torch
import numpy as np
import time
import os

from utils import Normalizer
from data_loader import DataGenerator
from config import Config
from trainer import Trainer
from models import HiddenVarPredictor, ParameterEstimator
from torch.utils.data import TensorDataset, DataLoader, random_split # DataLoader 추가

class ProposedEstimator:
    def __init__(self, sys_obj):
        self.sys = sys_obj
        self.config = Config()
        self.config.SYSTEM_NAME = sys_obj.name.lower()
        
        # EXPERIMENT_NAME 설정 (Trainer가 저장 경로로 사용)
        self.config.EXPERIMENT_NAME = f"{self.config.SYSTEM_NAME}_proposed_offline"
        self.device = self.config.DEVICE
        
        # [핵심 수정] Trainer가 실제로 가중치를 저장하는 경로와 완벽히 동기화
        self.model_dir = os.path.join(
            self.config.RESULTS_DIR, 
            self.config.SYSTEM_NAME, 
            self.config.EXPERIMENT_NAME
        )
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 가중치 파일명 (trainer.py가 저장하는 이름 기준)
        self.f_weight_path = os.path.join(self.model_dir, 'f_theta.pth')
        self.g_weight_path = os.path.join(self.model_dir, 'g_phi.pth')
        
        # 1. 데이터 로드 및 차원 동적 추론
        generator = DataGenerator(self.sys, self.config)
        self.obs_sim, self.hid_sim, self.params_sim, _ = generator.generate_data()
        
        self.config.FLAT_X_DIM = np.prod(self.obs_sim.shape[1:])
        self.config.FLAT_Y_DIM = np.prod(self.hid_sim.shape[1:])
        
        # 2. Normalizer 세팅
        scale_obs = np.percentile(np.abs(self.obs_sim), 99.9)
        scale_hid = np.percentile(np.abs(self.hid_sim), 99.9)
        use_log = (self.config.SYSTEM_NAME == 'ogtt_simul')
        self.normalizer = Normalizer(
            self.sys, self.device, 
            state_scales=[scale_obs * 1.2, scale_hid * 1.2], 
            param_bounds=(np.min(self.params_sim, axis=0)/1.2, np.max(self.params_sim, axis=0)*1.2),
            use_log_params=use_log, use_normalization=use_log
        )

        # 3. 모델 초기화 (Config 구조에 맞게)
        f_config = self.config.MODEL_CONFIG['f_theta']
        g_config = self.config.MODEL_CONFIG['g_phi']
        
        self.f_theta = HiddenVarPredictor(
            self.config.FLAT_X_DIM, 
            self.config.FLAT_Y_DIM, 
            len(sys_obj.param_names), 
            f_config
        ).to(self.device)
        
        self.g_phi = ParameterEstimator(
            self.config.FLAT_X_DIM, 
            self.config.FLAT_Y_DIM, 
            len(sys_obj.param_names), 
            g_config
        ).to(self.device)
        
        if os.path.exists(self.f_weight_path) and os.path.exists(self.g_weight_path):
            self.f_theta.load_state_dict(torch.load(self.f_weight_path, map_location=self.device))
            self.g_phi.load_state_dict(torch.load(self.g_weight_path, map_location=self.device))
            self.f_theta.eval()
            self.g_phi.eval()

    def train_offline(self):
        if os.path.exists(self.f_weight_path) and os.path.exists(self.g_weight_path):
            print(f"[Proposed] Found existing weights for {self.sys.name}. Skipping training.")
            return

        print(f"\n[Proposed] Starting offline training using defined Trainer for {self.sys.name}...")
        
        # 1. 데이터를 텐서로 변환하고 정규화 적용 (Trainer는 정규화된 입력을 기대함)
        x_t = self.normalizer.normalize_inputs(torch.tensor(self.obs_sim, dtype=torch.float32, device=self.device), 'observed')
        x_t = x_t.view(x_t.size(0), -1) # Flatten
        
        y_t = self.normalizer.normalize_inputs(torch.tensor(self.hid_sim, dtype=torch.float32, device=self.device), 'hidden')
        y_t = y_t.view(y_t.size(0), -1) # Flatten
        
        p_t = self.normalizer.normalize_params(torch.tensor(self.params_sim, dtype=torch.float32, device=self.device))
        
        # 2. Dataset 및 Train/Val Split 생성
        dataset = TensorDataset(x_t, y_t, p_t)
        
        val_size = int(len(dataset) * self.config.TEST_SPLIT)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 3. DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # 4. Trainer 초기화 (제공해주신 스펙에 정확히 맞춤)
        trainer = Trainer(
            f_theta=self.f_theta,
            g_phi=self.g_phi,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config
        )
        
        # 5. 훈련 실행
        trainer.train()
        
        # 6. 학습 완료 후 가중치 로드
        self.f_theta.load_state_dict(torch.load(self.f_weight_path, map_location=self.device))
        self.g_phi.load_state_dict(torch.load(self.g_weight_path, map_location=self.device))
        self.f_theta.eval()
        self.g_phi.eval()
        print(f"[Proposed] Training completed and loaded.")

    def fit(self, t_eval, x_obs, theta_init, x_hid_init):
        # --- 1. Feature Engineering (동적 미분 계산 내재화) ---
        # 들어오는 x_obs의 형태는 (T, obs_dim) 가정 (예: T, 1)
        T, obs_dim = x_obs.shape
        
        if getattr(self.config, 'USE_DERIVATIVE', False):
            # data_loader에서 사용한 것과 완벽히 동일한 미분기 로드
            from utils import get_derivative_estimator
            method = getattr(self.config, 'DERIVATIVE_METHOD', 'finite_diff')
            kwargs = {'order': 3} if method == 'poly' else {}
            
            estimator = get_derivative_estimator(method, **kwargs)
            
            # 각 관측 변수 채널별로 시간 미분(dy/dt) 계산
            x_dot = np.zeros_like(x_obs)
            for i in range(obs_dim):
                x_dot[:, i] = estimator.estimate(t_eval, x_obs[:, i])
                
            # 상태 변수와 미분값을 이어붙임. shape: (T, obs_dim * 2)
            x_obs_features = np.concatenate([x_obs, x_dot], axis=1)
        else:
            x_obs_features = x_obs

        # --- 2. 정규화 및 텐서 변환 (데이터 규격 맞추기) ---
        # 네트워크 입력 규격(FLAT_X_DIM)에 맞게 1차원으로 쭉 폅니다.
        x_obs_flat = x_obs_features.flatten()
        x_obs_t = torch.tensor(x_obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 훈련 때 피팅된 Normalizer로 스케일링 [0, 1] 또는 [-1, 1]
        x_obs_norm = self.normalizer.normalize_inputs(x_obs_t, variable_type='observed')
        
        # 초기 파라미터 섭동값(theta_init)도 동일하게 스케일링
        p_init_t = torch.tensor(theta_init, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_curr = self.normalizer.normalize_params(p_init_t)

        # --- 3. 고정점 반복 (Alternating Projection) ---
        start_time = time.time()
        
        # config에 1로 되어있던 실수를 방지하기 위해 최소 50번의 반복 보장
        max_iters = max(getattr(self.config, 'ITERATIONS', 50), 50) 
        
        with torch.no_grad():
            for _ in range(max_iters):
                # 1단계: 관측치와 현재 파라미터로 은닉 상태 추론
                y_guess_norm = self.f_theta(x_obs_norm, p_curr)
                
                # 2단계: 관측치와 방금 구한 은닉 상태로 다음 파라미터 추론
                p_next_norm = self.g_phi(x_obs_norm, y_guess_norm)
                
                # 수렴 판정: 파라미터 변화량이 1e-7 미만이면 조기 종료(Early stopping)
                if torch.norm(p_next_norm - p_curr) < 1e-7:
                    p_curr = p_next_norm
                    break
                    
                p_curr = p_next_norm
                
        exec_time = time.time() - start_time
        
        # --- 4. 역정규화 (Denormalization) ---
        theta_hat_t = self.normalizer.denormalize_params(p_curr)
        theta_hat = theta_hat_t.squeeze(0).cpu().numpy()
        
        # 은닉 상태는 평가 대상이 아니므로 더미(Zero) 반환
        return theta_hat, np.zeros_like(x_hid_init), exec_time