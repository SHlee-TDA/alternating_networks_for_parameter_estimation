import torch
import torch.nn as nn
import numpy as np
import time
import os

from config import Config
from data_loader import DataGenerator
from utils import Normalizer

# 초기값(p_init)을 입력으로 함께 받는 조건부 Direct 모델
class ConditionalDirectPredictor(nn.Module):
    def __init__(self, x_dim, num_params, hidden_dim=64):
        super().__init__()
        # 입력 차원: 관측 데이터 차원(x_dim) + 초기 섭동 파라미터 차원(num_params)
        self.net = nn.Sequential(
            nn.Linear(x_dim + num_params, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_params)
        )
        
    def forward(self, x, p_init):
        # x와 p_init을 결합(Concatenate)하여 추론
        combined_input = torch.cat([x, p_init], dim=1)
        return self.net(combined_input)

class DirectMLEstimator:
    def __init__(self, sys_obj):
        self.sys = sys_obj
        self.config = Config()
        self.config.SYSTEM_NAME = sys_obj.name.lower()
        self.device = self.config.DEVICE
        
        self.model_dir = os.path.join('checkpoints', self.sys.name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.weight_path = os.path.join(self.model_dir, 'conditional_direct_net_best.pth')
        
        # 1. 데이터 로드 및 차원 추론
        generator = DataGenerator(self.sys, self.config)
        self.obs_sim, self.hid_sim, self.params_sim, _ = generator.generate_data()
        
        self.config.FLAT_X_DIM = np.prod(self.obs_sim.shape[1:])
        self.config.FLAT_Y_DIM = np.prod(self.hid_sim.shape[1:])
        
        # 2. 정규화 (Normalization) 필수 적용
        scale_obs = np.percentile(np.abs(self.obs_sim), 99.9)
        scale_hid = np.percentile(np.abs(self.hid_sim), 99.9)
        # 스케일 붕괴 방지를 위해 모든 시스템에서 정규화 강제 활성화
        self.normalizer = Normalizer(
            self.sys, self.device, 
            state_scales=[scale_obs * 1.2, scale_hid * 1.2], 
            param_bounds=(np.min(self.params_sim, axis=0)/1.2, np.max(self.params_sim, axis=0)*1.2),
            use_log_params=True, 
            use_normalization=True 
        )
        
        # 3. 모델 초기화
        direct_hidden_dim = self.config.MODEL_CONFIG['f_theta']['hidden_dims'][0]
        self.num_params = len(sys_obj.param_names)
        
        self.network = ConditionalDirectPredictor(
            x_dim=self.config.FLAT_X_DIM, 
            num_params=self.num_params,
            hidden_dim=direct_hidden_dim 
        ).to(self.device)
        
        if os.path.exists(self.weight_path):
            self.network.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            self.network.eval()

    def train_offline(self):
        if os.path.exists(self.weight_path):
            print(f"[Direct ML] Found existing weights. Skipping training.")
            return

        print(f"\n[Direct ML] Starting Conditional offline training for {self.sys.name}...")
        
        # 1. 훈련 데이터를 텐서로 변환하고 "정규화" 적용
        x_t = torch.tensor(self.obs_sim, dtype=torch.float32, device=self.device)
        x_norm = self.normalizer.normalize_inputs(x_t, 'observed').view(x_t.size(0), -1) 
        
        p_true_t = torch.tensor(self.params_sim, dtype=torch.float32, device=self.device)
        p_true_norm = self.normalizer.normalize_params(p_true_t)
        
        dataset = torch.utils.data.TensorDataset(x_norm, p_true_norm)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.network.train()
        epochs = 100000
        for epoch in range(epochs):
            for batch_x_norm, batch_p_true_norm in loader:
                # 훈련 중: 참값(정규화됨)에 무작위 섭동을 가하여 가짜 초기값 생성
                # 정규화된 공간(-1 ~ 1)에서 섭동을 줍니다.
                noise = torch.empty_like(batch_p_true_norm).uniform_(-0.5, 0.5) 
                batch_p_init_norm = batch_p_true_norm + noise 
                
                optimizer.zero_grad()
                # 관측치와 섭동된 초기값을 함께 넣어 참값을 예측하도록 훈련
                p_pred_norm = self.network(batch_x_norm, batch_p_init_norm)
                loss = criterion(p_pred_norm, batch_p_true_norm)
                loss.backward()
                optimizer.step()
                
        torch.save(self.network.state_dict(), self.weight_path)
        self.network.eval()
        print(f"[Direct ML] Training completed and saved.")

    def fit(self, t_eval, x_obs, theta_init, x_hid_init):
        # 1. Feature Engineering (동적 미분 계산 내재화)
        # x_obs의 형태: (T, obs_dim) 가정
        T, obs_dim = x_obs.shape
        
        if getattr(self.config, 'USE_DERIVATIVE', False):
            from utils import get_derivative_estimator
            method = getattr(self.config, 'DERIVATIVE_METHOD', 'finite_diff')
            kwargs = {'order': 3} if method == 'poly' else {}
            estimator = get_derivative_estimator(method, **kwargs)
            
            x_dot = np.zeros_like(x_obs)
            for i in range(obs_dim):
                x_dot[:, i] = estimator.estimate(t_eval, x_obs[:, i])
                
            x_obs_features = np.concatenate([x_obs, x_dot], axis=1)
        else:
            x_obs_features = x_obs

        # 2. 정규화 및 텐서 변환
        x_obs_flat = x_obs_features.flatten()
        x_obs_t = torch.tensor(x_obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_obs_norm = self.normalizer.normalize_inputs(x_obs_t, variable_type='observed')
        
        p_init_t = torch.tensor(theta_init, dtype=torch.float32, device=self.device).unsqueeze(0)
        p_init_norm = self.normalizer.normalize_params(p_init_t)

        start_time = time.time()
        with torch.no_grad():
            # 3. 정규화된 관측치와 정규화된 섭동값을 네트워크에 입력
            p_pred_norm = self.network(x_obs_norm, p_init_norm)
        exec_time = time.time() - start_time

        # 4. 출력된 값을 원래 물리적 스케일로 역정규화
        theta_hat_t = self.normalizer.denormalize_params(p_pred_norm)
        theta_hat = theta_hat_t.squeeze(0).cpu().numpy()
        
        return theta_hat, np.zeros_like(x_hid_init), exec_time