# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    """모델 학습 과정을 담당하는 클래스"""
    def __init__(self, f_theta, g_phi, train_loader, config, normalizer):
        self.f_theta = f_theta.to(config.DEVICE)
        self.g_phi = g_phi.to(config.DEVICE)
        self.train_loader = train_loader
        self.config = config
        self.normalizer = normalizer
        self.optimizer = torch.optim.Adam(
            list(self.f_theta.parameters()) + list(self.g_phi.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.loss_fn = nn.MSELoss()

    def train(self):
        print(f"Training models for {self.config.EXPERIMENT_NAME}...")
        for epoch in range(self.config.EPOCHS):
            for x_batch, y_batch, p_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False):
                x_batch = x_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)

                p_batch_norm = self.normalizer.normalize(p_batch)
                
                self.optimizer.zero_grad()
                
                y_pred_f = self.f_theta(x_batch, p_batch)
                loss_f = self.loss_fn(y_pred_f, y_batch)
                
                p_pred_g = self.g_phi(x_batch, y_batch)
                p_pred_g_norm = self.normalizer.normalize(p_pred_g)
                loss_g = self.loss_fn(p_pred_g_norm, p_batch_norm)
                
                total_loss = loss_f + loss_g
                
                if self.config.USE_CONSISTENCY_LOSS:
                    p_reconstructed_norm = self.g_phi(x_batch, y_pred_f)
                    loss_consistency = self.loss_fn(p_reconstructed_norm, p_batch_norm)
                    total_loss += self.config.CONSISTENCY_LOSS_LAMBDA * loss_consistency
                
                if self.config.USE_SPECTRAL_NORM:
                    f_theta_penalty, g_phi_penalty = self.weight_product_penalty()
                    spectral_norm_loss = (f_theta_penalty + g_phi_penalty)
                    total_loss += spectral_norm_loss
                    
                total_loss.backward()
                self.optimizer.step()
        
        return self.f_theta, self.g_phi
    
    
    def weight_product_penalty(self):
        """모델의 모든 층에 대해 가중치 행렬의 스펙트럴 노름 곱을 계산합니다."""
        def spectral_norm(module):
            if hasattr(module, 'weight'):
                weight = module.weight
                u, s, v = torch.svd(weight)
                return s[0]  # 최대 특이값
            return 1.0  # 가중치가 없는 경우 페널티 없음

        f_theta_penalty = 1.0
        for module in self.f_theta.modules():
            f_theta_penalty *= spectral_norm(module)

        g_phi_penalty = 1.0
        for module in self.g_phi.modules():
            g_phi_penalty *= spectral_norm(module)

        return f_theta_penalty, g_phi_penalty