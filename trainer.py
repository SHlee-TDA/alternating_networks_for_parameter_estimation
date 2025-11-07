# trainer.py
import os
import pickle
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    """모델 학습 과정을 담당하는 클래스"""
    def __init__(self, 
                 f_theta, g_phi, 
                 train_loader, val_loader, 
                 config, normalizer):
        self.f_theta = f_theta.to(config.DEVICE)
        self.g_phi = g_phi.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.normalizer = normalizer
        self.optimizer = torch.optim.Adam(
            list(self.f_theta.parameters()) + list(self.g_phi.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.loss_fn = nn.MSELoss()
        
        # 결과 저장 경로
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)
        
    def train(self):
        print(f"Training models for {self.config.EXPERIMENT_NAME}...")
        history = defaultdict(list)
        best_val_loss = np.inf
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            self.f_theta.train()
            self.g_phi.train()
            
            epoch_train_losses = defaultdict(list)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)
            for x_batch, y_batch, p_batch in pbar:
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
                
                # logging
                epoch_train_losses['total_loss'].append(total_loss.item())
                epoch_train_losses['loss_f'].append(loss_f.item())
                epoch_train_losses['loss_g'].append(loss_g.item())
                if self.config.USE_CONSISTENCY_LOSS:
                    epoch_train_losses['loss_consistency'].append(loss_consistency.item())
                
                total_loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=total_loss.item())
            # Epoch 종료 후 검증
            val_losses = self.evaluate(self.val_loader)
            
            for k, v in epoch_train_losses.items():
                history[f'train_{k}'].append(np.mean(v))
            for k, v in val_losses.items():
                history[f'val_{k}'].append(v)
            
            if epoch % 100 == 0 or epoch == self.config.EPOCHS - 1:
                print(f"Epoch {epoch+1:04d} | Train Loss: {history['train_total_loss'][-1]:.4f} | Val Loss: {history['val_total_loss'][-1]:.4f}")
            
            if self.config.USE_EARLY_STOPPING:
                current_val_loss = history['val_total_loss'][-1]
                
                # 검증 손실이 개선되었는지 확인
                if current_val_loss < best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                    # 개선됨: best loss 업데이트, patience 초기화
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    
                    # Best 모델 가중치 저장 (선택적이지만 권장)
                    print(f"  -> New best validation loss: {best_val_loss:.4f}. Saving best model.")
                    torch.save(self.f_theta.state_dict(), os.path.join(self.results_path, 'f_theta_best.pth'))
                    torch.save(self.g_phi.state_dict(), os.path.join(self.results_path, 'g_phi_best.pth'))
                
                else:
                    # 개선 없음: patience 증가
                    patience_counter += 1
                
                # Patience 한계 도달 시 학습 중단
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\n[Early Stopping] Validation loss did not improve for {self.config.EARLY_STOPPING_PATIENCE} epochs.")
                    print(f"Stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    break # Epoch loop 탈출
                
                
        print(f"Training complete. Saving artifacts to {self.results_path}")
        # 1. 모델 가중치 저장
        torch.save(self.f_theta.state_dict(), os.path.join(self.results_path, 'f_theta.pth'))
        torch.save(self.g_phi.state_dict(), os.path.join(self.results_path, 'g_phi.pth'))

        # 2. Normalizer 저장 (추론 시 필수)
        with open(os.path.join(self.results_path, 'normalizer.pkl'), 'wb') as f:
            pickle.dump(self.normalizer, f)

        # 3. 사용된 config 저장 (모델 구조 복원 시 필수)
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('__') and not callable(v)}
        config_dict['DEVICE'] = str(config_dict.get('DEVICE')) # non-serializable 변환
        config_dict.pop('EXPERIMENTS', None) # 전체 리스트는 제외

        with open(os.path.join(self.results_path, 'config_run.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return self.f_theta, self.g_phi, history
    
    
    def weight_product_penalty(self):
        """모델의 모든 층에 대해 가중치 행렬의 스펙트럴 노름 곱을 계산합니다."""
        def spectral_norm(module):
            if hasattr(module, 'weight'):
                weight = getattr(module, 'weight_orig', module.weight) # <-- 수정됨
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
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.f_theta.eval()
        self.g_phi.eval()
        losses = defaultdict(list)
        for x_batch, y_batch, p_batch in loader:
            x_batch = x_batch.to(self.config.DEVICE)
            y_batch = y_batch.to(self.config.DEVICE)
            p_batch = p_batch.to(self.config.DEVICE)
            p_batch_norm = self.normalizer.normalize(p_batch)

            y_pred_f = self.f_theta(x_batch, p_batch)
            loss_f = self.loss_fn(y_pred_f, y_batch)

            p_pred_g = self.g_phi(x_batch, y_batch)
            p_pred_g_norm = self.normalizer.normalize(p_pred_g)
            loss_g = self.loss_fn(p_pred_g_norm, p_batch_norm)

            total_loss = loss_f + loss_g
            losses['total_loss'].append(total_loss.item())
            losses['loss_f'].append(loss_f.item())
            losses['loss_g'].append(loss_g.item())

            if self.config.USE_CONSISTENCY_LOSS:
                p_reconstructed_norm = self.g_phi(x_batch, y_pred_f)
                loss_consistency = self.loss_fn(p_reconstructed_norm, p_batch_norm)
                # total_loss += self.config.CONSISTENCY_LOSS_LAMBDA * loss_consistency
                losses['loss_consistency'].append(loss_consistency.item())

        # 평균 손실 반환
        return {k: np.mean(v) for k, v in losses.items()}