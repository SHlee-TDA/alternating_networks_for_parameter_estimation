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
    """모델 학습 과정을 담당하는 클래스
    
    [Optimization]
    1. Replaced SVD with Power Iteration for fast spectral norm estimation.
    2. Decoupled Spectral Normalization (Hard) and Product Penalty (Soft).
    
    """
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
        self.optimizer = torch.optim.AdamW(
            list(self.f_theta.parameters()) + list(self.g_phi.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.loss_fn = nn.MSELoss()
        
        # 결과 저장 경로
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)
        
        # [Power Iteration Cache]
        # 각 레이어별 u, v 벡터를 저장하여 다음 step의 초기값으로 사용 (수렴 속도 향상)
        self.spectral_cache = {}

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
                
                # 1. Forward Loss
                y_pred_f = self.f_theta(x_batch, p_batch)
                loss_f = self.loss_fn(y_pred_f, y_batch)
                
                # 2. Inverse Loss
                p_pred_g = self.g_phi(x_batch, y_batch)
                p_pred_g_norm = self.normalizer.normalize(p_pred_g)
                loss_g = self.loss_fn(p_pred_g_norm, p_batch_norm)
                
                total_loss = loss_f + loss_g
                
                # 3. Consistency Loss
                if self.config.USE_CONSISTENCY_LOSS:
                    p_reconstructed_norm = self.g_phi(x_batch, y_pred_f)
                    loss_consistency = self.loss_fn(p_reconstructed_norm, p_batch_norm)
                    total_loss += self.config.CONSISTENCY_LOSS_LAMBDA * loss_consistency
                
                # 4. Spectral Norm Penalty
                # config에 USE_SPECTRAL_PENALTY 옵션이 True일 때만 계산
                penalty_loss = torch.tensor(0.0, device=self.config.DEVICE)
                if getattr(self.config, 'USE_SPECTRAL_PENALTY', False):
                    f_penalty, g_penalty = self.compute_spectral_product_penalty()
                    penalty_loss = f_penalty + g_penalty
                    total_loss += penalty_loss

                
                
                # logging
                epoch_train_losses['total_loss'].append(total_loss.item())
                epoch_train_losses['loss_f'].append(loss_f.item())
                epoch_train_losses['loss_g'].append(loss_g.item())
                if self.config.USE_CONSISTENCY_LOSS:
                    epoch_train_losses['loss_consistency'].append(loss_consistency.item())
                
                total_loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=total_loss.item())

            # Validation
            val_losses = self.evaluate(self.val_loader)
            
            for k, v in epoch_train_losses.items():
                history[f'train_{k}'].append(np.mean(v))
            for k, v in val_losses.items():
                history[f'val_{k}'].append(v)
            
            # Logging & Early Stopping (기존 로직 유지)
            if epoch % 100 == 0 or epoch == self.config.EPOCHS - 1:
                print(f"Epoch {epoch+1:04d} | Train: {history['train_total_loss'][-1]:.4f} | Val: {history['val_total_loss'][-1]:.4f}")
            
            if self.config.USE_EARLY_STOPPING:
                current_val_loss = history['val_total_loss'][-1]
                if current_val_loss < best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, best_val_loss) # 저장 로직 메서드 분리
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\n[Early Stopping] Epoch {epoch+1}")
                    break
        
        self._save_final_artifacts()
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
    
    def estimate_spectral_norm(self, weight, n_power_iterations=5, layer_id=None):
        """
        Power Iteration을 사용하여 Spectral Norm(최대 특이값)을 근사 계산합니다.
        Gradient가 끊기지 않도록 구현합니다.
        """
        out_dim, in_dim = weight.shape
        
        # Cache Init
        if layer_id not in self.spectral_cache:
            u = torch.randn(out_dim, device=weight.device)
            u = u / u.norm()
            v = torch.randn(in_dim, device=weight.device) # dummy
            self.spectral_cache[layer_id] = {'u': u, 'v': v}
        
        u = self.spectral_cache[layer_id]['u']
        
        # Power Iteration
        # detach()를 사용하여 u 업데이트 과정 자체에는 그라디언트가 흐르지 않게 함 (메모리 절약)
        # 하지만 최종 s 계산에는 weight가 관여하므로 weight에 대한 그라디언트는 계산됨.
        with torch.no_grad():
            for _ in range(n_power_iterations):
                # v = W^T * u
                v = torch.mv(weight.t(), u)
                v = v / (v.norm() + 1e-12)
                
                # u = W * v
                u = torch.mv(weight, v)
                u = u / (u.norm() + 1e-12)
            
            # Update Cache
            self.spectral_cache[layer_id]['u'] = u
            self.spectral_cache[layer_id]['v'] = v
            
        # Spectral Norm Calculation (Differentiable)
        # sigma = u^T * W * v
        v = self.spectral_cache[layer_id]['v'] # Updated v
        weight_v = torch.mv(weight, v)
        sigma = torch.dot(u, weight_v)
        
        return sigma
    
    def compute_spectral_product_penalty(self):
        """
        Power Iteration을 사용하여 두 네트워크의 Spectral Norm Product Penalty를 계산
        """
        # F_theta Penalty
        f_prod = 1.0
        for name, module in self.f_theta.named_modules():
            if isinstance(module, nn.Linear):
                # layer_id를 고유하게 생성하여 캐싱 활용
                layer_id = f"f_{name}"
                weight = getattr(module, 'weight_orig', module.weight) # spectral_norm 적용된 경우 대비
                sigma = self.estimate_spectral_norm(weight, layer_id=layer_id)
                f_prod = f_prod * sigma
        
        # G_phi Penalty
        g_prod = 1.0
        for name, module in self.g_phi.named_modules():
            if isinstance(module, nn.Linear):
                layer_id = f"g_{name}"
                weight = getattr(module, 'weight_orig', module.weight)
                sigma = self.estimate_spectral_norm(weight, layer_id=layer_id)
                g_prod = g_prod * sigma

        # Penalty term: max(0, product - 1)^2 형태 등으로 줄 수 있으나
        # 여기서는 단순 product 합을 줄이는 방향으로 설정 (기존 의도 유지)
        # 만약 product < 1 을 강제하고 싶다면 hinge loss 형태로 변경 가능: torch.relu(prod - 1.0)
        # 하지만 기존 코드가 product 자체를 loss로 썼으므로 유지.
        
        return f_prod, g_prod
    

    def _save_checkpoint(self, epoch, loss):
        save_path = os.path.join(self.results_path, 'best_model.pth')
        try:
            torch.save({
                'f_theta_state_dict': self.f_theta.state_dict(),
                'g_phi_state_dict': self.g_phi.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': loss
            }, save_path)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def _save_final_artifacts(self):
        torch.save(self.f_theta.state_dict(), os.path.join(self.results_path, 'f_theta.pth'))
        torch.save(self.g_phi.state_dict(), os.path.join(self.results_path, 'g_phi.pth'))
        with open(os.path.join(self.results_path, 'normalizer.pkl'), 'wb') as f:
            pickle.dump(self.normalizer, f)
        
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('__') and not callable(v)}
        config_dict['DEVICE'] = str(config_dict.get('DEVICE'))
        config_dict.pop('EXPERIMENTS', None)
        with open(os.path.join(self.results_path, 'config_run.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

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
            
            # Consistency Loss Evaluation
            if getattr(self.config, 'USE_CONSISTENCY_LOSS', False):
                p_recon = self.g_phi(x_batch, y_pred_f)
                loss_cons = self.loss_fn(p_recon, p_batch_norm)
                losses['loss_consistency'].append(loss_cons.item())
                # Val loss에는 consistency를 더하지 않는 것이 일반적이나, 설정에 따름.
                # 여기서는 Pure Performance만 보려면 제외, Optimization Check면 포함.
                # 일관성을 위해 Train과 동일하게 포함하지 않고 모니터링만 함.

            losses['total_loss'].append(total_loss.item())
            losses['loss_f'].append(loss_f.item())
            losses['loss_g'].append(loss_g.item())

        return {k: np.mean(v) for k, v in losses.items()}