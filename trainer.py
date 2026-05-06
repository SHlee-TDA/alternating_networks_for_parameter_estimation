# trainer.py
import os
import pickle
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from losses import get_loss_function

class Trainer:
    """
    Manages the optimization process for f_theta and g_phi.
    
    Args:
        f_theta (nn.Module): Network predicting hidden states from params.
        g_phi (nn.Module): Network estimating params from hidden states.
        train_loader (DataLoader): Yields normalized (X, Y, P) batches.
        val_loader (DataLoader): Validation set.
        config (object): Configuration namespace.
    """
    def __init__(self, 
                 train_loader, val_loader, 
                 config,
                 hidden_net=None, param_net=None,
                 baseline_net=None
                 ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup results directory
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)
        self.spectral_cache = {}

        # Model initialization
        if getattr(self.config, 'RUN_BASELINE', False):
            self.baseline_net = baseline_net.to(config.DEVICE)
            
            self.optimizer = torch.optim.Adam(
                self.baseline_net.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                eps=1e-4
            )
            self.loss_fn = nn.MSELoss().to(config.DEVICE)
            
        else:
            self.hidden_net = hidden_net.to(config.DEVICE)
            self.param_net = param_net.to(config.DEVICE)
            
            # 기존 Joint 옵티마이저
            self.optimizer = torch.optim.Adam(
                list(self.hidden_net.parameters()) + list(self.param_net.parameters()),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                eps=1e-4
            )
            self.loss_fn = get_loss_function(
                f_theta=self.hidden_net,
                g_phi=self.param_net,
                config=self.config
            ).to(config.DEVICE)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.99999999999, patience=20, min_lr=1e-8, verbose=True
        )

    def train(self):
        """
        Executes the main training loop.
        
        Returns:
            hidden_net, param_net, history (dict of loss logs)
        """
        print(f"Training models for {self.config.EXPERIMENT_NAME}...")

        
        history = defaultdict(list)
        best_val_loss = np.inf
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            if getattr(self.config, 'RUN_BASELINE', False):
                self.baseline_net.train()
            else:
                self.hidden_net.train()
                self.param_net.train()

            
            epoch_train_losses = defaultdict(list)
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)
            
            for x_batch, y_batch, p_batch in pbar:
                # Inputs are already normalized in DataLoader
                x_batch = x_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)

                self.optimizer.zero_grad()
                # # --- 1. Forward Loss (State Estimation) ---
                # # f_theta: (X, P) -> Y_hat
                # y_pred = self.f_theta(x_batch, p_batch)
                # loss_f = self.loss_fn(y_pred, y_batch)
                
                # # --- 2. Inverse Loss (Parameter Estimation) ---
                # # g_phi: (X, Y) -> P_hat
                # p_pred = self.g_phi(x_batch, y_batch)
                # loss_g = self.loss_fn(p_pred, p_batch)
                
                
                # total_loss = loss_f + loss_g
                
                # # --- 3. Consistency Loss (Fixed Point Regularization) ---
                # # P -> f_theta -> Y_hat -> g_phi -> P_recon
                # # Enforces bijections on the manifold defined by the ODE
                # loss_consistency = torch.tensor(0.0, device=self.config.DEVICE)
                
                # if self.config.USE_CONSISTENCY_LOSS:
                #     p_reconstructed = self.g_phi(x_batch, y_pred)
                #     loss_consistency = self.loss_fn(p_reconstructed, p_batch)
                #     total_loss += self.config.CONSISTENCY_LOSS_LAMBDA * loss_consistency
                
                # # --- 4. Spectral Penalty (Optional) ---
                # # Soft constraint to encourage Lipschitz constant < 1
                # if getattr(self.config, 'USE_SPECTRAL_PENALTY', False):
                #     f_penalty, g_penalty = self.compute_spectral_product_penalty()
                #     total_loss += (f_penalty + g_penalty) * 0.01 # Scaling factor for penalty

                if getattr(self.config, 'RUN_BASELINE', False):
                    p_pred = self.baseline_net(x_batch)
                    total_loss = self.loss_fn(p_pred, p_batch)
                    metrics = {'mse_loss_baseline': total_loss.item()}
                else:
                    total_loss, metrics = self.loss_fn(x_batch, y_batch, p_batch)
                    
                total_loss.backward()
                
                # Gradient clipping
                #torch.nn.utils.clip_grad_norm_(self.hidden_net.parameters(), max_norm=1.0)
                #torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # logging
                # epoch_train_losses['total_loss'].append(total_loss.item())
                # epoch_train_losses['loss_f'].append(loss_f.item())
                # epoch_train_losses['loss_g'].append(loss_g.item())
                # epoch_train_losses['loss_consistency'].append(loss_consistency.item())
                for k, v in metrics.items():
                    epoch_train_losses[k].append(v)
                
                epoch_train_losses['total_loss'].append(total_loss.item())
                
                pbar.set_postfix(loss=total_loss.item())

            # --- Validation Phase ---
            val_losses = self.evaluate(self.val_loader)
            self.scheduler.step(val_losses['total_loss'])
            
            # Update History
            for k, v in epoch_train_losses.items():
                history[f'train_{k}'].append(np.mean(v))
            for k, v in val_losses.items():
                history[f'val_{k}'].append(v)
            
            # Logging & Checkpointing
            if epoch % 1000 == 0 or epoch == self.config.EPOCHS - 1:
                print(f"Epoch {epoch+1:04d} | Train: {history['train_total_loss'][-1]:.4f} | Val: {history['val_total_loss'][-1]:.4f}")
            
            # Early Stopping Logic
            if self.config.USE_EARLY_STOPPING:
                current_val_loss = history['val_total_loss'][-1]
                if current_val_loss < best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, best_val_loss)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\n[Early Stopping] Epoch {epoch+1}")
                    break
        
        self._save_final_artifacts()
        
        if getattr(self.config, 'RUN_BASELINE', False):
            return self.baseline_net, history
        return self.hidden_net, self.param_net, history    
       
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Computes loss on the validation/test set."""
        if getattr(self.config, 'RUN_BASELINE', False):
            self.baseline_net.eval()
        else:
            self.hidden_net.eval()
            self.param_net.eval()   
        
        losses = defaultdict(list)
        
        for x_batch, y_batch, p_batch in loader:
            x_batch = x_batch.to(self.config.DEVICE)
            y_batch = y_batch.to(self.config.DEVICE)
            p_batch = p_batch.to(self.config.DEVICE)
            
            # Forward
            # y_pred = self.f_theta(x_batch, p_batch)
            # loss_f = self.loss_fn(y_pred, y_batch)
            
            # # Inverse
            # p_pred = self.g_phi(x_batch, y_batch)
            # loss_g = self.loss_fn(p_pred, p_batch)
            
            # # Consistency
            # loss_const = torch.tensor(0.0, device=self.config.DEVICE)
            # if getattr(self.config, 'USE_CONSISTENCY_LOSS', False):
            #     p_recon = self.g_phi(x_batch, y_pred)
            #     loss_const = self.loss_fn(p_recon, p_batch)
            
            # total_loss = loss_f + loss_g
            # if self.config.USE_CONSISTENCY_LOSS:
            #     total_loss += self.config.CONSISTENCY_LOSS_LAMBDA * loss_const
            
            # # Logging
            # losses['total_loss'].append(total_loss.item())
            # losses['loss_f'].append(loss_f.item())
            # losses['loss_g'].append(loss_g.item())
            # losses['loss_consistency'].append(loss_const.item())
            if getattr(self.config, 'RUN_BASELINE', False):
                p_pred = self.baseline_net(x_batch)
                total_loss = self.loss_fn(p_pred, p_batch)
                metrics = {'mse_loss_baseline': total_loss.item()}
            else:
                total_loss, metrics = self.loss_fn(x_batch, y_batch, p_batch)
                
            for k, v in metrics.items():
                losses[k].append(v)
            
            losses['total_loss'].append(total_loss.item())

        return {k: np.mean(v) for k, v in losses.items()}
    
    def estimate_spectral_norm(self, weight, n_power_iterations=5, layer_id=None):
        """
        Approximates the spectral norm (sigma_max) via Power Iteration.
        
        Note:
            This is used for explicit regularization terms in the loss function.
            It uses a cache to persist singular vectors (u, v) across steps for faster convergence.
        """
        out_dim, in_dim = weight.shape
        
        # Initialize Cache if needed
        if layer_id not in self.spectral_cache:
            u = torch.randn(out_dim, device=weight.device)
            u = u / u.norm()
            self.spectral_cache[layer_id] = {'u': u, 'v': torch.randn(in_dim, device=weight.device)}
        
        u = self.spectral_cache[layer_id]['u']
        
        # Power Iteration (No Grad for updates)
        with torch.no_grad():
            for _ in range(n_power_iterations):
                v = torch.mv(weight.t(), u)
                v = v / (v.norm() + 1e-12)
                u = torch.mv(weight, v)
                u = u / (u.norm() + 1e-12)
            
            self.spectral_cache[layer_id]['u'] = u
            self.spectral_cache[layer_id]['v'] = v
            
        # Calculate differentiable spectral norm
        # sigma = u^T * W * v
        v = self.spectral_cache[layer_id]['v']
        weight_v = torch.mv(weight, v)
        sigma = torch.dot(u, weight_v)
        
        return sigma
    
    def compute_spectral_product_penalty(self):
        """
        Computes the product of spectral norms across all layers.
        Used to encourage the global Lipschitz constant to be minimal.
        """
        def get_network_penalty(network, prefix):
            prod = 1.0
            for name, module in network.named_modules():
                if isinstance(module, nn.Linear):
                    layer_id = f"{prefix}_{name}"
                    # Handle weight_orig if spectral_norm hook is applied
                    weight = getattr(module, 'weight_orig', module.weight)
                    sigma = self.estimate_spectral_norm(weight, layer_id=layer_id)
                    prod = prod * sigma
            return prod

        f_penalty = get_network_penalty(self.f_theta, "f")
        g_penalty = get_network_penalty(self.g_phi, "g")
        
        return f_penalty, g_penalty
    
    def _save_checkpoint(self, epoch, loss):
        filename = 'best_baseline_model.pth' if getattr(self.config, 'RUN_BASELINE', False) else 'best_model.pth'
        save_path = os.path.join(self.results_path, filename)
        
        try:
            if getattr(self.config, 'RUN_BASELINE', False):
                torch.save({
                    'baseline_state_dict': self.baseline_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': loss
                }, save_path)
            else:
                torch.save({
                    'Hnet_state_dict': self.hidden_net.state_dict(),
                    'Pnet_state_dict': self.param_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': loss
                }, save_path)
                
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def _save_final_artifacts(self):
        if getattr(self.config, 'RUN_BASELINE', False):
            torch.save(self.baseline_net.state_dict(), os.path.join(self.results_path, 'baseline_net.pth'))
        else:
            torch.save(self.hidden_net.state_dict(), os.path.join(self.results_path, 'Hnet.pth'))
            torch.save(self.param_net.state_dict(), os.path.join(self.results_path, 'Pnet.pth'))    