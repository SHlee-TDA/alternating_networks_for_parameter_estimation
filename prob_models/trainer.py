import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from prob_models.models import elbo_loss 

class ProbTrainer:
    """
    Manages the independent optimization process for the CVAE networks.
    """
    def __init__(self, train_loader, val_loader, config, 
                 hidden_cvae=None, param_cvae=None, baseline_cvae=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_baseline = getattr(config, 'RUN_BASELINE', False)
        
        # Setup results directory
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME + "_prob")
        os.makedirs(self.results_path, exist_ok=True)

        if self.is_baseline:
            assert baseline_cvae is not None, "Baseline CVAE model must be provided when RUN_BASELINE is True."
            self.baseline_cvae = baseline_cvae.to(config.DEVICE)
            self.opt_base = torch.optim.Adam(self.baseline_cvae.parameters(), lr=config.LEARNING_RATE)
            self.sched_base = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_base, mode='min', factor=0.5, patience=100)
        else:
            assert hidden_cvae is not None and param_cvae is not None, "Both Hidden and Parameter CVAE models must be provided when RUN_BASELINE is False."
            self.hidden_cvae = hidden_cvae.to(config.DEVICE)
            self.param_cvae = param_cvae.to(config.DEVICE)
            
            self.opt_hidden = torch.optim.Adam(self.hidden_cvae.parameters(), lr=config.LEARNING_RATE)
            self.opt_param = torch.optim.Adam(self.param_cvae.parameters(), lr=config.LEARNING_RATE)

            self.sched_hidden = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_hidden, mode='min', factor=0.5, patience=100)
            self.sched_param = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_param, mode='min', factor=0.5, patience=100)

    def train(self):
        mode_str = "Single CVAE (Baseline)" if self.is_baseline else "Iterative CVAEs (Proposed)"
        print(f"\nTraining {mode_str} for {self.config.SYSTEM_NAME}...")
        history = defaultdict(list)
        best_val_loss = np.inf
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            if self.is_baseline:
                self.baseline_cvae.train()
            else:
                self.hidden_cvae.train()
                self.param_cvae.train()
            
            epoch_train_losses = defaultdict(list)
            
            # KL Annealing
            warmup_epochs = getattr(self.config, 'KL_WARMUP_EPOCHS', 20)
            max_beta = getattr(self.config, 'KL_MAX_BETA', 1.0)
            beta = min(max_beta, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)
            
            for x_batch, y_batch, p_batch in pbar:
                x_batch = x_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                if self.is_baseline:
                    self.opt_base.zero_grad()
                    p_hat, mu, logvar = self.baseline_cvae(x_batch, p_batch)
                    loss_base, recon_base, kld_base = elbo_loss(p_hat, p_batch, mu, logvar, beta)
                    loss_base.backward()
                    torch.nn.utils.clip_grad_norm_(self.baseline_cvae.parameters(), 1.0)
                    self.opt_base.step()
                    
                    epoch_train_losses['baseline_loss'].append(loss_base.item())
                    epoch_train_losses['baseline_recon'].append(recon_base.item())
                    pbar.set_postfix(loss=loss_base.item(), beta=beta)
                else:
                    # 1. Condition Noise (Theorem 1: Local Contraction 유도)
                    # C1 fix (2026-07-13): read the canonical config keys CONDITION_NOISE_STD_*
                    # (DISCUSSION.md §E N1). The old COND_NOISE_STD_* keys never existed in
                    # config, so condition noise was silently pinned to 0.05 regardless of setting.
                    cond_std_y = getattr(self.config, 'CONDITION_NOISE_STD_Y', 0.05)
                    cond_std_p = getattr(self.config, 'CONDITION_NOISE_STD_P', 0.05)
                    
                    y_cond_noisy = y_batch + torch.randn_like(y_batch) * cond_std_y
                    p_cond_noisy = p_batch + torch.randn_like(p_batch) * cond_std_p

                    # 2. Target Noise (Theorem 2: Score Matching & Attraction 유도)
                    target_std_y = getattr(self.config, 'TARGET_NOISE_STD_Y', 0.05)
                    target_std_p = getattr(self.config, 'TARGET_NOISE_STD_P', 0.05)
                    
                    y_target_noisy = y_batch + torch.randn_like(y_batch) * target_std_y
                    p_target_noisy = p_batch + torch.randn_like(p_batch) * target_std_p
                    
                    # ==========================================
                    # Train Hidden State CVAE (H_net)
                    # ==========================================
                    self.opt_hidden.zero_grad()
                    # Encoder Input: Target Noisy (y_target_noisy)
                    # Condition Input: Condition Noisy (p_cond_noisy)
                    y_hat, mu_h, logvar_h = self.hidden_cvae(x_batch, y_target_noisy, p_cond_noisy)
                    
                    # Loss Calculation: Reconstruct the CLEAN target (y_batch)
                    loss_h, recon_h, kld_h = elbo_loss(y_hat, y_batch, mu_h, logvar_h, beta)
                    
                    loss_h.backward()
                    torch.nn.utils.clip_grad_norm_(self.hidden_cvae.parameters(), 1.0)
                    self.opt_hidden.step()

                    # ==========================================
                    # Train Parameter CVAE (P_net)
                    # ==========================================
                    self.opt_param.zero_grad()
                    # Encoder Input: Target Noisy (p_target_noisy)
                    # Condition Input: Condition Noisy (y_cond_noisy)
                    p_hat, mu_p, logvar_p = self.param_cvae(x_batch, y_cond_noisy, p_target_noisy)                
                    # Loss Calculation: Reconstruct the CLEAN target (p_batch)
                    loss_p, recon_p, kld_p = elbo_loss(p_hat, p_batch, mu_p, logvar_p, beta)
                    
                    loss_p.backward()
                    torch.nn.utils.clip_grad_norm_(self.param_cvae.parameters(), 1.0)
                    self.opt_param.step()

                    # Logging
                    epoch_train_losses['hidden_loss'].append(loss_h.item())
                    epoch_train_losses['param_loss'].append(loss_p.item())
                    epoch_train_losses['hidden_recon'].append(recon_h.item())
                    epoch_train_losses['param_recon'].append(recon_p.item())

                    pbar.set_postfix(h_loss=loss_h.item(), p_loss=loss_p.item(), beta=beta)

            # --- Validation Phase ---
            val_losses = self.evaluate(self.val_loader, beta)
            
            if self.is_baseline:
                combined_val_loss = val_losses['val_base_loss']
                combined_val_recon = val_losses['val_base_recon']
                self.sched_base.step(combined_val_recon)
                
                if epoch % 1000 == 0 or epoch == self.config.EPOCHS - 1:
                    print(f"Epoch {epoch+1:04d}/{self.config.EPOCHS} | Beta: {beta:.2f} | "
                          f"Baseline Loss: {combined_val_loss:.4f} | Baseline Recon: {combined_val_recon:.4f}")
            else:
                # Update Schedulers based on combined validation loss
                combined_val_loss = val_losses['val_hidden_loss'] + val_losses['val_param_loss']
                combined_val_recon = val_losses['val_hidden_recon'] + val_losses['val_param_recon']
                
                self.sched_hidden.step(combined_val_recon)
                self.sched_param.step(combined_val_recon)
                
                
                if epoch % 1000 == 0 or epoch == self.config.EPOCHS - 1:
                    h_loss = np.mean(epoch_train_losses['hidden_loss'])
                    p_loss = np.mean(epoch_train_losses['param_loss'])
                    print(f"Epoch {epoch+1:04d}/{self.config.EPOCHS} | Beta: {beta:.2f} | "
                        f"HiddenCVAE Loss: {h_loss:.4f} | ParamCVAE Loss: {p_loss:.4f} | "
                        f"Val Loss: {combined_val_loss:.4f} | Val Recon: {combined_val_recon:.4f}")
                
            # Update History
            for k, v in epoch_train_losses.items():
                history[f'train_{k}'].append(np.mean(v))
            for k, v in val_losses.items():
                history[k].append(v)
            
            # Checkpointing & Early Stopping 
            min_delta = getattr(self.config, 'EARLY_STOPPING_MIN_DELTA', 1e-4)
            if combined_val_recon < best_val_loss - min_delta:
                best_val_loss = combined_val_recon  
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_loss)
            else:
                patience_counter += 1
                
            if getattr(self.config, 'USE_EARLY_STOPPING', False) and patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n[Early Stopping] Epoch {epoch+1} (No improvement in Val Recon Loss)")
                break

        self._save_final_artifacts()
        if self.is_baseline:
            return self.baseline_cvae, history
        else:
            return self.hidden_cvae, self.param_cvae, history

    @torch.no_grad()
    def evaluate(self, loader, beta):
        if self.is_baseline:
            self.baseline_cvae.eval()
        else:
            self.hidden_cvae.eval()
            self.param_cvae.eval()
            
        losses = defaultdict(list)
        
        for x_batch, y_batch, p_batch in loader:
            x_batch = x_batch.to(self.config.DEVICE)
            y_batch = y_batch.to(self.config.DEVICE)
            p_batch = p_batch.to(self.config.DEVICE)
            
            if self.is_baseline:
                p_hat, mu_b, logvar_b = self.baseline_cvae(x_batch, p_batch)
                loss_b, recon_b, _ = elbo_loss(p_hat, p_batch, mu_b, logvar_b, beta)
                losses['val_base_loss'].append(loss_b.item())
                losses['val_base_recon'].append(recon_b.item())
            else:
                y_hat, mu_h, logvar_h = self.hidden_cvae(x_batch, y_batch, p_batch)
                loss_h, recon_h, _ = elbo_loss(y_hat, y_batch, mu_h, logvar_h, beta)
                
                p_hat, mu_p, logvar_p = self.param_cvae(x_batch, y_batch, p_batch)
                loss_p, recon_p, _ = elbo_loss(p_hat, p_batch, mu_p, logvar_p, beta)
                
                losses['val_hidden_loss'].append(loss_h.item())
                losses['val_param_loss'].append(loss_p.item())
                losses['val_hidden_recon'].append(recon_h.item())
                losses['val_param_recon'].append(recon_p.item())
            
        return {k: np.mean(v) for k, v in losses.items()}

    def _save_checkpoint(self, epoch, loss):
        save_path = os.path.join(self.results_path, 'best_prob_model.pth')
        state = {'epoch': epoch + 1, 'best_val_loss': loss}
        
        if self.is_baseline:
            state['baseline_cvae_state_dict'] = self.baseline_cvae.state_dict()
            state['opt_base_state_dict'] = self.opt_base.state_dict()
        else:
            state['hidden_cvae_state_dict'] = self.hidden_cvae.state_dict()
            state['param_cvae_state_dict'] = self.param_cvae.state_dict()
            state['opt_hidden_state_dict'] = self.opt_hidden.state_dict()
            state['opt_param_state_dict'] = self.opt_param.state_dict()
            
        torch.save(state, save_path)

    def _save_final_artifacts(self):
        if self.is_baseline:
            torch.save(self.baseline_cvae.state_dict(), os.path.join(self.results_path, 'baseline_cvae.pth'))
        else:
            torch.save(self.hidden_cvae.state_dict(), os.path.join(self.results_path, 'hidden_cvae.pth'))
            torch.save(self.param_cvae.state_dict(), os.path.join(self.results_path, 'param_cvae.pth'))