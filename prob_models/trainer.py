import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from prob_models.models import elbo_loss 

class ProbabilisticTrainer:
    """
    Manages the independent optimization process for the CVAE networks.
    """
    def __init__(self, train_loader, val_loader, config, hidden_cvae, param_cvae):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup results directory
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME + "_prob")
        os.makedirs(self.results_path, exist_ok=True)

        self.hidden_net = hidden_cvae.to(config.DEVICE)
        self.param_net = param_cvae.to(config.DEVICE)
        
        self.opt_hidden = torch.optim.Adam(self.hidden_net.parameters(), lr=config.LEARNING_RATE)
        self.opt_param = torch.optim.Adam(self.param_net.parameters(), lr=config.LEARNING_RATE)

        self.sched_hidden = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_hidden, mode='min', factor=0.5, patience=10)
        self.sched_param = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_param, mode='min', factor=0.5, patience=10)

    def train(self):
        print(f"Training Probabilistic Models for {self.config.EXPERIMENT_NAME}...")
        history = defaultdict(list)
        best_val_loss = np.inf
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            self.hidden_net.train()
            self.param_net.train()
            
            epoch_train_losses = defaultdict(list)
            
            # KL Annealing
            warmup_epochs = getattr(self.config, 'KL_WARMUP_EPOCHS', 20)
            beta = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS}", leave=False)
            
            for x_batch, y_batch, p_batch in pbar:
                x_batch = x_batch.to(self.config.DEVICE)
                y_batch = y_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)

                self.opt_hidden.zero_grad()
                y_hat, mu_h, logvar_h = self.hidden_net(x_batch, y_batch, p_batch)
                loss_h, recon_h, kld_h = elbo_loss(y_hat, y_batch, mu_h, logvar_h, beta)
                
                loss_h.backward()
                torch.nn.utils.clip_grad_norm_(self.hidden_net.parameters(), 1.0)
                self.opt_hidden.step()

                self.opt_param.zero_grad()
                p_hat, mu_p, logvar_p = self.param_net(x_batch, y_batch, p_batch)
                loss_p, recon_p, kld_p = elbo_loss(p_hat, p_batch, mu_p, logvar_p, beta)
                
                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), 1.0)
                self.opt_param.step()

                # Logging
                epoch_train_losses['hidden_loss'].append(loss_h.item())
                epoch_train_losses['param_loss'].append(loss_p.item())
                epoch_train_losses['hidden_recon'].append(recon_h.item())
                epoch_train_losses['param_recon'].append(recon_p.item())

                pbar.set_postfix(h_loss=loss_h.item(), p_loss=loss_p.item(), beta=beta)

            # --- Validation Phase ---
            val_losses = self.evaluate(self.val_loader, beta)
            
            # Update Schedulers based on combined validation loss
            combined_val_loss = val_losses['val_hidden_loss'] + val_losses['val_param_loss']
            self.sched_hidden.step(combined_val_loss)
            self.sched_param.step(combined_val_loss)
            
            if epoch % 1000 == 0 or epoch == self.config.EPOCHS - 1:
                h_loss = np.mean(epoch_train_losses['hidden_loss'])
                p_loss = np.mean(epoch_train_losses['param_loss'])
                print(f"Epoch {epoch+1:04d}/{self.config.EPOCHS} | Beta: {beta:.2f} | "
                      f"Net A Loss: {h_loss:.4f} | Net B Loss: {p_loss:.4f} | "
                      f"Val Loss: {combined_val_loss:.4f}")
            
            # Update History
            for k, v in epoch_train_losses.items():
                history[f'train_{k}'].append(np.mean(v))
            for k, v in val_losses.items():
                history[k].append(v)
            
            # Checkpointing & Early Stopping 
            if combined_val_loss < best_val_loss - getattr(self.config, 'EARLY_STOPPING_MIN_DELTA', 1e-4):
                best_val_loss = combined_val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_loss)
            else:
                patience_counter += 1
                
            if getattr(self.config, 'USE_EARLY_STOPPING', False) and patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n[Early Stopping] Epoch {epoch+1}")
                break

        self._save_final_artifacts()
        return self.hidden_net, self.param_net, history

    @torch.no_grad()
    def evaluate(self, loader, beta):
        self.hidden_net.eval()
        self.param_net.eval()
        losses = defaultdict(list)
        
        for x_batch, y_batch, p_batch in loader:
            x_batch = x_batch.to(self.config.DEVICE)
            y_batch = y_batch.to(self.config.DEVICE)
            p_batch = p_batch.to(self.config.DEVICE)
            
            y_hat, mu_h, logvar_h = self.hidden_net(x_batch, y_batch, p_batch)
            loss_h, recon_h, _ = elbo_loss(y_hat, y_batch, mu_h, logvar_h, beta)
            
            p_hat, mu_p, logvar_p = self.param_net(x_batch, y_batch, p_batch)
            loss_p, recon_p, _ = elbo_loss(p_hat, p_batch, mu_p, logvar_p, beta)
            
            losses['val_hidden_loss'].append(loss_h.item())
            losses['val_param_loss'].append(loss_p.item())
            losses['val_hidden_recon'].append(recon_h.item())
            losses['val_param_recon'].append(recon_p.item())
            
        return {k: np.mean(v) for k, v in losses.items()}

    def _save_checkpoint(self, epoch, loss):
        save_path = os.path.join(self.results_path, 'best_prob_model.pth')
        torch.save({
            'hidden_net_state_dict': self.hidden_net.state_dict(),
            'param_net_state_dict': self.param_net.state_dict(),
            'opt_hidden_state_dict': self.opt_hidden.state_dict(),
            'opt_param_state_dict': self.opt_param.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': loss
        }, save_path)

    def _save_final_artifacts(self):
        torch.save(self.hidden_net.state_dict(), os.path.join(self.results_path, 'hidden_cvae.pth'))
        torch.save(self.param_net.state_dict(), os.path.join(self.results_path, 'param_cvae.pth'))