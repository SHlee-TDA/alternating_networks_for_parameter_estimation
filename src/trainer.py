# trainer.py
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.losses import get_loss_function

class Trainer:
    """
    Manages the optimization process for the Alternating Networks framework.
    
    Args:
        train_loader (DataLoader): Yields normalized (X, Y, P) batches for training.
        val_loader (DataLoader): Validation set.
        config (object): Configuration namespace containing hyperparameters.
        hidden_net (nn.Module, optional): f_theta predicting hidden states (Y) from (X, P).
        param_net (nn.Module, optional): g_phi estimating parameters (P) from (X, Y).
        baseline_net (nn.Module, optional): Single network baseline model.
    """
    def __init__(self, 
                 train_loader, val_loader, 
                 config,
                 hidden_net=None, param_net=None,
                 baseline_net=None,
                 analyzer=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.analyzer = analyzer
        
        # Setup results directory
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)
        
        self.spectral_cache = {}

        # Initialization
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
            
            # Joint optimizer for alternating minimization
            self.optimizer = torch.optim.Adam(
                list(self.hidden_net.parameters()) + list(self.param_net.parameters()),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                eps=1e-4
            )
            
            # Encapsulated loss function (See losses.py)
            self.loss_fn = get_loss_function(
                hidden_net=self.hidden_net,
                param_net=self.param_net,
                config=self.config
            ).to(config.DEVICE)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.9, patience=200, min_lr=1e-8, verbose=True
        )

    def train(self):
        """
        Executes the main training loop.
        
        Returns:
            Tuple containing the trained network(s) and the history of loss logs.
        """
        print(f"Training models for {self.config.EXPERIMENT_NAME}...")

        history = defaultdict(list)
        best_val_loss = np.inf
        patience_counter = 0
        
        # Optional: Prepare fixed samples for phase portrait visualization
        fixed_x_sample = None
        fixed_p_target = None
        if getattr(self.config, 'PLOT_PHASE_EVOLUTION', False) and not getattr(self.config, 'RUN_BASELINE', False):
            if self.analyzer is not None:
                x_val, _, p_val = next(iter(self.train_loader))
                fixed_x_sample = x_val[0:1].to(self.config.DEVICE)  # shape: (1, ...)
                fixed_p_target = self.analyzer.normalizer.denormalize_params(p_val)[0].cpu().numpy() # shape: (param_dim,)
            else:
                print("Warning: PLOT_PHASE_EVOLUTION is True, but analyzer is not provided.")
        
        
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
                
                # Forward Pass & Loss Computation
                if getattr(self.config, 'RUN_BASELINE', False):
                    p_pred = self.baseline_net(x_batch)
                    total_loss = self.loss_fn(p_pred, p_batch)
                    metrics = {'mse_loss_baseline': total_loss.item()}
                else:
                    # Loss functions (Forward, Inverse, Consistency) are handled internally
                    total_loss, metrics = self.loss_fn(x_batch, y_batch, p_batch)
                    
                total_loss.backward()
                self.optimizer.step()
                
                # Logging step metrics
                for k, v in metrics.items():
                    epoch_train_losses[k].append(v)
                
                epoch_train_losses['total_loss'].append(total_loss.item())
                pbar.set_postfix(loss=total_loss.item())

            # Validation Phase
            val_losses = self.evaluate(self.val_loader)
            self.scheduler.step(val_losses['total_loss'])
            
            # Update History
            for k, v in epoch_train_losses.items():
                history[f'train_{k}'].append(np.mean(v))
            for k, v in val_losses.items():
                history[f'val_{k}'].append(v)
            
            # Logging
            if epoch % 1000 == 0 or epoch == self.config.EPOCHS - 1:
                print(f"Epoch {epoch+1:04d} | Train: {history['train_total_loss'][-1]:.4f} | Val: {history['val_total_loss'][-1]:.4f}")
            
            # Optional: Phase Portrait Visualization
            total_eps = self.config.EPOCHS
            percentiles = [0, 25, 50, 75, 100]
            target_eps = []
            for p in percentiles:
                ep = int((total_eps - 1) * (p / 100))
                target_eps.append(ep)

            if getattr(self.config, 'PLOT_PHASE_EVOLUTION', False) and not getattr(self.config, 'RUN_BASELINE', False):
                if epoch in target_eps:
                    if self.analyzer is not None and fixed_x_sample is not None:
                        self.hidden_net.eval()
                        self.param_net.eval()
                        
                        self.analyzer.plot_phase_portraits(fixed_x_sample, fixed_p_target, epoch=epoch)
                        
                        self.hidden_net.train()
                        self.param_net.train()
                        
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
                    print(f"\n[Early Stopping] Triggered at Epoch {epoch+1}")
                    break
        
        self._save_final_artifacts()
        
        if getattr(self.config, 'RUN_BASELINE', False):
            return self.baseline_net, history
        return self.hidden_net, self.param_net, history    
       
    @torch.no_grad()
    def evaluate(self, loader):
        """Computes loss on the validation/test set without gradient updates."""
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
    
    def _save_checkpoint(self, epoch, loss):
        """Saves the model states for early stopping mechanisms."""
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
        """Saves final networks at the end of training."""
        if getattr(self.config, 'RUN_BASELINE', False):
            torch.save(self.baseline_net.state_dict(), os.path.join(self.results_path, 'baseline_net.pth'))
        else:
            torch.save(self.hidden_net.state_dict(), os.path.join(self.results_path, 'Hnet.pth'))
            torch.save(self.param_net.state_dict(), os.path.join(self.results_path, 'Pnet.pth'))