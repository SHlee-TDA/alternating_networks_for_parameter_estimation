# infer.py
import numpy as np
import torch
from src.analyzer import get_analyzer_class

# =============================================================================
# 1. Inference Engine
# =============================================================================

class InferenceEngine:
    """
    Handles the inference process for the currently trained model.
    Works seamlessly for both Alternating Networks and Single Network Baselines.
    """
    def __init__(self, normalizer, config, hidden_net=None, param_net=None):
        self.normalizer = normalizer
        self.config = config
        self.hidden_net = hidden_net
        # if baseline mode, param_net is actually the single network baseline
        self.param_net = param_net 
        
        if self.hidden_net: self.hidden_net.eval()
        if self.param_net: self.param_net.eval()

    def run_fixed_point_iteration(self, x_norm, p_initial_guess, max_iter=None, tol=1e-6):
        """Alternating Minimization (Only used when not running baseline)"""
        max_iter = max_iter or self.config.ITERATIONS
        batch_size = x_norm.size(0)
        
        p_init_norm = self.normalizer.normalize_params(p_initial_guess)
        p_curr = p_init_norm.repeat(batch_size, 1).to(self.config.DEVICE)
        
        with torch.no_grad():
            for _ in range(max_iter):
                y_guess = self.hidden_net(x_norm, p_curr)
                p_next = self.param_net(x_norm, y_guess)
                
                if torch.norm(p_next - p_curr, dim=1).max().item() < tol:
                    p_curr = p_next
                    break
                p_curr = p_next
                    
        return p_curr

    def get_predictions(self, loader, p_initial_guess=None):
        """
        Extracts Ground Truth and Predictions for the current active model.
        Returns: (p_true, p_pred) as physical-scale numpy arrays.
        """
        all_p_true, all_p_pred = [], []

        with torch.no_grad():
            for x_batch, _, p_batch in loader:
                x_batch = x_batch.to(self.config.DEVICE)
                
                # 1. Ground Truth
                all_p_true.append(self.normalizer.denormalize_params(p_batch).cpu().numpy())
                
                # 2. Prediction
                if getattr(self.config, 'RUN_BASELINE', False):
                    p_pred_norm = self.param_net(x_batch)
                else:
                    p_pred_norm = self.run_fixed_point_iteration(x_batch, p_initial_guess)
                    
                all_p_pred.append(self.normalizer.denormalize_params(p_pred_norm).cpu().numpy())

        return np.concatenate(all_p_true, axis=0), np.concatenate(all_p_pred, axis=0)


# =============================================================================
# 2. Evaluation Phase Coordinator
# =============================================================================

def run_evaluation_phase(run_config, logger, system, history, 
                         hidden_net, param_net,
                         test_l, real_test_loader, p_init, normalizer, 
                         device):
    """
    Orchestrates Phase 3: Evaluates the performance of the currently trained model.
    """
    print("\n" + "="*60)
    mode_name = "Baseline (Single Net)" if getattr(run_config, 'RUN_BASELINE', False) else "Alternating Net"
    print(f"  -> Starting Phase 3: Inference & Evaluation [{mode_name}]...")
    print("="*60)

    # 1. Instantiate Inference Engine
    engine = InferenceEngine(normalizer, run_config, hidden_net, param_net)
    
    # 2. Extract Predictions for Synthetic Data
    print("  -> Extracting predictions for Synthetic Test Data...")
    p_true, p_pred = engine.get_predictions(test_l, p_init)

    # 3. Instantiate Analyzer & Execute Visualizations
    print("  -> Generating visual analysis & calculating metrics...")
    AnalyzerClass = get_analyzer_class(run_config.SYSTEM_NAME)
    
    analyzer = AnalyzerClass(
        hidden_net=hidden_net, param_net=param_net, 
        normalizer=normalizer, config=run_config, 
        system=system, history=history
    )
    
    # 3-1. Basic Training Logs
    analyzer.plot_loss_curves()
    
    # 3-2. Structural Analysis (Only for Alternating Networks)
    if not getattr(run_config, 'RUN_BASELINE', False) and hidden_net and param_net:
        analyzer.analyze_spectral_norms()
        analyzer.plot_spectral_norms_by_layer()
        
        x_sample_batch, _, p_sample_batch = next(iter(test_l))
        x_sample = x_sample_batch[0:1].to(device)
        p_target = normalizer.denormalize_params(p_sample_batch)[0].cpu().numpy()
        analyzer.plot_phase_portraits(x_sample, p_target)
    
    # 3-3. Evaluate Single Model Accuracy
    p_dummy = np.zeros_like(p_true)
    if hasattr(analyzer, 'evaluate_simulation'):
        if getattr(run_config, 'RUN_BASELINE', False):
            analyzer.evaluate_simulation(p_true, p_ours=p_dummy, p_base=p_pred)
        else:
            analyzer.evaluate_simulation(p_true, p_ours=p_pred, p_base=p_dummy)

    # 4. Execute Real Data Evaluation
    if real_test_loader is not None and hasattr(analyzer, 'evaluate_real_data'):
        print("\n  -> Extracting predictions for Real Clinical Data...")
        p_true_real, p_pred_real = engine.get_predictions(real_test_loader, p_init)
        
        # =====================================================================
        param_dim = len(system.param_names) # 항상 2 (si, sigma)
        
        p_true_real = p_true_real.reshape(-1, param_dim)
        p_pred_real = p_pred_real.reshape(-1, param_dim)
        p_dummy_real = np.zeros_like(p_pred_real)
        # =====================================================================

        if getattr(run_config, 'RUN_BASELINE', False):
            analyzer.evaluate_real_data(p_true_real, p_ours=p_dummy_real, p_base=p_pred_real)
        else:
            analyzer.evaluate_real_data(p_true_real, p_ours=p_pred_real, p_base=p_dummy_real)

    # 5. Final Logging
    test_mse = float(np.mean((p_true - p_pred)**2))
    metrics = {
        'train_loss': history['train_total_loss'][-1] if 'train_total_loss' in history else -1,
        'val_loss': history['val_total_loss'][-1] if 'val_total_loss' in history else -1,
        'test_mse': test_mse,
        'epoch': len(history.get('train_total_loss', []))
    }
    
    logger.log_result_to_csv(metrics)
    print(f"\n  -> Experiment Completed Successfully. Final MSE: {test_mse:.6f}")
    
