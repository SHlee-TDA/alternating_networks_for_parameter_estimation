# infer.py
import os
import glob
import traceback
import numpy as np
import torch

from analyzer import get_analyzer_class
from models import SingleNetworkBaseline

# =============================================================================
# 1. Inference Engine
# =============================================================================

class AlternatingInference:
    """
    Handles the iterative inference process for the Alternating Networks framework.
    Uses H_phi (Hidden Network) and P_psi (Parameter Network) to converge on estimates.
    """
    def __init__(self, hidden_net, param_net, normalizer, config):
        self.hidden_net = hidden_net
        self.param_net = param_net
        self.normalizer = normalizer
        self.config = config
        
        if self.hidden_net: 
            self.hidden_net.eval()
        if self.param_net: 
            self.param_net.eval()

    def run_fixed_point_iteration(self, x_norm, p_initial_guess, max_iter=None, tol=1e-6):
        """
        Executes the iterative inference logic (Alternating Minimization):
        1. P_curr = Normalized(P_init_guess)
        2. Loop: 
            Y_guess = H_phi(X, P_curr) 
            P_next  = P_psi(X, Y_guess)
        """
        max_iter = max_iter or self.config.ITERATIONS
        batch_size = x_norm.size(0)
        
        # Initialize parameters
        p_init_norm = self.normalizer.normalize_params(p_initial_guess)
        p_curr = p_init_norm.repeat(batch_size, 1).to(self.config.DEVICE)
        
        with torch.no_grad():
            for _ in range(max_iter):
                # Step A: Predict Hidden States via H_phi
                y_guess = self.hidden_net(x_norm, p_curr)
                
                # Step B: Estimate Parameters via P_psi
                p_next = self.param_net(x_norm, y_guess)
                
                # Check Convergence
                diff = torch.norm(p_next - p_curr, dim=1).max().item()
                p_curr = p_next
                
                if diff < tol:
                    break
                    
        return p_curr

    def get_all_predictions(self, loader, p_initial_guess, baseline_model=None):
        """
        Iterates over a DataLoader to extract Ground Truth, Ours, and Baseline predictions.
        
        Returns:
            Tuple of denormalized (physical scale) numpy arrays: (p_true, p_ours, p_base)
        """
        if baseline_model: 
            baseline_model.eval()

        all_p_true, all_p_ours, all_p_base = [], [], []

        with torch.no_grad():
            for x_batch, _, p_batch in loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                # 1. Ground Truth Parameters
                all_p_true.append(self.normalizer.denormalize_params(p_batch).cpu().numpy())
                
                # 2. Ours (Alternating Framework)
                p_ours_norm = self.run_fixed_point_iteration(x_batch, p_initial_guess)
                all_p_ours.append(self.normalizer.denormalize_params(p_ours_norm).cpu().numpy())
                
                # 3. Baseline Predictions
                if baseline_model:
                    p_base_norm = baseline_model(x_batch)
                    all_p_base.append(self.normalizer.denormalize_params(p_base_norm).cpu().numpy())
                else:
                    all_p_base.append(np.zeros_like(all_p_true[-1]))

        return (
            np.concatenate(all_p_true, axis=0), 
            np.concatenate(all_p_ours, axis=0), 
            np.concatenate(all_p_base, axis=0)
        )


# =============================================================================
# 2. Evaluation Phase Coordinator
# =============================================================================

def run_evaluation_phase(run_config, logger, system, history, 
                         hidden_net, param_net, baseline_net, 
                         test_l, real_test_loader, p_init, normalizer, 
                         sample_x, sample_p, device):
    """
    Orchestrates Phase 3: Inference generation, Baseline loading, and Analyzer execution.
    """
    if getattr(run_config, 'RUN_BASELINE', False):
        print("  -> [Baseline Mode] Training Completed. Awaiting comparative analysis step.")
        metrics = {
            'train_loss': history['train_total_loss'][-1],
            'val_loss': history['val_total_loss'][-1],
            'test_mse': -1, 
            'epoch': len(history['train_total_loss'])
        }
        logger.log_result_to_csv(metrics)
        return

    print("\n" + "="*60)
    print("  -> Starting Phase 3: Inference & Evaluation...")
    print("="*60)
    
    # 1. Load Baseline Model for Comparison
    baseline_model = None
    try:
        print("  -> Loading Baseline Model for comparison...")
        baseline_model = SingleNetworkBaseline(
            flat_x_dim=sample_x.shape[1],
            flat_y_dim=sample_p.shape[1],
            model_config=run_config.MODEL_CONFIG['param_net'],
            use_spectral_norm=None
        ).to(device)
        
        base_dir = os.path.dirname(logger.results_dir)
        baseline_paths = glob.glob(os.path.join(base_dir, '*', 'baseline_net.pth'))
        
        if baseline_paths:
            latest_baseline_path = max(baseline_paths, key=os.path.getmtime)
            print(f"  -> Found baseline weights at: {latest_baseline_path}")
            baseline_model.load_state_dict(torch.load(latest_baseline_path))
        else:
            print("  -> [Warning] baseline_net.pth not found. Skipping baseline comparison.")
            baseline_model = None
            
    except Exception as e:
        print("  -> [Warning] Baseline model loading failed.")
        traceback.print_exc()
        baseline_model = None

    # 2. Instantiate Inference Engine
    inference_engine = AlternatingInference(hidden_net, param_net, normalizer, run_config)
    
    # 3. Extract Predictions for Simulation Test Data
    print("  -> Extracting predictions for Synthetic Test Data...")
    p_true, p_ours, p_base = inference_engine.get_all_predictions(test_l, p_init, baseline_model)

    # 4. Instantiate Analyzer & Execute Visualizations
    print("  -> Generating visual analysis & calculating metrics...")
    AnalyzerClass = get_analyzer_class(run_config.SYSTEM_NAME)
    
    # Inject models (H_phi, P_psi) and context into the Analyzer
    analyzer = AnalyzerClass(
        hidden_net=hidden_net, param_net=param_net, 
        normalizer=normalizer, config=run_config, 
        system=system, history=history
    )
    
    # 4-1. Basic Training & Structural Analysis
    analyzer.plot_loss_curves()
    if hidden_net and param_net:
        analyzer.analyze_spectral_norms()
        analyzer.plot_spectral_norms_by_layer()
        
        # Extract a single sample for Phase Portrait mapping
        x_sample_batch, _, p_sample_batch = next(iter(test_l))
        x_sample = x_sample_batch[0:1].to(device)
        p_target = normalizer.denormalize_params(p_sample_batch)[0].cpu().numpy()
        
        analyzer.plot_phase_portraits(x_sample, p_target)
    
    # 4-2. System-specific Comparative Analysis (Simulation)
    if hasattr(analyzer, 'evaluate_simulation'):
        analyzer.evaluate_simulation(p_true, p_ours, p_base)
    elif hasattr(analyzer, 'run_comparison'):
        analyzer.run_comparison(p_true, p_ours, p_base)

    # 5. Execute Real Data Evaluation (Domain Adaptation Check)
    if real_test_loader is not None and hasattr(analyzer, 'evaluate_real_data'):
        print("\n  -> Extracting predictions for Real Clinical Data...")
        p_true_real, p_ours_real, p_base_real = inference_engine.get_all_predictions(real_test_loader, p_init, baseline_model)
        analyzer.evaluate_real_data(p_true_real, p_ours_real, p_base_real)

    # 6. Final Logging
    test_mse = float(np.mean((p_true - p_ours)**2))
    metrics = {
        'train_loss': history['train_total_loss'][-1],
        'val_loss': history['val_total_loss'][-1] if 'val_total_loss' in history else -1,
        'test_mse': test_mse,
        'epoch': len(history['train_total_loss'])
    }
    
    logger.log_result_to_csv(metrics)
    print(f"\n  -> Experiment Completed Successfully. Final MSE: {test_mse:.6f}")