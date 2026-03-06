import os
import json
import itertools
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
from sklearn.metrics import mean_squared_error, r2_score
from utils import euler_maruyama
from systems.ogtt_simul import OgttSimul, OGTTModel, ODE_PARAMS, SYS_PARAMS

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class Analyzer:
    """
    Encapsulates evaluation metrics and visualization routines.
    """
    def __init__(self, f_theta, g_phi, test_loader, config, system, p_initial_guess, normalizer=None, history=None):
        self.f_theta = f_theta.to(config.DEVICE)
        self.g_phi = g_phi.to(config.DEVICE)
        self.test_loader = test_loader 
        self.config = config
        self.system = system
        self.normalizer = normalizer
        self.p_initial_guess = p_initial_guess
        self.history = history
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)

    def plot_loss_curves(self):
        """
        Visualizes training metrics.
        Generates a SEPARATE plot file for each loss component found in history.
        (e.g., 'total_loss.png', 'loss_f.png', 'loss_recur.png')
        """
        if self.history is None:
            return

        # 1. 기록된 메트릭 이름 추출 (예: 'total_loss', 'loss_f', 'loss_recur')
        # Trainer가 'train_' 접두사를 붙여 저장하므로, 이를 제거하여 기본 이름을 찾습니다.
        metric_names = set()
        for key in self.history.keys():
            if key.startswith('train_'):
                metric_names.add(key.replace('train_', ''))

        print(f"  -> Plotting curves for: {metric_names}")

        # 2. 각 메트릭별로 그래프 그리기 및 개별 저장
        for metric in metric_names:
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            # 데이터가 없으면 건너뜀
            if train_key not in self.history:
                continue

            plt.figure(figsize=(8, 6))
            
            # Train Curve
            plt.plot(self.history[train_key], label='Train', color='cornflowerblue', linewidth=2)
            
            # Val Curve (존재할 경우에만)
            if val_key in self.history:
                plt.plot(self.history[val_key], label='Val', color='sandybrown', linestyle='--', linewidth=2)

            plt.title(f"Convergence: {metric}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (Log Scale)")
            plt.yscale('log') # 수렴 확인을 위해 로그 스케일 권장
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # 파일명: metric 이름 그대로 사용 (예: loss_recur.png)
            save_name = f"{metric}.png"
            plt.savefig(os.path.join(self.results_path, save_name), dpi=150)
            plt.close() # 메모리 해제

        # 3. JSON 로그 저장 (기존 유지)
        with open(os.path.join(self.results_path, 'loss_history.json'), 'w') as f:
            # numpy float 등을 json serializable하게 변환
            serializable_history = {k: [float(v) for v in vals] for k, vals in self.history.items()}
            json.dump(serializable_history, f, indent=4)

    def plot_scatter(self, p_true, p_pred, prefix="sim"):
        """
        Generates parity plots (True vs Predicted) for each system parameter.
        Includes RMSE and Pearson correlation metrics.
        """
        param_names = self.system.param_names
        num_params = len(param_names)
        
        scatter_colors = ['steelblue', 'mediumseagreen', 'indianred']
        
        for i in range(num_params):
            name = param_names[i]
            true_vals = p_true[:, i]
            pred_vals = p_pred[:, i]
            
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            if not np.all(mask):
                true_vals = true_vals[mask]
                pred_vals = pred_vals[mask]
            
            if len(true_vals) < 2:
                continue

            # Common Range
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            
            # Metrics
            mse = mean_squared_error(true_vals, pred_vals)
            pearson_r, _ = pearsonr(true_vals, pred_vals)
            slope, intercept, _, _, _ = linregress(true_vals, pred_vals)
            
            # --- 1. Accuracy Plot (MSE) ---
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.scatter(true_vals, pred_vals, alpha=0.6, s=20, label='Samples', 
                       color=scatter_colors[0], edgecolors='white', linewidth=0.5)
            # Reference Line (Red Dashed)
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal ($y=x$)')
            
            ax1.set_xlabel(f"True {name}")
            ax1.set_ylabel(f"Predicted {name}")
            ax1.set_title(f"{name} Accuracy (MSE={mse:.4f})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            save_name1 = f'{prefix}_scatter_mse_{name}.png'
            plt.savefig(os.path.join(self.results_path, save_name1), dpi=150)
            plt.close(fig1)
            
            # --- 2. Trend Plot (Pearson R) ---
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(true_vals, pred_vals, alpha=0.6, s=20, label='Samples', 
                       color=scatter_colors[1], edgecolors='white', linewidth=0.5)
            
            # Trend Line (Red Dashed)
            x_trend = np.array([min_val, max_val])
            y_trend = slope * x_trend + intercept
            ax2.plot(x_trend, y_trend, 'r--', linewidth=2, label=f'Trend ($r$={pearson_r:.3f})')
            
            ax2.set_xlabel(f"True {name}")
            ax2.set_ylabel(f"Predicted {name}")
            ax2.set_title(f"{name} Correlation (Pearson $r$={pearson_r:.3f})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            save_name2 = f'{prefix}_scatter_trend_{name}.png'
            plt.savefig(os.path.join(self.results_path, save_name2), dpi=150)
            plt.close(fig2)
            
            print(f"  -> Saved {save_name1} and {save_name2}")
        
        np.savez(os.path.join(self.results_path, f'{prefix}_predictions.npz'), 
                 p_true=p_true, p_pred=p_pred)
    
    def _run_fixed_point_iteration(self, x_norm, max_iter=10, tol=1e-4):
        """
        Executes the iterative inference logic defined by the user.
        
        Flow:
            1. P_curr = Normalized(P_init_guess)
            2. Loop:
                Y_guess = f_theta(X, P_curr)
                P_next  = g_phi(X, Y_guess)
                if converged: break
                P_curr = P_next
        
        Args:
            x_norm (Tensor): Normalized observed data (Batch, Dim).
            
        Returns:
            p_final_norm (Tensor): Converged parameter estimates.
        """
        batch_size = x_norm.size(0)
        
        # 1. Prepare Initial Guess (Normalize & Broadcast)
        # self.p_init is physical scale, so we normalize it first.
        p_init_norm = self.normalizer.normalize_params(self.p_initial_guess)
        p_curr = p_init_norm.repeat(batch_size, 1)
        
        # 2. Iteration Loop
        for k in range(max_iter):
            # Step A: Guess Hidden States (P -> Y)
            y_guess = self.f_theta(x_norm, p_curr)
            
            # Step B: Update Parameters (Y -> P)
            p_next = self.g_phi(x_norm, y_guess)
            
            # Check Convergence
            diff = torch.norm(p_next - p_curr, dim=1).max().item()
            p_curr = p_next
            
            if diff < tol:
                break
                
        return p_curr
    
    def evaluate_predictions(self):
        """
        Quantifies parameter estimation accuracy on the synthetic test set.
        
        Returns:
            p_true (np.array): Ground truth parameters (Physical Scale).
            p_pred (np.array): Estimated parameters (Physical Scale).
        """
        self.f_theta.eval()
        self.g_phi.eval()
        all_p_true = [] 
        all_p_pred = []
        
        with torch.no_grad():
            for x_batch, _, p_batch in self.test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                p_pred_norm = self._run_fixed_point_iteration(x_batch, 
                                                              max_iter=self.config.ITERATIONS,
                                                              tol=1e-6)
                
                # Denormalize for metric calculation
                p_pred_phys = self.normalizer.denormalize_params(p_pred_norm)
                p_true_phys = self.normalizer.denormalize_params(p_batch)
                
                all_p_true.append(p_true_phys.cpu().numpy())
                all_p_pred.append(p_pred_phys.cpu().numpy())

        return np.concatenate(all_p_true), np.concatenate(all_p_pred)

    def evaluate_real_data(self, real_test_loader):
        print(f"\n=== Real Data Evaluation ({len(real_test_loader.dataset)} samples) ===")
        self.f_theta.eval()
        self.g_phi.eval()
        
        all_p_true = []
        all_p_pred = []
        
        with torch.no_grad():
            for x_batch, _, p_batch in real_test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                p_pred_norm = self._run_fixed_point_iteration(x_batch, 
                                                              max_iter=self.config.ITERATIONS,
                                                              tol=1e-6)
                
                # Denormalize for metric calculation
                p_pred_phys = self.normalizer.denormalize_params(p_pred_norm)
                p_true_phys = self.normalizer.denormalize_params(p_batch)
                
                all_p_true.append(p_true_phys.cpu().numpy())
                all_p_pred.append(p_pred_phys.cpu().numpy())

        pred_params = np.concatenate(all_p_pred)
        true_params = np.concatenate(all_p_true)
        
        self.plot_scatter(true_params, pred_params, prefix="real_data")
    
    def plot_phase_portraits(self):
        num_params = len(self.system.param_names)
        if num_params < 2: return

        x_sample = None
        p_true_raw = None
        
        for x, _, p in self.test_loader:
            x_sample = x[0:1].to(self.config.DEVICE) # (1, Dim)
            
            p_vec = p[0].to(self.config.DEVICE)
            p_true_raw = self.normalizer.denormalize_params(p_vec).cpu().numpy()
            break
            
        combos = list(itertools.combinations(range(num_params), 2))[:3]
        fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 6), squeeze=False)
        
        for i, (p1, p2) in enumerate(combos):
            self._plot_single_portrait(axes.flatten()[i], x_sample, p_true_raw, (p1, p2))
            
        plt.savefig(os.path.join(self.results_path, 'phase_portraits.png'), dpi=150)
        plt.close(fig)

    def _plot_single_portrait(self, ax, x_observed, p_target, p_dims):
        p1_idx, p2_idx = p_dims
        name1 = self.system.param_names[p1_idx]
        name2 = self.system.param_names[p2_idx]
        
        center1, center2 = p_target[p1_idx], p_target[p2_idx]
        
        range1 = np.linspace(max(0.01, center1 * 0.2), center1 * 1.8, 20)
        range2 = np.linspace(max(0.01, center2 * 0.2), center2 * 1.8, 20)
        grid1, grid2 = np.meshgrid(range1, range2)
        
        p_grid_raw = torch.tensor(p_target, dtype=torch.float32).repeat(grid1.size, 1).to(self.config.DEVICE)
        p_grid_raw[:, p1_idx] = torch.tensor(grid1.flatten(), dtype=torch.float32)
        p_grid_raw[:, p2_idx] = torch.tensor(grid2.flatten(), dtype=torch.float32)
        
        p_grid_norm = self.normalizer.normalize_params(p_grid_raw)
        x_batch = x_observed.repeat(grid1.size, 1) 
        
        with torch.no_grad():
            y_hat = self.f_theta(x_batch, p_grid_norm)
            p_next_norm = self.g_phi(x_batch, y_hat)
            
        p_next_raw = self.normalizer.denormalize_params(p_next_norm).cpu().numpy()
        dp = p_next_raw - p_grid_raw.cpu().numpy()
        
        u = dp[:, p1_idx].reshape(grid1.shape)
        v = dp[:, p2_idx].reshape(grid1.shape)
        speed = np.sqrt(u**2 + v**2)
        
        # Streamplot
        ax.streamplot(grid1, grid2, u, v, color=speed, cmap='autumn_r', linewidth=1, density=1.0, arrowsize=1.0)
        ax.plot(center1, center2, 'bx', markersize=12, markeredgewidth=3, label='Ground Truth', zorder=10)

        # Trajectories (multiple-inits)
        start_points = [
            [range1[2], range2[2]], [range1[2], range2[-3]], 
            [range1[-3], range2[2]], [range1[-3], range2[-3]], 
            [center1 * 0.5, center2 * 1.5] # 임의 지점
        ]
        
        for start in start_points:
            p_start = torch.tensor(p_target, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
            p_start[0, p1_idx] = float(start[0])
            p_start[0, p2_idx] = float(start[1])
            
            traj = [p_start.cpu().numpy()]
            p_curr = self.normalizer.normalize_params(p_start)
            
            with torch.no_grad():
                for _ in range(self.config.ITERATIONS): 
                    y = self.f_theta(x_observed, p_curr)
                    p_curr = self.g_phi(x_observed, y)
                    traj.append(self.normalizer.denormalize_params(p_curr).cpu().numpy())
            
            traj_np = np.concatenate(traj, axis=0)
            
            ax.plot(traj_np[:, p1_idx], traj_np[:, p2_idx], 'k-o', linewidth=1.5, markersize=3, alpha=0.6)
            ax.plot(traj_np[-1, p1_idx], traj_np[-1, p2_idx], 'r*', markersize=10, zorder=11, label='Converged' if start == start_points[0] else "")

        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_title(f"Dynamics: {name1} vs {name2}")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        ax.grid(True, alpha=0.3)


    def _get_model_spectral_norms(self, model):
        """
        Performs a dummy forward pass to trigger Spectral Normalization hooks,
        updating the effective weights, and then measures the spectral norm of each linear layer.
        """
        norms, indices = [], []
        linear_idx = 1
        scale_factor = 0.99
        # 1. Trigger Hook to update effective weights
        try:
            # Identify the correct container
            if hasattr(model, 'network'):
                container = model.network
            elif hasattr(model, 'net'):
                container = model.net.network if hasattr(model.net, 'network') else model.net
            else:
                raise AttributeError("Model has no 'network' or 'net' attribute")

            # Find the first linear layer to determine input dimension
            first_linear = None
            for layer in container:
                if isinstance(layer, torch.nn.Linear):
                    first_linear = layer
                    break
            
            if first_linear is not None:
                in_dim = first_linear.in_features
                dummy_input = torch.randn(1, in_dim, device=self.config.DEVICE)
                with torch.no_grad():
                    container(dummy_input) # Updates layer.weight via hook
                
        except Exception as e:
            print(f"Warning: Dummy forward failed ({e}). Values might be stale.")

        # 2. Measure Spectral Norm of updated weights
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                # Use 'weight' (effective) instead of 'weight_orig'
                weight = layer.weight 
                
                # Compute Spectral Norm (Largest Singular Value)
                norm = torch.linalg.norm(weight, ord=2).item()
                norms.append(scale_factor * norm)
                indices.append(linear_idx)
                linear_idx += 1
                
        return {'indices': indices, 'norms': norms}
    
    def _analyze_spectral_norms_single(self, model, model_name):
        """Computes and logs the spectral norms for all linear layers in a single model."""
        print(f"\n--- Analyzing Spectral Norms for {model_name} ---")
        
        data = self._get_model_spectral_norms(model)
        norms = data['norms']
        
        for i, norm in enumerate(norms):
            print(f"  Layer {i+1}: Effective Spectral Norm = {norm:.4f}")
        
        prod = np.prod(norms)
        print(f"Product of norms for {model_name}: {prod:.4f}")
        return prod

    def analyze_spectral_norms(self):
        """
        Verifies the Contraction Mapping condition by analyzing the product of spectral norms across both networks.
        Condition: Lip(f) * Lip(g) < 1
        """
        prod_f = self._analyze_spectral_norms_single(self.f_theta, "f_theta")
        prod_g = self._analyze_spectral_norms_single(self.g_phi, "g_phi")
        total_prod = prod_f * prod_g
        
        print("\n" + "="*50)
        print(f"Total Product of Spectral Norms: {total_prod:.4f}")
        
        # Check condition with a small numerical tolerance
        if total_prod < 1.0 + 1e-4:
            print("✅ Contraction mapping condition is satisfied.")
        else:
            print(f"⚠️ Contraction mapping condition is NOT satisfied (L={total_prod:.4f}).")
        print("="*50 + "\n")

    def plot_spectral_norms_by_layer(self):
        """Visualizes the spectral norm of each layer as a bar chart and saves the data."""
        print("Plotting and saving spectral norms by layer...")

        f_theta_data = self._get_model_spectral_norms(self.f_theta)
        g_phi_data = self._get_model_spectral_norms(self.g_phi)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        
        def _plot_for_model(ax, norm_data, model_name):
            ax.bar(norm_data['indices'], norm_data['norms'], color='skyblue', edgecolor='black')
            ax.axhline(y=1.0, color='r', linestyle='--', label='Threshold y=1')
            ax.set_xlabel("Linear Layer Index")
            ax.set_ylabel("Spectral Norm")
            ax.set_title(f"Model: {model_name}")
            ax.set_xticks(norm_data['indices'])
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        _plot_for_model(axes[0], f_theta_data, "f_theta (HiddenVarPredictor)")
        _plot_for_model(axes[1], g_phi_data, "g_phi (ParameterEstimator)")

        save_path = os.path.join(self.results_path, 'spectral_norms_plot.png')
        plt.savefig(save_path)
        plt.close(fig)

        # Save raw data for further analysis
        all_norm_data = {
            'f_theta': f_theta_data,
            'g_phi': g_phi_data
        }
        with open(os.path.join(self.results_path, 'spectral_norms_by_layer.json'), 'w') as f:
            json.dump(all_norm_data, f, indent=4)