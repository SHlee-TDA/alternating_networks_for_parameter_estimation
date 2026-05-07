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

class BaseAnalyzer:
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
        
        # BUGFIX: GPU CRASH
        p_grid_raw = torch.tensor(p_target, dtype=torch.float32).repeat(grid1.size, 1).to(self.config.DEVICE)
        p_grid_raw[:, p1_idx] = torch.tensor(grid1.flatten(), dtype=torch.float32).to(self.config.DEVICE)
        p_grid_raw[:, p2_idx] = torch.tensor(grid2.flatten(), dtype=torch.float32).to(self.config.DEVICE)
        
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
            
        def run_comparison(self, baseline_model):
            """자식 클래스에서 오버라이딩하여 각 벤치마크에 맞는 비교 분석을 수행합니다."""
            print(f"  -> [Info] No specific comparison logic defined for {self.__class__.__name__}.")
            
# Benchmark Specific Analyzers (자식 클래스)

class SIRAnalyzer(BaseAnalyzer):
    def run_comparison(self, baseline_model):
        """
        Baseline(Single Net)과 Ours(Alternating Net)의 예측 결과를 비교하고 시각화합니다.
        SIR 시스템에 특화된 평가를 수행합니다.
        """
        print("\n=== Running SIR Comparison Analysis (Baseline vs Ours) ===")

        if baseline_model is None:
            print("  -> [Warning] Baseline model not provided. Skipping SIR comparison.")
            return

        self.f_theta.eval()
        self.g_phi.eval()
        baseline_model.eval()

        all_p_true = []
        all_p_ours = []
        all_p_base = []

        with torch.no_grad():
            for x_batch, _, p_batch in self.test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                # 1. True Parameters (Denormalize)
                p_true_phys = self.normalizer.denormalize_params(p_batch).cpu().numpy()
                all_p_true.append(p_true_phys)
                
                # 2. Baseline Predictions
                p_base_norm = baseline_model(x_batch)
                p_base_phys = self.normalizer.denormalize_params(p_base_norm).cpu().numpy()
                all_p_base.append(p_base_phys)
                
                # 3. Ours (Alternating) Predictions
                p_ours_norm = self._run_fixed_point_iteration(x_batch, 
                                                              max_iter=self.config.ITERATIONS,
                                                              tol=1e-6)
                p_ours_phys = self.normalizer.denormalize_params(p_ours_norm).cpu().numpy()
                all_p_ours.append(p_ours_phys)

        p_true = np.concatenate(all_p_true, axis=0)
        p_ours = np.concatenate(all_p_ours, axis=0)
        p_base = np.concatenate(all_p_base, axis=0)

        # R0 = beta / gamma (파라미터 순서가 beta, gamma라고 가정)
        R0_true = p_true[:, 0] / p_true[:, 1]
        R0_ours = p_ours[:, 0] / p_ours[:, 1]
        R0_base = p_base[:, 0] / p_base[:, 1]

        # 4. 평가 점수 계산 및 JSON 저장
        metrics_dict = self._calculate_and_save_metrics(p_true, p_ours, p_base, R0_true, R0_ours, R0_base)
        
        # 5. 시각화 (Scatter Plot에 점수 포함)
        self._plot_sir_scatter(p_true, p_ours, p_base, R0_true, R0_ours, R0_base, metrics_dict)
        self._plot_trajectory_comparison(p_true, p_ours, p_base, R0_true)

    def _calculate_and_save_metrics(self, p_true, p_ours, p_base, R0_true, R0_ours, R0_base):
        def calc_metrics(true, pred):
            rmse = float(np.sqrt(np.mean((true - pred)**2)))
            mae = float(np.mean(np.abs(true - pred)))
            r = float(pearsonr(true, pred)[0]) if len(true) > 1 else 0.0
            return {"RMSE": rmse, "MAE": mae, "Pearson_r": r}

        metrics = {
            "Beta": {
                "Baseline": calc_metrics(p_true[:, 0], p_base[:, 0]),
                "Ours": calc_metrics(p_true[:, 0], p_ours[:, 0])
            },
            "Gamma": {
                "Baseline": calc_metrics(p_true[:, 1], p_base[:, 1]),
                "Ours": calc_metrics(p_true[:, 1], p_ours[:, 1])
            },
            "R0": {
                "Baseline": calc_metrics(R0_true, R0_base),
                "Ours": calc_metrics(R0_true, R0_ours)
            }
        }

        # 표(Table) 콘솔 출력
        print("\n" + "="*85)
        print(f"{'Metric':<10} | {'Baseline (RMSE / MAE / r)':<33} | {'Ours (RMSE / MAE / r)':<33}")
        print("-" * 85)
        for key, vals in metrics.items():
            b = vals['Baseline']
            o = vals['Ours']
            b_str = f"{b['RMSE']:.4f} / {b['MAE']:.4f} / {b['Pearson_r']:.4f}"
            o_str = f"{o['RMSE']:.4f} / {o['MAE']:.4f} / {o['Pearson_r']:.4f}"
            print(f"{key:<10} | {b_str:<33} | {o_str:<33}")
        print("="*85 + "\n")

        # JSON 파일로 저장
        save_path = os.path.join(self.results_path, 'sir_comparison_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"  -> Saved quantitative metrics to {save_path}")
        
        return metrics

    def _plot_sir_scatter(self, p_true, p_ours, p_base, R0_true, R0_ours, R0_base, metrics_dict):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        titles = [r'Infection Rate ($\beta$)', r'Recovery Rate ($\gamma$)', r'Basic Reproduction Number ($\mathcal{R}_0$)']
        keys = ["Beta", "Gamma", "R0"]
        
        true_vals = [p_true[:, 0], p_true[:, 1], R0_true]
        ours_vals = [p_ours[:, 0], p_ours[:, 1], R0_ours]
        base_vals = [p_base[:, 0], p_base[:, 1], R0_base]

        for i in range(3):
            ax = axes[i]
            
            # Scatter plots
            ax.scatter(true_vals[i], base_vals[i], alpha=0.4, color='indianred', label='Baseline (Single)', marker='x')
            ax.scatter(true_vals[i], ours_vals[i], alpha=0.5, color='steelblue', label='Ours (Alternating)', marker='o', edgecolors='white', linewidth=0.5)
            
            # y=x line
            min_val = min(np.min(true_vals[i]), np.min(ours_vals[i]), np.min(base_vals[i]))
            max_val = max(np.max(true_vals[i]), np.max(ours_vals[i]), np.max(base_vals[i]))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal ($y=x$)')
            
            if i == 2:
                # R0 = 1 Threshold
                ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
                ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7)

            # Metric Text Box 삽입 (좌측 상단)
            b_mets = metrics_dict[keys[i]]['Baseline']
            o_mets = metrics_dict[keys[i]]['Ours']
            
            textstr = '\n'.join((
                r'$\bf{Baseline}$',
                f'RMSE: {b_mets["RMSE"]:.4f}',
                f'r: {b_mets["Pearson_r"]:.4f}',
                '',
                r'$\bf{Ours}$',
                f'RMSE: {o_mets["RMSE"]:.4f}',
                f'r: {o_mets["Pearson_r"]:.4f}'
            ))
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            ax.set_title(titles[i], fontsize=14, pad=15)
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'sir_scatter_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Saved scatter plots to {save_path}")

    def _plot_trajectory_comparison(self, p_true, p_ours, p_base, R0_true):
        # (이전과 동일한 _plot_trajectory_comparison 코드 유지)
        from scipy.integrate import solve_ivp
        
        try:
            idx_epidemic = np.where(R0_true > 1.2)[0][0]
            idx_decay = np.where(R0_true < 0.8)[0][0]
        except IndexError:
            print("  -> [Warning] Could not find both R0>1.2 and R0<0.8 cases. Using random indices.")
            idx_epidemic, idx_decay = 0, len(R0_true)-1

        indices = [idx_epidemic, idx_decay]
        titles = [r'Epidemic Regime ($\mathcal{R}_0 > 1$)', r'Decay Regime ($\mathcal{R}_0 < 1$)']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t_span = (0, 110)
        t_eval = np.linspace(0, 110, 200)
        y0 = [49.0, 1.0, 0.0]

        def sir_ode(t, y, beta, gamma):
            S, I, R = y
            N = 50.0
            return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]

        for i, idx in enumerate(indices):
            sol_true = solve_ivp(sir_ode, t_span, y0, args=tuple(p_true[idx]), t_eval=t_eval)
            sol_base = solve_ivp(sir_ode, t_span, y0, args=tuple(p_base[idx]), t_eval=t_eval)
            sol_ours = solve_ivp(sir_ode, t_span, y0, args=tuple(p_ours[idx]), t_eval=t_eval)

            # Observed State S(t)
            ax_s = axes[i, 0]
            ax_s.plot(sol_true.t, sol_true.y[0], 'k-', lw=3, label='True S(t)')
            ax_s.plot(sol_base.t, sol_base.y[0], 'indianred', linestyle='--', lw=2, label='Baseline Pred')
            ax_s.plot(sol_ours.t, sol_ours.y[0], 'steelblue', linestyle='-.', lw=2, label='Ours Pred')
            ax_s.set_title(f"Observed State: S(t) | {titles[i]}")
            ax_s.set_xlabel("Time")
            ax_s.set_ylabel("Population")
            ax_s.legend()
            ax_s.grid(True, alpha=0.3)

            # Hidden State I(t)
            ax_i = axes[i, 1]
            ax_i.plot(sol_true.t, sol_true.y[1], 'k-', lw=3, label='True I(t) (Hidden)')
            ax_i.plot(sol_base.t, sol_base.y[1], 'indianred', linestyle='--', lw=2, label='Baseline Pred')
            ax_i.plot(sol_ours.t, sol_ours.y[1], 'steelblue', linestyle='-.', lw=2, label='Ours Pred')
            ax_i.set_title(f"Hidden State: I(t) | {titles[i]}")
            ax_i.set_xlabel("Time")
            ax_i.legend()
            ax_i.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'sir_trajectory_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Saved trajectory plots to {save_path}")

class LotkaVolterraAnalyzer(BaseAnalyzer):
    def run_comparison(self, baseline_model):
        """
        Lotka-Volterra 모델에 대한 Baseline vs Ours 비교 분석
        """
        print("\n=== Running Lotka-Volterra Comparison Analysis ===")

        if baseline_model is None:
            print("  -> [Warning] Baseline model not provided. Skipping LV comparison.")
            return

        self.f_theta.eval()
        self.g_phi.eval()
        baseline_model.eval()

        all_p_true = []
        all_p_ours = []
        all_p_base = []
        sample_x_for_plot = None # 시각화를 위한 샘플 데이터 저장

        with torch.no_grad():
            for x_batch, _, p_batch in self.test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                if sample_x_for_plot is None:
                    sample_x_for_plot = x_batch[0].cpu().numpy()
                
                # 1. True
                p_true_phys = self.normalizer.denormalize_params(p_batch).cpu().numpy()
                all_p_true.append(p_true_phys)
                
                # 2. Baseline
                p_base_norm = baseline_model(x_batch)
                p_base_phys = self.normalizer.denormalize_params(p_base_norm).cpu().numpy()
                all_p_base.append(p_base_phys)
                
                # 3. Ours
                p_ours_norm = self._run_fixed_point_iteration(x_batch, 
                                                              max_iter=self.config.ITERATIONS,
                                                              tol=1e-6)
                p_ours_phys = self.normalizer.denormalize_params(p_ours_norm).cpu().numpy()
                all_p_ours.append(p_ours_phys)

        p_true = np.concatenate(all_p_true, axis=0)
        p_ours = np.concatenate(all_p_ours, axis=0)
        p_base = np.concatenate(all_p_base, axis=0)

        # 4. 정량 지표 계산 및 저장
        metrics_dict = self._calculate_and_save_metrics(p_true, p_ours, p_base)
        
        # 5. 시각화 
        self._plot_lv_scatter(p_true, p_ours, p_base, metrics_dict)
        self._plot_lv_trajectories_and_phase(p_true, p_ours, p_base)

    def _calculate_and_save_metrics(self, p_true, p_ours, p_base):
        def calc_metrics(true, pred):
            rmse = float(np.sqrt(np.mean((true - pred)**2)))
            mae = float(np.mean(np.abs(true - pred)))
            r = float(pearsonr(true, pred)[0]) if len(true) > 1 else 0.0
            return {"RMSE": rmse, "MAE": mae, "Pearson_r": r}

        metrics = {}
        for i, p_name in enumerate(self.system.param_names):
            metrics[p_name] = {
                "Baseline": calc_metrics(p_true[:, i], p_base[:, i]),
                "Ours": calc_metrics(p_true[:, i], p_ours[:, i])
            }

        print("\n" + "="*85)
        print(f"{'Metric':<10} | {'Baseline (RMSE / MAE / r)':<33} | {'Ours (RMSE / MAE / r)':<33}")
        print("-" * 85)
        for key, vals in metrics.items():
            b, o = vals['Baseline'], vals['Ours']
            b_str = f"{b['RMSE']:.4f} / {b['MAE']:.4f} / {b['Pearson_r']:.4f}"
            o_str = f"{o['RMSE']:.4f} / {o['MAE']:.4f} / {o['Pearson_r']:.4f}"
            print(f"{key:<10} | {b_str:<33} | {o_str:<33}")
        print("="*85 + "\n")

        save_path = os.path.join(self.results_path, 'lv_comparison_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics

    def _plot_lv_scatter(self, p_true, p_ours, p_base, metrics_dict):
        num_params = len(self.system.param_names)
        fig, axes = plt.subplots(1, num_params, figsize=(5*num_params, 5.5))
        if num_params == 1: axes = [axes]

        for i, p_name in enumerate(self.system.param_names):
            ax = axes[i]
            true_v, ours_v, base_v = p_true[:, i], p_ours[:, i], p_base[:, i]
            
            ax.scatter(true_v, base_v, alpha=0.4, color='indianred', label='Baseline', marker='x')
            ax.scatter(true_v, ours_v, alpha=0.5, color='steelblue', label='Ours', marker='o', edgecolors='white', lw=0.5)
            
            min_v, max_v = min(np.min(true_v), np.min(base_v)), max(np.max(true_v), np.max(base_v))
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Ideal')
            
            b_mets, o_mets = metrics_dict[p_name]['Baseline'], metrics_dict[p_name]['Ours']
            textstr = '\n'.join((
                r'$\bf{Baseline}$', f'RMSE: {b_mets["RMSE"]:.4f}', f'r: {b_mets["Pearson_r"]:.4f}', '',
                r'$\bf{Ours}$', f'RMSE: {o_mets["RMSE"]:.4f}', f'r: {o_mets["Pearson_r"]:.4f}'
            ))
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

            ax.set_title(f"Parameter: {p_name}", fontsize=14, pad=15)
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'lv_scatter_comparison.png'), dpi=300)
        plt.close()

    def _plot_lv_trajectories_and_phase(self, p_true, p_ours, p_base):
        from scipy.integrate import solve_ivp
        
        # 진폭이 큰 극단적 케이스 하나를 랜덤(또는 특정 룰)으로 추출
        idx = np.random.randint(0, len(p_true))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        t_span = (0, 50) # LV 시스템의 시간에 맞게 조절 필요 (예: 0~50)
        t_eval = np.linspace(0, 50, 500)
        
        # 시스템 초기값 가져오기
        y0 = self.system.initial_conditions
        y0 = [y[0] if isinstance(y, list) else y for y in y0]

        # 동적으로 파라미터 언패킹 (alpha, beta, delta, gamma 가정)
        def lv_ode(t, y, *params):
            x, y_predator = y
            alpha, beta, delta, gamma = params[0], params[1], params[2], params[3]
            dxdt = alpha * x - beta * x * y_predator
            dydt = delta * x * y_predator - gamma * y_predator
            return [dxdt, dydt]

        sol_true = solve_ivp(lv_ode, t_span, y0, args=tuple(p_true[idx]), t_eval=t_eval)
        sol_base = solve_ivp(lv_ode, t_span, y0, args=tuple(p_base[idx]), t_eval=t_eval)
        sol_ours = solve_ivp(lv_ode, t_span, y0, args=tuple(p_ours[idx]), t_eval=t_eval)

        # 1. Prey (x) Trajectory
        ax = axes[0]
        ax.plot(sol_true.t, sol_true.y[0], 'k-', lw=2, label='True Prey')
        ax.plot(sol_base.t, sol_base.y[0], 'indianred', linestyle='--', lw=2, label='Baseline')
        ax.plot(sol_ours.t, sol_ours.y[0], 'steelblue', linestyle='-.', lw=2, label='Ours')
        ax.set_title("Prey Dynamics (Observed)")
        ax.set_xlabel("Time"); ax.legend(); ax.grid(True, alpha=0.3)

        # 2. Predator (y) Trajectory
        ax = axes[1]
        ax.plot(sol_true.t, sol_true.y[1], 'k-', lw=2, label='True Predator')
        ax.plot(sol_base.t, sol_base.y[1], 'indianred', linestyle='--', lw=2, label='Baseline')
        ax.plot(sol_ours.t, sol_ours.y[1], 'steelblue', linestyle='-.', lw=2, label='Ours')
        ax.set_title("Predator Dynamics (Hidden)")
        ax.set_xlabel("Time"); ax.legend(); ax.grid(True, alpha=0.3)

        # 3. Phase Portrait (Prey vs Predator)
        ax = axes[2]
        ax.plot(sol_true.y[0], sol_true.y[1], 'k-', lw=2, label='True Limit Cycle')
        ax.plot(sol_base.y[0], sol_base.y[1], 'indianred', linestyle='--', lw=2, label='Baseline')
        ax.plot(sol_ours.y[0], sol_ours.y[1], 'steelblue', linestyle='-.', lw=2, label='Ours')
        ax.set_title("Phase Portrait")
        ax.set_xlabel("Prey (x)"); ax.set_ylabel("Predator (y)")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'lv_dynamics_comparison.png'), dpi=300)
        plt.close()

class OgttSimulAnalyzer(BaseAnalyzer):
    def run_comparison(self, baseline_model):
        print("\n=== Running OGTT Simulation Comparison Analysis ===")
        if baseline_model is None:
            print("  -> [Warning] Baseline model not provided.")
            return

        p_true, p_ours, p_base = self._get_predictions(self.test_loader, baseline_model)

        metrics_dict = self._calculate_and_save_metrics(p_true, p_ours, p_base, filename="ogtt_sim_metrics.json")
        self._plot_ogtt_scatter(p_true, p_ours, p_base, metrics_dict, prefix="sim")
        self._plot_symmetric_collapse(p_true, p_ours, p_base, prefix="sim")
        self._plot_ogtt_trajectories(p_true, p_ours, p_base)

    def evaluate_real_data(self, real_test_loader, baseline_model=None):
        """
        Real Dataset(Sumner)에 대해 Baseline과 Ours를 동시 평가합니다.
        """
        print(f"\n=== Running OGTT REAL DATA Comparison ({len(real_test_loader.dataset)} samples) ===")
        
        p_true, p_ours, p_base = self._get_predictions(real_test_loader, baseline_model)

        metrics_dict = self._calculate_and_save_metrics(p_true, p_ours, p_base, filename="ogtt_real_metrics.json")
        self._plot_ogtt_scatter(p_true, p_ours, p_base, metrics_dict, prefix="real")
        self._plot_symmetric_collapse(p_true, p_ours, p_base, prefix="real")

    def _get_predictions(self, loader, baseline_model):
        """Loader에서 데이터를 뽑아 P_true, P_ours, P_base를 반환합니다."""
        self.f_theta.eval()
        self.g_phi.eval()
        if baseline_model: baseline_model.eval()

        all_p_true, all_p_ours, all_p_base = [], [], []

        with torch.no_grad():
            for x_batch, _, p_batch in loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                # 1. True
                all_p_true.append(self.normalizer.denormalize_params(p_batch).cpu().numpy())
                
                # 2. Ours
                p_ours_norm = self._run_fixed_point_iteration(x_batch, max_iter=self.config.ITERATIONS, tol=1e-6)
                all_p_ours.append(self.normalizer.denormalize_params(p_ours_norm).cpu().numpy())
                
                # 3. Baseline
                if baseline_model:
                    p_base_norm = baseline_model(x_batch)
                    all_p_base.append(self.normalizer.denormalize_params(p_base_norm).cpu().numpy())
                else:
                    all_p_base.append(np.zeros_like(all_p_true[-1]))

        return np.concatenate(all_p_true, axis=0), np.concatenate(all_p_ours, axis=0), np.concatenate(all_p_base, axis=0)

    def _calculate_and_save_metrics(self, p_true, p_ours, p_base, filename):
        def calc_metrics(true, pred):
            rmse = float(np.sqrt(np.mean((true - pred)**2)))
            mae = float(np.mean(np.abs(true - pred)))
            r = float(pearsonr(true, pred)[0]) if len(true) > 1 else 0.0
            return {"RMSE": rmse, "MAE": mae, "Pearson_r": r}

        metrics = {}
        for i, p_name in enumerate(self.system.param_names):
            metrics[p_name] = {
                "Baseline": calc_metrics(p_true[:, i], p_base[:, i]),
                "Ours": calc_metrics(p_true[:, i], p_ours[:, i])
            }

        print("\n" + "="*85)
        print(f"{'OGTT Metric':<12} | {'Baseline (RMSE / MAE / r)':<33} | {'Ours (RMSE / MAE / r)':<33}")
        print("-" * 85)
        for key, vals in metrics.items():
            b, o = vals['Baseline'], vals['Ours']
            print(f"{key:<12} | {b['RMSE']:.4f} / {b['MAE']:.4f} / {b['Pearson_r']:.4f} | {o['RMSE']:.4f} / {o['MAE']:.4f} / {o['Pearson_r']:.4f}")
        print("="*85 + "\n")

        with open(os.path.join(self.results_path, filename), 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics

    def _plot_symmetric_collapse(self, p_true, p_ours, p_base, prefix="sim"):
        """보내주신 코드를 바탕으로 Baseline과 Ours를 함께 비교하는 핵심 플롯"""
        print(f"[Info] Generating Symmetric Collapse Visualization ({prefix})")
        
        idx_si, idx_sigma = 0, 1 # S_I와 Sigma의 인덱스
        
        si_true, sigma_true = p_true[:, idx_si], p_true[:, idx_sigma]
        si_ours, sigma_ours = p_ours[:, idx_si], p_ours[:, idx_sigma]
        si_base, sigma_base = p_base[:, idx_si], p_base[:, idx_sigma]
        
        k_true = si_true * sigma_true
        k_ours = si_ours * sigma_ours
        k_base = si_base * sigma_base
        
        sns.set_style("ticks")
        fig = plt.figure(figsize=(20, 6))
        
        # --- Panel 1: Log-Log Joint Distribution ---
        ax1 = fig.add_subplot(131)
        ax1.scatter(si_true, sigma_true, c='grey', alpha=0.15, s=20, label='Ground Truth')
        # Baseline은 붉은색 X로 (대각선으로 붕괴됨을 보여줌)
        ax1.scatter(si_base, sigma_base, c='crimson', alpha=0.5, s=15, marker='x', label='Baseline')
        # Ours는 푸른색 O로 (분포를 따라감을 보여줌)
        ax1.scatter(si_ours, sigma_ours, c='steelblue', alpha=0.6, s=15, edgecolors='white', lw=0.5, label='Ours')
        
        ax1.set_xscale('log'); ax1.set_yscale('log')
        vmin, vmax = min(si_true.min(), sigma_true.min()), max(si_true.max(), sigma_true.max())
        ax1.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='y=x (Symmetric Line)')
        
        ax1.set_xlabel("$S_I$ [Log]", fontsize=12); ax1.set_ylabel("$\sigma$ [Log]", fontsize=12)
        ax1.set_title("1. Log-Log Joint Distribution", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.2)
        
        # --- Panel 2: Product Distribution ---
        ax2 = fig.add_subplot(132)
        sns.kdeplot(np.log10(k_true), ax=ax2, color='grey', fill=True, alpha=0.3, linewidth=2, label='Log($K_{true}$)')
        sns.kdeplot(np.log10(k_base), ax=ax2, color='crimson', linestyle='--', linewidth=2, label='Log($K_{base}$)')
        sns.kdeplot(np.log10(k_ours), ax=ax2, color='steelblue', linewidth=2, label='Log($K_{ours}$)')
        
        ax2.set_xlabel("Log Product ($\log_{10} K$)", fontsize=12); ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("2. Product Consistency (Stiff Direction)", fontsize=14, fontweight='bold')
        ax2.legend(); ax2.grid(True, alpha=0.2)
        
        # --- Panel 3: S_I Correlation (Sloppy Direction Failure) ---
        # 개별 파라미터(S_I)에 대해 Baseline이 얼마나 망가졌고 Ours가 얼마나 맞추는지 보여줍니다.
        ax3 = fig.add_subplot(133)
        ax3.scatter(np.log10(si_true), np.log10(si_base), c='crimson', alpha=0.4, s=15, marker='x', label='Baseline')
        ax3.scatter(np.log10(si_true), np.log10(si_ours), c='steelblue', alpha=0.5, s=15, edgecolors='white', lw=0.5, label='Ours')
        
        min_log, max_log = np.log10(si_true).min(), np.log10(si_true).max()
        ax3.plot([min_log, max_log], [min_log, max_log], 'k--', lw=2, label='Ideal (y=x)')
        
        ax3.set_xlabel("Log True $S_I$", fontsize=12); ax3.set_ylabel("Log Pred $S_I$", fontsize=12)
        ax3.set_title("3. $S_I$ Prediction (Sloppy Direction)", fontsize=14, fontweight='bold')
        ax3.legend(); ax3.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{prefix}_symmetric_collapse.png'), dpi=300)
        plt.close()

    def _plot_ogtt_scatter(self, p_true, p_ours, p_base, metrics_dict, prefix="sim"):
        # 기존과 동일한 개별 파라미터 산점도
        num_params = len(self.system.param_names)
        fig, axes = plt.subplots(1, num_params, figsize=(5*num_params, 5.5))
        if num_params == 1: axes = [axes]

        for i, p_name in enumerate(self.system.param_names):
            ax = axes[i]
            true_v, ours_v, base_v = p_true[:, i], p_ours[:, i], p_base[:, i]
            
            ax.scatter(true_v, base_v, alpha=0.3, color='indianred', label='Baseline', marker='x')
            ax.scatter(true_v, ours_v, alpha=0.5, color='steelblue', label='Ours', marker='o', edgecolors='white', lw=0.5)
            
            min_v, max_v = min(np.min(true_v), np.min(base_v)), max(np.max(true_v), np.max(base_v))
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Ideal')
            
            b_mets, o_mets = metrics_dict[p_name]['Baseline'], metrics_dict[p_name]['Ours']
            textstr = '\n'.join((
                r'$\bf{Baseline}$', f'RMSE: {b_mets["RMSE"]:.4f}', f'r: {b_mets["Pearson_r"]:.4f}', '',
                r'$\bf{Ours}$', f'RMSE: {o_mets["RMSE"]:.4f}', f'r: {o_mets["Pearson_r"]:.4f}'
            ))
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

            ax.set_title(f"{p_name}", fontsize=14, pad=15)
            ax.set_xlabel('True Value'); ax.set_ylabel('Predicted Value')
            ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{prefix}_scatter_comparison.png'), dpi=300)
        plt.close()

    def _plot_ogtt_trajectories(self, p_true, p_ours, p_base):
        from scipy.integrate import solve_ivp
        idx = np.argmax(np.abs(p_true[:, 0] - p_base[:, 0])) # 오차가 가장 큰 샘플
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        t_span = (0, 120); t_eval = np.linspace(0, 120, 200)
        y0 = self.system.initial_conditions
        y0 = [y[0] if isinstance(y, list) else y for y in y0]

        def ogtt_ode(t, y, *params): return self.system.ode_func(t, y, params)

        sol_true = solve_ivp(ogtt_ode, t_span, y0, args=tuple(p_true[idx]), t_eval=t_eval)
        sol_base = solve_ivp(ogtt_ode, t_span, y0, args=tuple(p_base[idx]), t_eval=t_eval)
        sol_ours = solve_ivp(ogtt_ode, t_span, y0, args=tuple(p_ours[idx]), t_eval=t_eval)

        ax = axes[0]
        ax.plot(sol_true.t, sol_true.y[0], 'k-', lw=2, label='True Glucose')
        ax.plot(sol_base.t, sol_base.y[0], 'indianred', linestyle='--', lw=2, label='Baseline')
        ax.plot(sol_ours.t, sol_ours.y[0], 'steelblue', linestyle='-.', lw=2, label='Ours')
        ax.set_title("Glucose Dynamics (Observed)")
        ax.set_xlabel("Time (min)"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(sol_true.t, sol_true.y[1], 'k-', lw=2, label='True Insulin (Hidden)')
        ax.plot(sol_base.t, sol_base.y[1], 'indianred', linestyle='--', lw=2, label='Baseline (Failed)')
        ax.plot(sol_ours.t, sol_ours.y[1], 'steelblue', linestyle='-.', lw=2, label='Ours')
        ax.set_title("Insulin Dynamics (Hidden State Failure)")
        ax.set_xlabel("Time (min)"); ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'sim_trajectory_comparison.png'), dpi=300)
        plt.close()


# ==========================================
# 3. Factory Function (main.py에서 호출할 함수)
# ==========================================
def get_analyzer_class(system_name):
    """시스템 이름에 따라 알맞은 Analyzer 클래스를 반환합니다."""
    if system_name == 'sir':
        return SIRAnalyzer
    elif system_name == 'lotka_volterra':
        return LotkaVolterraAnalyzer
    elif system_name == 'ogtt_simul':
        return OgttSimulAnalyzer
    else:
        print(f"  -> [Warning] Unknown system '{system_name}'. Falling back to BaseAnalyzer.")
        return BaseAnalyzer