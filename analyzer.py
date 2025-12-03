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
from systems.ogtt_simul import OgttSimul, OGTTModel, ode_params, sys_params

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class Analyzer:
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
        if self.history is None: return
        print("Plotting loss curves...")
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
        
        # 부드러운 색상 사용
        c_train = 'cornflowerblue'
        c_val = 'sandybrown'

        # Plot f_theta
        axs[0].plot(self.history['train_loss_f'], label='Train', color=c_train, linewidth=2)
        axs[0].plot(self.history['val_loss_f'], label='Val', color=c_val, linestyle='--', linewidth=2)
        axs[0].set_title("Loss: f_theta (Hidden Predictor)")
        axs[0].set_yscale('log') 
        axs[0].legend(); axs[0].grid(True, which="both", ls="-", alpha=0.3)

        # Plot g_phi
        axs[1].plot(self.history['train_loss_g'], label='Train', color=c_train, linewidth=2)
        axs[1].plot(self.history['val_loss_g'], label='Val', color=c_val, linestyle='--', linewidth=2)
        axs[1].set_title("Loss: g_phi (Parameter Estimator)")
        axs[1].set_yscale('log')
        axs[1].legend(); axs[1].grid(True, which="both", ls="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'loss_curves.png'), dpi=150)
        plt.close(fig)

        with open(os.path.join(self.results_path, 'loss_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

    def evaluate_predictions(self):
        """Sim Test Set 평가"""
        print("Evaluating predictions on Sim Test Set...")
        self.f_theta.eval(); self.g_phi.eval()
        all_p_true, all_p_pred = [], []
        
        with torch.no_grad():
            for x_batch, _, p_batch in self.test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                p_batch = p_batch.to(self.config.DEVICE)
                
                # 초기값 정규화
                p_curr_norm = self.normalizer.normalize_params(
                    self.p_initial_guess.repeat(x_batch.size(0), 1)
                )
                
                for _ in range(self.config.ITERATIONS):
                    y_hat_norm = self.f_theta(x_batch, p_curr_norm)
                    p_curr_norm = self.g_phi(x_batch, y_hat_norm)
                
                # 역정규화
                p_pred = self.normalizer.denormalize_params(p_curr_norm)
                p_true = self.normalizer.denormalize_params(p_batch)
                
                all_p_pred.append(p_pred.cpu().numpy())
                all_p_true.append(p_true.cpu().numpy())
                
        return np.concatenate(all_p_true), np.concatenate(all_p_pred)

    def plot_scatter(self, p_true, p_pred, prefix="sim"):
        """MSE용 그림과 Trend(Pearson)용 그림 분리 저장 + NaN 필터링"""
        print(f"Plotting scatter ({prefix})...")
        param_names = self.system.param_names
        num_params = len(param_names)
        
        # 부드러운 색상 (파랑, 초록 계열)
        scatter_colors = ['steelblue', 'mediumseagreen', 'indianred']
        
        for i in range(num_params):
            name = param_names[i]
            true_vals = p_true[:, i]
            pred_vals = p_pred[:, i]
            
            # [안전장치] NaN / Inf 제거
            mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
            if not np.all(mask):
                print(f"  ⚠️ Warning: Dropping {len(true_vals) - np.sum(mask)} NaN/Inf samples in {name}")
                true_vals = true_vals[mask]
                pred_vals = pred_vals[mask]
            
            if len(true_vals) < 2:
                print(f"  ❌ Not enough valid samples for {name}. Skipping.")
                continue

            # Common Range
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            
            # Metrics
            mse = mean_squared_error(true_vals, pred_vals)
            pearson_r, _ = pearsonr(true_vals, pred_vals)
            slope, intercept, _, _, _ = linregress(true_vals, pred_vals)
            
            # --- 1. Accuracy Plot (MSE Focus) ---
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
            
            # --- 2. Trend Plot (Pearson R Focus) ---
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(true_vals, pred_vals, alpha=0.6, s=20, label='Samples', 
                       color=scatter_colors[1], edgecolors='white', linewidth=0.5)
            
            # Trend Line (Red Dashed - 요청사항)
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

    def plot_phase_portraits(self):
        print("Plotting phase portraits...")
        num_params = len(self.system.param_names)
        if num_params < 2: return

        # 배치 전체 평균 입력값 사용
        x_mean_batch = None
        p_mean = None
        
        for x, _, p in self.test_loader:
            x_mean_batch = x.mean(dim=0, keepdim=True).to(self.config.DEVICE)
            p_tmp = p.mean(dim=0).cpu().numpy() 
            p_mean = self.normalizer.denormalize_params(torch.tensor(p_tmp).to(self.config.DEVICE)).cpu().numpy()
            break
            
        combos = list(itertools.combinations(range(num_params), 2))[:3]
        fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 6), squeeze=False)
        
        for i, (p1, p2) in enumerate(combos):
            self._plot_single_portrait(axes.flatten()[i], x_mean_batch, p_mean, (p1, p2))
            
        plt.savefig(os.path.join(self.results_path, 'phase_portraits.png'), dpi=150)
        plt.close(fig)

    def _plot_single_portrait(self, ax, x_observed, p_mean, p_dims):
        p1_idx, p2_idx = p_dims
        name1 = self.system.param_names[p1_idx]
        name2 = self.system.param_names[p2_idx]
        
        center1, center2 = p_mean[p1_idx], p_mean[p2_idx]
        # 범위 설정
        range1 = np.linspace(max(0, center1 * 0.2), center1 * 1.8, 15)
        range2 = np.linspace(max(0, center2 * 0.2), center2 * 1.8, 15)
        grid1, grid2 = np.meshgrid(range1, range2)
        
        # Grid Tensor (Numpy -> Tensor)
        p_grid_raw = torch.tensor(p_mean, dtype=torch.float32).repeat(grid1.size, 1).to(self.config.DEVICE)
        p_grid_raw[:, p1_idx] = torch.tensor(grid1.flatten(), dtype=torch.float32)
        p_grid_raw[:, p2_idx] = torch.tensor(grid2.flatten(), dtype=torch.float32)
        
        p_grid_norm = self.normalizer.normalize_params(p_grid_raw)
        x_batch = x_observed.repeat(grid1.size, 1) 
        
        # Vector Field
        with torch.no_grad():
            y_hat = self.f_theta(x_batch, p_grid_norm)
            p_next_norm = self.g_phi(x_batch, y_hat)
            
        p_next_raw = self.normalizer.denormalize_params(p_next_norm).cpu().numpy()
        dp = p_next_raw - p_grid_raw.cpu().numpy()
        
        u = dp[:, p1_idx].reshape(grid1.shape)
        v = dp[:, p2_idx].reshape(grid1.shape)
        speed = np.sqrt(u**2 + v**2)
        
        # Streamplot (부드러운 색상)
        ax.streamplot(grid1, grid2, u, v, color=speed, cmap='autumn_r', linewidth=1, density=1.2, arrowsize=1.0)
        
        # Trajectories (다중 시작점)
        start_points = [
            [range1[2], range2[2]], [range1[2], range2[-3]], 
            [range1[-3], range2[2]], [range1[-3], range2[-3]], 
            [center1, center2]
        ]
        
        for start in start_points:
            # [수정] Numpy float -> Python float 변환 (TypeError 방지)
            p_start = torch.tensor(p_mean, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
            p_start[0, p1_idx] = float(start[0])
            p_start[0, p2_idx] = float(start[1])
            
            traj = [p_start.cpu().numpy()]
            p_curr = self.normalizer.normalize_params(p_start)
            
            with torch.no_grad():
                for _ in range(10):
                    y = self.f_theta(x_observed, p_curr)
                    p_curr = self.g_phi(x_observed, y)
                    traj.append(self.normalizer.denormalize_params(p_curr).cpu().numpy())
            
            traj_np = np.concatenate(traj, axis=0)
            # 궤적 (검은 실선 + 점)
            ax.plot(traj_np[:, p1_idx], traj_np[:, p2_idx], 'k-o', linewidth=1.5, markersize=3, alpha=0.6)
            ax.plot(traj_np[-1, p1_idx], traj_np[-1, p2_idx], 'r*', markersize=12, zorder=5)

        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_title(f"Dynamics: {name1} vs {name2}")
        ax.grid(True, alpha=0.3)

    def evaluate_real_data(self, real_test_loader, num_vis=5):
        print(f"\n=== Real Data Evaluation ({len(real_test_loader.dataset)} samples) ===")
        self.f_theta.eval(); self.g_phi.eval()
        
        all_p_pred = []
        all_x_denorm = [] 
        all_y_denorm = []
        all_p_ref = [] 

        with torch.no_grad():
            for x_batch, y_batch, p_batch in real_test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                y_denorm = self.normalizer.denormalize_inputs(y_batch, variable_type='hidden')

                p_curr_norm = self.normalizer.normalize_params(
                    self.p_initial_guess.repeat(x_batch.size(0), 1)
                )
                
                for _ in range(self.config.ITERATIONS):
                    y_hat = self.f_theta(x_batch, p_curr_norm)
                    p_curr_norm = self.g_phi(x_batch, y_hat)
                
                p_pred = self.normalizer.denormalize_params(p_curr_norm)
                x_denorm = self.normalizer.denormalize_inputs(x_batch, variable_type='observed')
                
                all_p_pred.append(p_pred.cpu().numpy())
                all_x_denorm.append(x_denorm.cpu().numpy())
                all_y_denorm.append(y_denorm.cpu().numpy())
                
                p_ref = self.normalizer.denormalize_params(p_batch.to(self.config.DEVICE))
                all_p_ref.append(p_ref.cpu().numpy())

        pred_params = np.concatenate(all_p_pred)
        x_raw = np.concatenate(all_x_denorm)
        p_ref_total = np.concatenate(all_p_ref)
        
        # [수정] 조건 완화: 하나라도 유효한 값이 있으면 그림
        valid_mask = ~np.isnan(p_ref_total).any(axis=1)
        if np.sum(valid_mask) > 0:
            print(f"  -> Plotting Real Data Scatter ({np.sum(valid_mask)} valid samples)...")
            self.plot_scatter(p_ref_total, pred_params, prefix="real_data")
        else:
            print("  -> Skipping Real Data Scatter (All ground truth values are NaN).")
        
        #if getattr(self.config, 'USE_SDE', False):
        #    self._evaluate_uncertainty(x_raw, pred_params, num_vis, y_raw=np.concatenate(all_y_denorm))
            
        print("Real data evaluation complete.")

    def _evaluate_uncertainty(self, x_raw, p_pred, num_vis, y_raw=None):
        print(f"  -> Running SDE Coverage Test & Visualization ({num_vis} samples)...")
        save_path = os.path.join(self.results_path, "real_reconstructions")
        os.makedirs(save_path, exist_ok=True)
        
        aug_factor = getattr(self.config, 'AUGMENTATION_FACTOR', 20)
        t_points = self.system.t_points
        sde_scales = getattr(self.config, 'SDE_SCALE_FACTORS', {'bias_scale': 1.0, 'diffusion_scale': 1.0})
        
        coverage_stats = []
        r2_stats = []
        
        total_samples = len(x_raw)
        indices = np.random.choice(total_samples, min(num_vis, total_samples), replace=False)
        
        for idx in indices:
            real_G = x_raw[idx, :, 0]
            pred_params = p_pred[idx]
            g0 = real_G[0]
            i0 = y_raw[idx, 0, 0] if y_raw is not None else 15.0 
            
            temp_theta = {'si': pred_params[0], 'sigma': pred_params[1]}
            temp_model = OGTTModel(ode_params, sys_params, temp_theta)
            n5, n6 = temp_model.find_steady_state_N(g0)
            y0 = [g0, i0, n5, n6]
            
            sim_G_ensemble = []
            sys_instance = OgttSimul()
            sys_instance.bias_scale = sde_scales.get('bias_scale', 1.0)
            sys_instance.diffusion_scale = sde_scales.get('diffusion_scale', 1.0)
            
            for _ in range(aug_factor):
                y_sim = euler_maruyama(
                    sys_instance.drift_func, sys_instance.diffusion_func, sys_instance.t_span,
                    y0, t_points, pred_params, dt_sim=0.01, system=sys_instance
                )
                sim_G_ensemble.append(y_sim[0, :])
            
            sim_G_ensemble = np.array(sim_G_ensemble)
            mean_traj = np.mean(sim_G_ensemble, axis=0)
            lower_bound = np.percentile(sim_G_ensemble, 5, axis=0)
            upper_bound = np.percentile(sim_G_ensemble, 95, axis=0)
            
            is_covered = (real_G >= lower_bound) & (real_G <= upper_bound)
            cov_ratio = np.mean(is_covered)
            coverage_stats.append(cov_ratio)
            
            recon_r2 = r2_score(real_G, mean_traj)
            r2_stats.append(recon_r2)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(t_points, lower_bound, upper_bound, color='orange', alpha=0.3, label='SDE 90% CI')
            ax.plot(t_points, mean_traj, 'orange', linestyle='--', label='Reconstructed Mean')
            ax.plot(t_points, real_G, 'k-o', label='Real Patient Data', linewidth=2)
            
            ax.set_title(f"Patient {idx}\nsi={pred_params[0]:.2f}, sigma={pred_params[1]:.2f}\nCov: {cov_ratio*100:.0f}% | $R^2$: {recon_r2:.3f}")
            ax.set_xlabel("Time (min)"); ax.set_ylabel("Glucose (mg/dL)")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_path, f"recon_patient_{idx}.png"))
            plt.close()
            
        avg_cov = np.mean(coverage_stats) if coverage_stats else 0.0
        avg_r2 = np.mean(r2_stats) if r2_stats else 0.0
        print(f"  -> SDE Evaluation: Avg Coverage={avg_cov*100:.1f}%, Avg R2={avg_r2:.4f}")
        return avg_cov, avg_r2

    def _get_model_spectral_norms(self, model):
        """
        모델 내부의 모든 Linear 레이어를 찾아 Spectral Norm을 계산합니다.
        model.modules()를 사용하여 중첩된 구조(ResidualBlock 등) 내부도 탐색합니다.
        """
        norms, indices = [], []
        linear_idx = 1
        
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                # weight_orig가 있으면(spectral_norm 적용 시) 그것을 사용, 아니면 weight 사용
                weight = getattr(layer, 'weight_orig', layer.weight)
                norm = torch.linalg.norm(weight, ord=2).item()
                norms.append(norm)
                indices.append(linear_idx)
                linear_idx += 1
        return {'indices': indices, 'norms': norms}
    
    def _analyze_spectral_norms_single(self, model, model_name):
        """단일 모델의 스펙트럴 노름을 계산하고 출력합니다."""
        print(f"\n--- Analyzing Spectral Norms for {model_name} ---")
        norms = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                weight = getattr(layer, 'weight_orig', layer.weight)
                norm = torch.linalg.norm(weight, ord=2).item()
                norms.append(norm)
                print(f"  Layer: Spectral Norm = {norm:.4f}")
        
        prod = np.prod(norms)
        print(f"Product of norms for {model_name}: {prod:.4f}")
        return prod

    def analyze_spectral_norms(self):
        """두 네트워크의 스펙트럴 노름과 그 곱(Lipshitz Constant)을 분석합니다."""
        prod_f = self._analyze_spectral_norms_single(self.f_theta, "f_theta")
        prod_g = self._analyze_spectral_norms_single(self.g_phi, "g_phi")
        total_prod = prod_f * prod_g
        
        print("\n" + "="*50)
        print(f"Total Product of Spectral Norms: {total_prod:.4f}")
        if total_prod < 1:
            print("✅ Contraction mapping condition is satisfied.")
        else:
            print("⚠️ Contraction mapping condition is NOT satisfied.")
        print("="*50 + "\n")

    def plot_spectral_norms_by_layer(self):
        """각 모델의 레이어별 스펙트럴 노름을 막대그래프로 시각화합니다."""
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

        # 데이터 저장
        all_norm_data = {
            'f_theta': f_theta_data,
            'g_phi': g_phi_data
        }
        with open(os.path.join(self.results_path, 'spectral_norms_by_layer.json'), 'w') as f:
            json.dump(all_norm_data, f, indent=4)

    # -------------------------------------------------------------------------
    # [추가] Summary Metrics Calculation
    # -------------------------------------------------------------------------
    def compute_summary_metrics(self, p_true, p_pred, real_data_loader=None):
        """[수정] R2 대신 Pearson R 저장"""
        print("Computing summary metrics...")
        param_names = self.system.param_names
        metrics = {}
        
        pearson_sum = 0
        for i, name in enumerate(param_names):
            # [수정] Pearson R 계산
            pr, _ = pearsonr(p_true[:, i], p_pred[:, i])
            metrics[f'PearsonR_{name}'] = pr
            metrics[f'MSE_{name}'] = mean_squared_error(p_true[:, i], p_pred[:, i])
            pearson_sum += pr
        
        metrics['PearsonR_Avg'] = pearson_sum / len(param_names)
        metrics['MSE_Total'] = mean_squared_error(p_true, p_pred)
        
        # Spectral Norm
        try:
            f_norms = self._get_model_spectral_norms(self.f_theta)['norms']
            g_norms = self._get_model_spectral_norms(self.g_phi)['norms']
            metrics['Lip_Total'] = np.prod(f_norms) * np.prod(g_norms)
        except:
            metrics['Lip_Total'] = -1.0
            
        return metrics

    def save_metrics_to_csv(self, metrics):
        """결과 메트릭을 CSV 파일에 추가(Append)합니다."""
        file_path = os.path.join(self.config.RESULTS_DIR, 'phase5_summary.csv')
        file_exists = os.path.isfile(file_path)
        
        # 실험 이름 추가
        metrics_with_name = {'Experiment': self.config.EXPERIMENT_NAME}
        metrics_with_name.update(metrics)
        
        with open(file_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_with_name.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_with_name)
        print(f"Appended metrics to {file_path}")