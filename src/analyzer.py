# analyzer.py
import os
import json
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class BaseAnalyzer:
    """
    Encapsulates evaluation metrics and visualization routines.
    Acts as the base class for system-specific analyzers.
    """
    def __init__(self, hidden_net, param_net, normalizer, config, system, history=None):
        self.H_phi = hidden_net.to(config.DEVICE) if hidden_net else None  # f_theta -> H_phi
        self.P_psi = param_net.to(config.DEVICE) if param_net else None    # g_phi -> P_psi
        self.normalizer = normalizer
        self.config = config
        self.system = system
        self.history = history
        
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)

    # ==========================================
    # 1. Base Evaluation & Metrics
    # ==========================================

    def calculate_and_save_metrics(self, p_true, p_ours, p_base, filename="comparison_metrics.json", extra_metrics=None):
        """Computes RMSE, MAE, and Pearson R, handling empty (zero) baselines safely."""
        def calc_metrics(true, pred):
            if np.all(pred == 0):  # Safe check for zero baseline (no baseline case)
                return {"RMSE": np.nan, "MAE": np.nan, "Pearson_r": np.nan}
            
            rmse = float(np.sqrt(np.mean((true - pred)**2)))
            mae = float(np.mean(np.abs(true - pred)))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = float(pearsonr(true, pred)[0]) if len(true) > 1 and np.std(pred) > 1e-8 else 0.0
            return {"RMSE": rmse, "MAE": mae, "Pearson_r": r}

        metrics = {}
        for i, p_name in enumerate(self.system.param_names):
            metrics[p_name] = {
                "Baseline": calc_metrics(p_true[:, i], p_base[:, i]),
                "Ours": calc_metrics(p_true[:, i], p_ours[:, i])
            }
            
        if extra_metrics:
            for key, (e_true, e_ours, e_base) in extra_metrics.items():
                metrics[key] = {
                    "Baseline": calc_metrics(e_true, e_base),
                    "Ours": calc_metrics(e_true, e_ours)
                }

        print("\n" + "="*85)
        print(f"{'Metric':<12} | {'Baseline (RMSE / MAE / r)':<33} | {'Ours (RMSE / MAE / r)':<33}")
        print("-" * 85)
        for key, vals in metrics.items():
            b, o = vals['Baseline'], vals['Ours']
            b_str = f"{b['RMSE']:.4f} / {b['MAE']:.4f} / {b['Pearson_r']:.4f}" if not np.isnan(b['RMSE']) else "N/A (No Baseline)"
            print(f"{key:<12} | {b_str:<33} | {o['RMSE']:.4f} / {o['MAE']:.4f} / {o['Pearson_r']:.4f}")
        print("="*85 + "\n")

        save_path = os.path.join(self.results_path, filename)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics

    def plot_loss_curves(self):
        """Visualizes and saves independent training metrics components."""
        if self.history is None:
            return

        metric_names = {key.replace('train_', '') for key in self.history.keys() if key.startswith('train_')}
        print(f"  -> Plotting loss curves for: {metric_names}")

        for metric in metric_names:
            train_key, val_key = f'train_{metric}', f'val_{metric}'
            if train_key not in self.history:
                continue

            plt.figure(figsize=(8, 6))
            plt.plot(self.history[train_key], label='Train', color='cornflowerblue', linewidth=2)
            if val_key in self.history:
                plt.plot(self.history[val_key], label='Val', color='sandybrown', linestyle='--', linewidth=2)

            plt.title(f"Convergence: {metric}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (Log Scale)")
            plt.yscale('log')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            plt.savefig(os.path.join(self.results_path, f"{metric}.png"), dpi=150)
            plt.close()

    # ==========================================
    # 2. Spectral Norm Analysis (Restored)
    # ==========================================

    def _get_model_spectral_norms(self, model):
        """Measures the spectral norm of each linear layer in the model."""
        norms, indices = [], []
        linear_idx = 1
        scale_factor = 0.99
        
        try:
            # Trigger Hook to update effective weights (dummy forward)
            container = model.network if hasattr(model, 'network') else model.net
            first_linear = next((l for l in container if isinstance(l, torch.nn.Linear)), None)
            if first_linear is not None:
                dummy_input = torch.randn(1, first_linear.in_features, device=self.config.DEVICE)
                with torch.no_grad(): container(dummy_input)
        except Exception as e:
            print(f"  -> [Warning] Dummy forward failed for SN analysis: {e}")

        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight 
                norm = torch.linalg.norm(weight, ord=2).item()
                norms.append(scale_factor * norm)
                indices.append(linear_idx)
                linear_idx += 1
                
        return {'indices': indices, 'norms': norms}

    def analyze_spectral_norms(self):
        """Verifies Contraction Mapping condition: Lip(H_phi) * Lip(P_psi) < 1"""
        def get_prod(model, name):
            data = self._get_model_spectral_norms(model)
            prod = np.prod(data['norms'])
            print(f"  -> Product of norms for {name}: {prod:.4f}")
            return prod

        print("\n--- Spectral Norm Analysis ---")
        prod_h = get_prod(self.H_phi, "H_phi (Hidden Net)")
        prod_p = get_prod(self.P_psi, "P_psi (Param Net)")
        total_prod = prod_h * prod_p
        
        print(f"Total Lipschitz Product: {total_prod:.4f}")
        if total_prod < 1.0 + 1e-4:
            print("✅ Contraction mapping condition is satisfied.")
        else:
            print(f"⚠️ Contraction mapping condition NOT satisfied (L={total_prod:.4f}).")
        print("------------------------------\n")

    def plot_spectral_norms_by_layer(self):
        """Visualizes the spectral norm of each layer as a bar chart."""
        print("  -> Plotting Spectral Norms by layer...")
        h_phi_data = self._get_model_spectral_norms(self.H_phi)
        p_psi_data = self._get_model_spectral_norms(self.P_psi)

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

        _plot_for_model(axes[0], h_phi_data, r"$H_\phi$ (Hidden Net)")
        _plot_for_model(axes[1], p_psi_data, r"$P_\psi$ (Param Net)")

        plt.savefig(os.path.join(self.results_path, 'spectral_norms_plot.png'), dpi=150)
        plt.close(fig)

    # ==========================================
    # 3. Phase Portraits (Restored)
    # ==========================================

    def plot_phase_portraits(self, x_sample, p_target):
        """
        Visualizes the alternating fixed-point dynamics in parameter space.
        Args:
            x_sample (Tensor): A single normalized observed trajectory (1, Dim).
            p_target (np.array): Ground truth parameters corresponding to x_sample.
        """
        num_params = len(self.system.param_names)
        if num_params < 2: return

        print("  -> Generating Phase Portraits (Fixed-point dynamics)...")
        combos = list(itertools.combinations(range(num_params), 2))[:3]
        fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 6), squeeze=False)
        
        for i, (p1, p2) in enumerate(combos):
            self._plot_single_portrait(axes.flatten()[i], x_sample, p_target, (p1, p2))
            
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
        p_grid_raw[:, p1_idx] = torch.tensor(grid1.flatten(), dtype=torch.float32).to(self.config.DEVICE)
        p_grid_raw[:, p2_idx] = torch.tensor(grid2.flatten(), dtype=torch.float32).to(self.config.DEVICE)
        
        p_grid_norm = self.normalizer.normalize_params(p_grid_raw)
        x_batch = x_observed.repeat(grid1.size, 1) 
        
        with torch.no_grad():
            y_hat = self.H_phi(x_batch, p_grid_norm)
            p_next_norm = self.P_psi(x_batch, y_hat)
            
        p_next_raw = self.normalizer.denormalize_params(p_next_norm).cpu().numpy()
        dp = p_next_raw - p_grid_raw.cpu().numpy()
        
        u = dp[:, p1_idx].reshape(grid1.shape)
        v = dp[:, p2_idx].reshape(grid1.shape)
        speed = np.sqrt(u**2 + v**2)
        
        ax.streamplot(grid1, grid2, u, v, color=speed, cmap='autumn_r', linewidth=1, density=1.0)
        ax.plot(center1, center2, 'bx', markersize=12, markeredgewidth=3, label='Ground Truth', zorder=10)

        # Plot trajectories from different initial points
        start_points = [[range1[2], range2[2]], [range1[2], range2[-3]], [range1[-3], range2[2]], [range1[-3], range2[-3]]]
        
        for start in start_points:
            p_start = torch.tensor(p_target, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
            p_start[0, p1_idx], p_start[0, p2_idx] = float(start[0]), float(start[1])
            
            traj = [p_start.cpu().numpy()]
            p_curr = self.normalizer.normalize_params(p_start)
            
            with torch.no_grad():
                for _ in range(self.config.ITERATIONS): 
                    y = self.H_phi(x_observed, p_curr)
                    p_curr = self.P_psi(x_observed, y)
                    traj.append(self.normalizer.denormalize_params(p_curr).cpu().numpy())
            
            traj_np = np.concatenate(traj, axis=0)
            ax.plot(traj_np[:, p1_idx], traj_np[:, p2_idx], 'k-o', linewidth=1.5, markersize=3, alpha=0.6)
            ax.plot(traj_np[-1, p1_idx], traj_np[-1, p2_idx], 'r*', markersize=10, zorder=11)

        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_title(f"Dynamics: {name1} vs {name2}")
        ax.grid(True, alpha=0.3)
        
    def plot_scatter_comparison(self, true_vals_list, ours_vals_list, base_vals_list, metrics_dict, param_keys, param_titles=None, prefix="sim"):
        """
        Generates parity (scatter) plots uniformly for any system.
        Args:
            true_vals_list, ours_vals_list, base_vals_list: Lists of 1D arrays to plot.
            param_names: Titles/keys for each subplot.
        """
        param_titles = param_titles or param_keys
        num_params = len(param_keys)
        fig, axes = plt.subplots(1, num_params, figsize=(5*num_params, 5.5))
        if num_params == 1: axes = [axes]

        for i, p_key in enumerate(param_keys):
            ax = axes[i]
            t_v, o_v, b_v = true_vals_list[i], ours_vals_list[i], base_vals_list[i]
            
            is_dummy_base = np.all(b_v == 0)
            is_dummy_ours = np.all(o_v == 0)
            
            if not is_dummy_base:
                ax.scatter(t_v, b_v, alpha=0.4, color='indianred', label='Baseline', marker='x')
            if not is_dummy_ours:
                ax.scatter(t_v, o_v, alpha=0.5, color='steelblue', label='Ours', marker='o', edgecolors='white', lw=0.5)
            
            # Automatic axis scaling
            min_vals, max_vals = [np.min(t_v)], [np.max(t_v)]
            if not is_dummy_base:
                min_vals.append(np.min(b_v)); max_vals.append(np.max(b_v))
            if not is_dummy_ours:
                min_vals.append(np.min(o_v)); max_vals.append(np.max(o_v))
                
            min_v, max_v = min(min_vals), max(max_vals)
            ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Ideal')
            
            b_mets, o_mets = metrics_dict[p_key]['Baseline'], metrics_dict[p_key]['Ours']
            b_text = f'RMSE: {b_mets["RMSE"]:.4f}\nr: {b_mets["Pearson_r"]:.4f}' if not np.isnan(b_mets["RMSE"]) else 'N/A'
            o_text = f'RMSE: {o_mets["RMSE"]:.4f}\nr: {o_mets["Pearson_r"]:.4f}' if not np.isnan(o_mets["RMSE"]) else 'N/A'
            
            textstr = '\n'.join((r'$\bf{Baseline}$', b_text, '', r'$\bf{Ours}$', o_text))
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

            ax.set_title(param_titles[i], fontsize=14, pad=15)
            ax.set_xlabel('True Value'); ax.set_ylabel('Predicted Value')
            ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, f'{prefix}_scatter_comparison.png'), dpi=300)
        plt.close()

    def plot_trajectory_panel(self, ax, t, y_true, y_base, y_ours, title, ylabel=None, xlabel="Time"):
        """Draws a single trajectory comparison on a given matplotlib axis."""
        ax.plot(t, y_true, 'k-', lw=2, label='True Dynamics')
        if not np.all(y_base == 0):
            ax.plot(t, y_base, 'indianred', linestyle='--', lw=2, label='Baseline')
        if not np.all(y_ours == 0):
            ax.plot(t, y_ours, 'steelblue', linestyle='-.', lw=2, label='Ours')
        ax.set_title(title); ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
        
# ==========================================
# Benchmark Specific Analyzers (Child Classes)
# ==========================================

class SIRAnalyzer(BaseAnalyzer):
    def evaluate_simulation(self, p_true, p_ours, p_base):
        valid_base = not np.all(p_base == 0)
        valid_ours = not np.all(p_ours == 0)
        
        R0_true = p_true[:, 0] / p_true[:, 1]
        R0_ours = p_ours[:, 0] / p_ours[:, 1] if valid_ours else np.zeros_like(R0_true)
        R0_base = p_base[:, 0] / p_base[:, 1] if valid_base else np.zeros_like(R0_true)
        
        metrics_dict = self.calculate_and_save_metrics(
            p_true, p_ours, p_base, "sir_comparison_metrics.json", 
            extra_metrics={"R0": (R0_true, R0_ours, R0_base)}
        )
        
        t_vals = [p_true[:, 0], p_true[:, 1], R0_true]
        o_vals = [p_ours[:, 0], p_ours[:, 1], R0_ours]
        b_vals = [p_base[:, 0], p_base[:, 1], R0_base]
        
        # [핵심 버그 픽스] 키 값과 타이틀 값을 분리하여 전달합니다.
        p_keys = list(self.system.param_names) + ["R0"]
        p_titles = [r'Infection Rate ($\beta$)', r'Recovery Rate ($\gamma$)', r'Basic Reproduction ($\mathcal{R}_0$)']
        
        self.plot_scatter_comparison(t_vals, o_vals, b_vals, metrics_dict, p_keys, p_titles, prefix="sir")
        self._plot_trajectory_comparison(p_true, p_ours, p_base, R0_true, valid_base, valid_ours)

    def _plot_trajectory_comparison(self, p_true, p_ours, p_base, R0_true, valid_base, valid_ours):
        from scipy.integrate import solve_ivp
        try:
            idx_epidemic, idx_decay = np.where(R0_true > 1.2)[0][0], np.where(R0_true < 0.8)[0][0]
        except IndexError:
            idx_epidemic, idx_decay = 0, len(R0_true)-1

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t_eval = np.linspace(0, 110, 200)
        def sir_ode(t, y, beta, gamma): return [-beta * y[0] * y[1] / 50.0, beta * y[0] * y[1] / 50.0 - gamma * y[1], gamma * y[1]]

        for i, idx in enumerate([idx_epidemic, idx_decay]):
            title_prefix = f"Epidemic (R0={R0_true[idx]:.2f})" if i == 0 else f"Decay (R0={R0_true[idx]:.2f})"
            sol_t = solve_ivp(sir_ode, (0, 110), [49., 1., 0.], args=tuple(p_true[idx]), t_eval=t_eval)
            
            y_o_s, y_o_i = np.zeros_like(sol_t.y[0]), np.zeros_like(sol_t.y[1])
            if valid_ours:
                sol_o = solve_ivp(sir_ode, (0, 110), [49., 1., 0.], args=tuple(p_ours[idx]), t_eval=t_eval)
                y_o_s, y_o_i = sol_o.y[0], sol_o.y[1]
                
            y_b_s, y_b_i = np.zeros_like(sol_t.y[0]), np.zeros_like(sol_t.y[1])
            if valid_base:
                sol_b = solve_ivp(sir_ode, (0, 110), [49., 1., 0.], args=tuple(p_base[idx]), t_eval=t_eval)
                y_b_s, y_b_i = sol_b.y[0], sol_b.y[1]

            self.plot_trajectory_panel(axes[i, 0], sol_t.t, sol_t.y[0], y_b_s, y_o_s, f"S(t) | {title_prefix}", "Population")
            self.plot_trajectory_panel(axes[i, 1], sol_t.t, sol_t.y[1], y_b_i, y_o_i, f"I(t) Hidden | {title_prefix}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'sir_trajectory_comparison.png'), dpi=300)
        plt.close()

class LotkaVolterraAnalyzer(BaseAnalyzer):
    def evaluate_simulation(self, p_true, p_ours, p_base):
        metrics_dict = self.calculate_and_save_metrics(p_true, p_ours, p_base, "lv_comparison_metrics.json")
        t_vals = [p_true[:, i] for i in range(p_true.shape[1])]
        o_vals = [p_ours[:, i] for i in range(p_ours.shape[1])]
        b_vals = [p_base[:, i] for i in range(p_base.shape[1])]
        
        self.plot_scatter_comparison(t_vals, o_vals, b_vals, metrics_dict, self.system.param_names, prefix="lv")
        self._plot_lv_trajectories_and_phase(p_true, p_ours, p_base)

    def _plot_lv_trajectories_and_phase(self, p_true, p_ours, p_base):
        from scipy.integrate import solve_ivp
        idx, valid_base, valid_ours = np.random.randint(0, len(p_true)), not np.all(p_base == 0), not np.all(p_ours == 0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        def lv_ode(t, y, a, b, d, g): return [a*y[0] - b*y[0]*y[1], d*y[0]*y[1] - g*y[1]]
        y0 = [y[0] if isinstance(y, list) else y for y in self.system.initial_conditions]
        
        sol_t = solve_ivp(lv_ode, (0, 50), y0, args=tuple(p_true[idx]), t_eval=np.linspace(0, 50, 500))
        
        y_o_x, y_o_y = np.zeros_like(sol_t.y[0]), np.zeros_like(sol_t.y[1])
        if valid_ours:
            sol_o = solve_ivp(lv_ode, (0, 50), y0, args=tuple(p_ours[idx]), t_eval=np.linspace(0, 50, 500))
            y_o_x, y_o_y = sol_o.y[0], sol_o.y[1]
            
        y_b_x, y_b_y = np.zeros_like(sol_t.y[0]), np.zeros_like(sol_t.y[1])
        if valid_base:
            sol_b = solve_ivp(lv_ode, (0, 50), y0, args=tuple(p_base[idx]), t_eval=np.linspace(0, 50, 500))
            y_b_x, y_b_y = sol_b.y[0], sol_b.y[1]

        self.plot_trajectory_panel(axes[0], sol_t.t, sol_t.y[0], y_b_x, y_o_x, "Prey (Observed)")
        self.plot_trajectory_panel(axes[1], sol_t.t, sol_t.y[1], y_b_y, y_o_y, "Predator (Hidden)")

        axes[2].plot(sol_t.y[0], sol_t.y[1], 'k-', lw=2, label='True Cycle')
        if valid_base: axes[2].plot(y_b_x, y_b_y, 'indianred', linestyle='--', lw=2, label='Baseline')
        if valid_ours: axes[2].plot(y_o_x, y_o_y, 'steelblue', linestyle='-.', lw=2, label='Ours')
        axes[2].set_title("Phase Portrait"); axes[2].set_xlabel("Prey"); axes[2].set_ylabel("Predator"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.results_path, 'lv_dynamics_comparison.png'), dpi=300); plt.close()

class OgttSimulAnalyzer(BaseAnalyzer):
    def evaluate_simulation(self, p_true, p_ours, p_base):
        metrics = self.calculate_and_save_metrics(p_true, p_ours, p_base, "ogtt_sim_metrics.json")
        self._do_ogtt_plots(p_true, p_ours, p_base, metrics, "sim")
        self._plot_ogtt_trajectories(p_true, p_ours, p_base)

    def evaluate_real_data(self, p_true, p_ours, p_base):
        metrics = self.calculate_and_save_metrics(p_true, p_ours, p_base, "ogtt_real_metrics.json")
        self._do_ogtt_plots(p_true, p_ours, p_base, metrics, "real")

    def _do_ogtt_plots(self, p_true, p_ours, p_base, metrics_dict, prefix):
        t_vals = [p_true[:, i] for i in range(p_true.shape[1])]
        o_vals = [p_ours[:, i] for i in range(p_ours.shape[1])]
        b_vals = [p_base[:, i] for i in range(p_base.shape[1])]
        self.plot_scatter_comparison(t_vals, o_vals, b_vals, metrics_dict, self.system.param_names, prefix=prefix)
        self._plot_symmetric_collapse(p_true, p_ours, p_base, prefix)

    def _plot_ogtt_trajectories(self, p_true, p_ours, p_base):
        from scipy.integrate import solve_ivp
        from systems.ogtt_simul import OGTTModel, SYS_PARAMS, ODE_PARAMS
        
        idx = np.random.randint(0, len(p_true)) 
        valid_base, valid_ours = not np.all(p_base == 0), not np.all(p_ours == 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        y0_2d = [y[0] if isinstance(y, list) else y for y in self.system.initial_conditions][:2]
        G0, I0 = y0_2d[0], y0_2d[1]
        
        # 2. 파라미터(theta)에 맞춰 N5, N6의 Steady State를 계산하여 4차원 벡터로 패딩하는 헬퍼 함수
        def get_full_y0(params):
            theta_dict = {name: val for name, val in zip(self.system.param_names, params)}
            model = OGTTModel(ODE_PARAMS, SYS_PARAMS, theta_dict)
            n5_ss, n6_ss = model.find_steady_state_N(G0)
            return [G0, I0, n5_ss, n6_ss]
            
        y0_true = get_full_y0(p_true[idx])
        sol_t = solve_ivp(lambda t, y: self.system.ode_func(t, y, tuple(p_true[idx])), (0, 120), y0_true, t_eval=np.linspace(0, 120, 200))
        
        y_o_g, y_o_i = np.zeros_like(sol_t.y[0]), np.zeros_like(sol_t.y[1])
        if valid_ours:
            y0_ours = get_full_y0(p_ours[idx])
            sol_o = solve_ivp(lambda t, y: self.system.ode_func(t, y, tuple(p_ours[idx])), (0, 120), y0_ours, t_eval=np.linspace(0, 120, 200))
            y_o_g, y_o_i = sol_o.y[0], sol_o.y[1]
            
        y_b_g, y_b_i = np.zeros_like(sol_t.y[0]), np.zeros_like(sol_t.y[1])
        if valid_base:
            y0_base = get_full_y0(p_base[idx])
            sol_b = solve_ivp(lambda t, y: self.system.ode_func(t, y, tuple(p_base[idx])), (0, 120), y0_base, t_eval=np.linspace(0, 120, 200))
            y_b_g, y_b_i = sol_b.y[0], sol_b.y[1]

        self.plot_trajectory_panel(axes[0], sol_t.t, sol_t.y[0], y_b_g, y_o_g, "Glucose (Observed)", "mg/dL", "Time (min)")
        self.plot_trajectory_panel(axes[1], sol_t.t, sol_t.y[1], y_b_i, y_o_i, "Insulin (Hidden Error)", "uU/mL", "Time (min)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'sim_trajectory_comparison.png'), dpi=300)
        plt.close()
        
    def _plot_symmetric_collapse(self, p_true, p_ours, p_base, prefix="sim"):
        valid_base, valid_ours = not np.all(p_base == 0), not np.all(p_ours == 0)
        idx_si, idx_sigma = 0, 1 
        si_true, sigma_true = p_true[:, idx_si], p_true[:, idx_sigma]
        si_ours, sigma_ours = p_ours[:, idx_si], p_ours[:, idx_sigma]
        si_base, sigma_base = p_base[:, idx_si], p_base[:, idx_sigma]
        
        sns.set_style("ticks")
        fig = plt.figure(figsize=(20, 6))
        
        ax1 = fig.add_subplot(131)
        ax1.scatter(si_true, sigma_true, c='grey', alpha=0.15, s=20, label='Ground Truth')
        if valid_base: ax1.scatter(si_base, sigma_base, c='crimson', alpha=0.5, s=15, marker='x', label='Baseline')
        if valid_ours: ax1.scatter(si_ours, sigma_ours, c='steelblue', alpha=0.6, s=15, edgecolors='white', lw=0.5, label='Ours')
        ax1.set_xscale('log'); ax1.set_yscale('log')
        vmin, vmax = min(si_true.min(), sigma_true.min()), max(si_true.max(), sigma_true.max())
        ax1.plot([vmin, vmax], [vmin, vmax], 'b--', linewidth=1.5, label='y=x (Symmetric Line)')
        ax1.set_title("1. Log-Log Joint Distribution", fontsize=14, fontweight='bold'); ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.2)
        
        ax2 = fig.add_subplot(132)
        sns.kdeplot(np.log10(si_true * sigma_true), ax=ax2, color='grey', fill=True, alpha=0.3, linewidth=2, label='Log($K_{true}$)')
        if valid_base: sns.kdeplot(np.log10(si_base * sigma_base), ax=ax2, color='crimson', linestyle='--', linewidth=2, label='Log($K_{base}$)')
        if valid_ours: sns.kdeplot(np.log10(si_ours * sigma_ours), ax=ax2, color='steelblue', linewidth=2, label='Log($K_{ours}$)')
        ax2.set_title("2. Product Consistency (Stiff Direction)", fontsize=14, fontweight='bold'); ax2.legend(); ax2.grid(True, alpha=0.2)
        
        ax3 = fig.add_subplot(133)
        if valid_base: ax3.scatter(np.log10(si_true), np.log10(si_base), c='crimson', alpha=0.4, s=15, marker='x', label='Baseline')
        if valid_ours: ax3.scatter(np.log10(si_true), np.log10(si_ours), c='steelblue', alpha=0.5, s=15, edgecolors='white', lw=0.5, label='Ours')
        min_log, max_log = np.log10(si_true).min(), np.log10(si_true).max()
        ax3.plot([min_log, max_log], [min_log, max_log], 'k--', lw=2, label='Ideal (y=x)')
        ax3.set_title("3. $S_I$ Prediction (Sloppy Direction)", fontsize=14, fontweight='bold'); ax3.legend(); ax3.grid(True, alpha=0.2)
        
        plt.tight_layout(); plt.savefig(os.path.join(self.results_path, f'{prefix}_symmetric_collapse.png'), dpi=300); plt.close()

# ==========================================
# 3. Factory Function 
# ==========================================
def get_analyzer_class(system_name):
    if system_name == 'sir':
        return SIRAnalyzer
    elif system_name == 'lotka_volterra':
        return LotkaVolterraAnalyzer
    elif system_name == 'ogtt_simul':
        return OgttSimulAnalyzer
    else:
        print(f"  -> [Warning] Unknown system '{system_name}'. Falling back to BaseAnalyzer.")
        return BaseAnalyzer