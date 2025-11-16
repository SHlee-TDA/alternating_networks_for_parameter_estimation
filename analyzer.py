# analyzer.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools 
import json

class Analyzer:
    """학습된 모델의 성능 분석 및 시각화를 담당하는 클래스"""
    def __init__(self, f_theta, g_phi, test_loader, config, system, p_initial_guess, normalizer=None, history=None):
        self.f_theta = f_theta.to(config.DEVICE)
        self.g_phi = g_phi.to(config.DEVICE)
        self.test_loader = test_loader
        self.config = config
        self.system = system
        self.normalizer = normalizer
        self.p_initial_guess = p_initial_guess
        self.history = history
        # 결과 저장 경로를 시스템 및 실험 이름에 따라 동적으로 설정
        self.results_path = os.path.join(config.RESULTS_DIR, config.SYSTEM_NAME, config.EXPERIMENT_NAME)
        os.makedirs(self.results_path, exist_ok=True)

    def plot_loss_curves(self):
        if self.history is None:
            print("No history found, skipping loss curve plotting.")
            return

        print("Plotting and saving loss curves...")
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.history['train_total_loss']) + 1)

        ax.plot(epochs, self.history['train_loss_f'], label='Train Loss of f_theta', color='blue')
        ax.plot(epochs, self.history['train_loss_g'], label='Train Loss of g_phi', color='green')
        ax.plot(epochs, self.history['val_loss_f'], label='Validation Loss of f_theta', color='orange', linestyle='--')
        ax.plot(epochs, self.history['val_loss_g'], label='Validation Loss of g_phi', color='red', linestyle='--')

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training and Validation Loss ({self.config.EXPERIMENT_NAME})")
        ax.legend()
        ax.grid(True)

        save_path = os.path.join(self.results_path, 'loss_curves.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        # [3단계 기능] 손실 데이터도 JSON으로 저장
        data_path = os.path.join(self.results_path, 'loss_history.json')
        # history는 이미 딕셔너리이므로 변환 필요 없음
        with open(data_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Saved loss history data to {data_path}")
    
    def _get_model_spectral_norms(self, model):
        norms, indices = [], []
        linear_idx = 1
        for layer in model.network.children():
            if isinstance(layer, torch.nn.Linear):
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
        for layer in model.network.children():
            if isinstance(layer, torch.nn.Linear):
                # spectral_norm 적용 여부에 따라 올바른 가중치를 가져옵니다.
                weight = getattr(layer, 'weight_orig', layer.weight)
                norm = torch.linalg.norm(weight, ord=2).item()
                norms.append(norm)
                print(f"  Layer: Spectral Norm = {norm:.4f}")
        prod = np.prod(norms)
        print(f"Product of norms for {model_name}: {prod:.4f}")
        return prod

    def analyze_spectral_norms(self):
        """두 네트워크의 스펙트럴 노름과 그 곱을 분석합니다."""
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

    def evaluate_predictions(self):
        """테스트 데이터셋에 대해 모델의 최종 예측을 평가합니다."""
        print("Evaluating model predictions...")
        self.f_theta.eval()
        self.g_phi.eval()
        all_p_true, all_p_pred = [], []
        with torch.no_grad():
            for x_batch, _, p_batch in self.test_loader:
                x_batch = x_batch.to(self.config.DEVICE)
                
                p_n_norm = self.normalizer.normalize(self.p_initial_guess.repeat(x_batch.size(0), 1))
                
                for _ in range(self.config.ITERATIONS):
                    # f_theta에 입력하기 위해 역정규화
                    p_n = self.normalizer.denormalize(p_n_norm)
                    y_hat = self.f_theta(x_batch, p_n)
                    # g_phi는 정규화된 p_{n+1}을 출력
                    p_n_norm = self.g_phi(x_batch, y_hat)
                
                p_final_pred = self.normalizer.denormalize(p_n_norm)
                
                all_p_pred.append(p_final_pred.cpu().numpy())
                all_p_true.append(p_batch.numpy())
        return np.concatenate(all_p_true), np.concatenate(all_p_pred)

    def plot_scatter(self, p_true, p_pred):
        """예측 정확도를 산점도로 시각화합니다."""
        print("Plotting and saving scatter results...")
        param_names_latex = [f'$\\${name}$' for name in self.system.param_names]
        num_params = len(param_names_latex)
        
        ncols = 2 if num_params > 1 else 1
        nrows = (num_params + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        axes = axes.flatten()

        for i in range(num_params):
            r_val = np.corrcoef(p_true[:, i], p_pred[:, i])[0, 1]
            axes[i].scatter(p_true[:, i], p_pred[:, i], alpha=0.5, s=10)
            axes[i].plot([p_true[:, i].min(), p_true[:, i].max()], [p_true[:, i].min(), p_true[:, i].max()], 'r--')
            axes[i].set_xlabel(f"Exact {param_names_latex[i]}")
            axes[i].set_ylabel(f"Predicted {param_names_latex[i]}")
            axes[i].set_title(f"Parameter: {param_names_latex[i]}, R={r_val:.4f}")
            axes[i].grid(True)
        
        for j in range(num_params, len(axes)):
            axes[j].set_visible(False)
            
        save_path = os.path.join(self.results_path, 'scatter_plot.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        # Save values
        data_path = os.path.join(self.results_path, 'predictions.npz')
        np.savez(data_path, p_true=p_true, p_pred=p_pred)
        print(f"Saved prediction data to {data_path}")
        
    def plot_spectral_norms_by_layer(self):
        """각 모델의 레이어별 스펙트럴 노름을 막대그래프로 시각화합니다."""
        """각 모델의 레이어별 스펙트럴 노름을 막대그래프로 시각화합니다."""
        print("Plotting and saving spectral norms by layer...")

        # [수정] 헬퍼를 사용하여 데이터 먼저 추출
        f_theta_data = self._get_model_spectral_norms(self.f_theta)
        g_phi_data = self._get_model_spectral_norms(self.g_phi)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        fig.suptitle(...)

        # [수정] 내부 함수 대신 추출된 데이터로 플롯
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

        # [추가] 스펙트럴 노름 데이터 저장
        all_norm_data = {
            'f_theta': f_theta_data,
            'g_phi': g_phi_data
        }
        data_path = os.path.join(self.results_path, 'spectral_norms_by_layer.json')
        with open(data_path, 'w') as f:
            json.dump(all_norm_data, f, indent=4)
        print(f"Saved spectral norm data to {data_path}")

    def plot_phase_portraits(self):
        print("Plotting and saving phase portraits...")
        num_params = len(self.system.param_names)
        if num_params < 2:
            print("Cannot plot phase portraits for less than 2 parameters.")
            return

        # 1. 모든 2D 파라미터 조합 생성
        all_combinations = list(itertools.combinations(range(num_params), 2))
        
        # 2. 최대 3개의 플롯만 그리도록 제한
        max_plots = 3
        plot_combinations = all_combinations[:max_plots]
        num_plots = len(plot_combinations)

        # 3. 플롯 개수에 맞게 서브플롯 동적 생성
        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f'Phase Portraits for {self.config.EXPERIMENT_NAME}', fontsize=16)
        
        x_sample, _, p_sample = self.test_loader.dataset[0]
        x_sample_batch = x_sample.unsqueeze(0).to(self.config.DEVICE)
        true_params = p_sample.numpy()
        
        # 4. 선택된 조합에 대해 플롯 생성
        for i, p_dims in enumerate(plot_combinations):
            self._plot_single_portrait(axes[i], x_sample_batch, true_params, p_dims)

        save_path = os.path.join(self.results_path, 'phase_portraits.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig)

    def _plot_single_portrait(self, ax, x_observed, true_params, p_dims, num_iterations=10):
        """하나의 2D 파라미터 평면에 대한 위상 초상과 궤적을 그립니다."""
        
        # --- (self.normalizer가 있는지 확인) ---
        if self.normalizer is None:
            print("Normalizer is missing. Cannot plot phase portraits correctly.")
            return
            
        p1_idx, p2_idx = p_dims
        p1_name, p2_name = self.system.param_names[p1_idx], self.system.param_names[p2_idx]
        
        # 1. 그리드는 원본(Raw) 스케일로 생성 (이 부분은 동일)
        p1_range = np.linspace(true_params[p1_idx] * 0.5, true_params[p1_idx] * 1.5, 20)
        p2_range = np.linspace(true_params[p2_idx] * 0.5, true_params[p2_idx] * 1.5, 20)
        p1_grid, p2_grid = np.meshgrid(p1_range, p2_range)
        
        # p0_grid_raw: 원본 스케일의 그리드 포인트
        p0_grid_raw = torch.tensor(true_params, dtype=torch.float32).repeat(p1_grid.size, 1).to(self.config.DEVICE)
        p0_grid_raw[:, p1_idx] = torch.tensor(p1_grid.flatten(), dtype=torch.float32)
        p0_grid_raw[:, p2_idx] = torch.tensor(p2_grid.flatten(), dtype=torch.float32)
        
        # [FIX] 2. 연산을 위해 그리드를 정규화합니다. (p_n_norm)
        p0_grid_norm = self.normalizer.normalize(p0_grid_raw)
        
        with torch.no_grad():
            x_batch = x_observed.repeat(p1_grid.size, 1)
            
            # [FIX] 3. f_theta에는 정규화된 p를 denormalize해서 입력
            y_hat = self.f_theta(x_batch, self.normalizer.denormalize(p0_grid_norm))
            
            # [FIX] 4. g_phi는 정규화된 p_{n+1}을 출력 (p1_grid_norm)
            p1_grid_norm = self.g_phi(x_batch, y_hat)
        
        # [FIX] 5. 벡터 필드(dp)를 계산합니다.
        # p_{n+1}_raw 와 p_{n}_raw를 구해서 빼야 플롯에 의미가 있습니다.
        p1_grid_raw = self.normalizer.denormalize(p1_grid_norm).cpu().numpy()
        dp_raw = p1_grid_raw - p0_grid_raw.cpu().numpy()

        # [FIX] 6. 원본 스케일의 그리드(p1_grid, p2_grid)에 원본 스케일의 벡터(dp_raw)를 그립니다.
        ax.quiver(p1_grid, p2_grid, dp_raw[:, p1_idx], dp_raw[:, p2_idx], color='teal', alpha=0.6, width=0.003)
        ax.plot(true_params[p1_idx], true_params[p2_idx], 'r*', markersize=18, label='True Value', zorder=10)
        
        
        # --- [FIX] 궤적(Trajectory) 계산 수정 ---
        
        # [FIX] 7. 시작점을 원본(raw) 스케일로 정의
        p_start_raw = torch.tensor(true_params, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
        p_start_raw[0, p1_idx] = p1_range[0]
        p_start_raw[0, p2_idx] = p2_range[-1]

        trajectory_raw = [p_start_raw.clone()] # 플롯을 위해 원본 스케일 값 저장
        
        # [FIX] 8. 반복은 정규화된(normalized) 값으로 시작
        p_current_norm = self.normalizer.normalize(p_start_raw)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                # [FIX] 9. f_theta 입력을 위해 denormalize
                p_current_raw_for_f = self.normalizer.denormalize(p_current_norm)
                
                # 10. y_hat 계산
                y_hat = self.f_theta(x_observed, p_current_raw_for_f)
                
                # [FIX] 11. g_phi에서 다음 스텝(normalized) p를 얻음
                p_next_norm = self.g_phi(x_observed, y_hat)
                
                # [FIX] 12. 플롯을 위해 원본(raw) 스케일로 변환하여 저장
                trajectory_raw.append(self.normalizer.denormalize(p_next_norm).clone())
                
                # [FIX] 13. 상태 업데이트는 정규화된(normalized) 값으로 수행
                p_current_norm = p_next_norm
        
        # [FIX] 14. 궤적 플롯 (이제 trajectory_raw는 모두 원본 스케일 값임)
        traj_np = torch.cat(trajectory_raw).cpu().numpy()
        starts, ends = traj_np[:-1], traj_np[1:]
        vectors = ends - starts
        
        ax.quiver(starts[:, p1_idx], starts[:, p2_idx], vectors[:, p1_idx], vectors[:, p2_idx], 
                  color='purple', angles='xy', scale_units='xy', scale=1, width=0.004, 
                  label=f'Iterative Path ({num_iterations} steps)', zorder=5)
        ax.scatter(traj_np[:, p1_idx], traj_np[:, p2_idx], color='purple', s=30, zorder=6)
        
        ax.set_xlabel(f"Parameter ${p1_name}$")
        ax.set_ylabel(f"Parameter ${p2_name}$")
        ax.set_title(f"Phase Portrait on (${p1_name}$, ${p2_name}$) Plane")
        ax.legend()
        ax.grid(True)