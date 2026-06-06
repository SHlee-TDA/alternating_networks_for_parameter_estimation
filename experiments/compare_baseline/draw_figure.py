import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.colors import LogNorm

# --- LaTeX 논문용 고품질 세팅 ---
plt.rcParams.update({
    'font.size': 14, 
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'figure.dpi': 300, 
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.grid': False
})

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from systems.sir import Sir

def load_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_loss_landscape(sys_obj, beta_range, gamma_range):
    print("🌍 Computing the Loss Landscape (this may take 5-10 seconds)...")
    theta_true = np.array([0.25, 0.1])
    t_eval = sys_obj.t_points
    y0 = [48.0, 2.0, 0.0]
    
    sol = solve_ivp(sys_obj.ode_func, [0, 110], y0, t_eval=t_eval, args=(theta_true,))
    x_obs_clean = sol.y[sys_obj.observed_var_idx].reshape(-1, 1)
    
    noise_rng = np.random.default_rng(2024)
    noise_level = 0.02
    noise = noise_rng.normal(0, noise_level * np.abs(x_obs_clean), x_obs_clean.shape)
    x_obs_noisy = x_obs_clean + noise
    x_obs_noisy = np.clip(x_obs_noisy, 0.0, 50.0)
    
    B, G = np.meshgrid(beta_range, gamma_range)
    Z = np.zeros_like(B)
    
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            beta, gamma = B[i, j], G[i, j]
            sol_grid = solve_ivp(sys_obj.ode_func, [0, 110], y0, t_eval=t_eval, args=([beta, gamma],))
            if sol_grid.success and sol_grid.y.shape[1] == len(t_eval):
                x_pred = sol_grid.y[sys_obj.observed_var_idx].reshape(-1, 1)
                Z[i, j] = np.mean((x_pred - x_obs_noisy)**2)
            else:
                Z[i, j] = np.nan 
                
    return B, G, Z

def plot_professional_trajectories(sys_data, sys_obj, radii_keys=["1.0", "2.0"], max_lines_per_ring=15):
    methods = list(sys_data.keys())
    theta_true = np.array([0.25, 0.1])
    
    xlim_range = (0.05, 0.45)
    ylim_range = (0.01, 0.25)
    
    beta_vals = np.linspace(xlim_range[0], xlim_range[1], 40)
    gamma_vals = np.linspace(ylim_range[0], ylim_range[1], 40)
    B, G, Z = compute_loss_landscape(sys_obj, beta_vals, gamma_vals)
    
    random.seed(42)
    
    for method in methods:
        # 각 고리별 궤적을 모두 모읍니다.
        sampled_trajectories = []
        for r_key in radii_keys:
            if r_key in sys_data[method]:
                trajectories = sys_data[method][r_key].get('all_trajectories', [])
                if trajectories:
                    sampled_trajectories.extend(random.sample(trajectories, min(len(trajectories), max_lines_per_ring)))
                    
        if not sampled_trajectories: continue
        
        fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        
        # 🌟 1. 지형도 바탕색 및 등고선
        levels = np.logspace(-1, 4, 30)
        contourf_obj = ax.contourf(B, G, Z, levels=levels, norm=LogNorm(), 
                                   cmap='viridis', alpha=0.85, extend='max')
        ax.contour(B, G, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # 컬러바 부착
        cbar = fig.colorbar(contourf_obj, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('MSE Loss (Log Scale)', fontsize=14)
        
        # 🌟 2. 이론적 출발선(Theoretical Rings) 그리기
        angles = np.linspace(0, 2 * np.pi, 200)
        for r_val in [1.0, 2.0]:
            # R에 비례한 타원(Ellipse) 궤도 수식
            ring_b = theta_true[0] * (1 + r_val * np.cos(angles))
            ring_g = theta_true[1] * (1 + r_val * np.sin(angles))
            ax.plot(ring_b, ring_g, color='black', linestyle=':', alpha=0.4, linewidth=1.5, zorder=3)
            # 고리 라벨(R=1.0, R=2.0) 추가 (옵션)
            if r_val == 2.0:
                ax.text(theta_true[0], theta_true[1] + theta_true[1]*r_val + 0.005, f'R={r_val}', 
                        color='black', alpha=0.6, fontsize=12, ha='center')
        
        # 🌟 3. 스타일 세팅
        is_ours = 'Proposed' in method
        color = '#D55E00' if is_ours else '#FFFFFF' 
        line_style = '-' if is_ours else '--'       
        alpha_line = 1.0 if is_ours else 0.8
        line_width = 2.5 if is_ours else 1.8
        
        for traj in sampled_trajectories:
            traj_np = np.array(traj)
            if traj_np.ndim == 3: traj_np = traj_np.squeeze(1)
            if traj_np.ndim != 2 or traj_np.shape[1] < 2: continue
            
            b_vals, g_vals = traj_np[:, 0], traj_np[:, 1]
            
            # 궤적 선 그리기
            ax.plot(b_vals, g_vals, linestyle=line_style, color=color, 
                    alpha=alpha_line, linewidth=line_width, zorder=4)
            
            # 시작점 표시
            ax.scatter(b_vals[0], g_vals[0], c='dimgray', edgecolors='white', 
                       alpha=1.0, s=40, zorder=5)
            
            # 🌟 4. 화살표 비율 세련되게 축소 및 렌더링
            if len(b_vals) > 2:
                mid_idx = max(1, len(b_vals) // 2)
                dx = b_vals[mid_idx] - b_vals[mid_idx-1]
                dy = g_vals[mid_idx] - g_vals[mid_idx-1]
                
                if (xlim_range[0] < b_vals[mid_idx] < xlim_range[1]) and \
                   (ylim_range[0] < g_vals[mid_idx] < ylim_range[1]):
                    
                    if np.hypot(dx, dy) > 1e-3:
                        # 🚨 head_length와 head_width를 날렵하게 축소
                        ax.annotate('', xy=(b_vals[mid_idx], g_vals[mid_idx]), 
                                    xytext=(b_vals[mid_idx-1], g_vals[mid_idx-1]),
                                    arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.25", 
                                                    color=color, lw=line_width, alpha=alpha_line), 
                                    zorder=6)
                        
        # 정답(True Parameter) 강조
        ax.scatter(theta_true[0], theta_true[1], c='gold', marker='*', s=500, edgecolors='black', zorder=10)
        
        ax.set_title(method, fontweight='bold', pad=10)
        ax.set_xlabel(r'Infection Rate ($\beta$)')
        ax.set_ylabel(r'Recovery Rate ($\gamma$)')
        
        ax.set_xlim(xlim_range)
        ax.set_ylim(ylim_range)
        
        safe_method_name = method.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"figures/fig_landscape_accessibility_{safe_method_name}.pdf"
        os.makedirs('figures', exist_ok=True)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

if __name__ == "__main__":
    import glob
    # JSON 파일 포맷 확인 (멀티링 데이터)
    json_files = glob.glob('results/final_benchmark/spider_web_multi_ring*.json')
    if not json_files:
        print("No JSON files found.")
    else:
        latest_json = max(json_files, key=os.path.getctime)
        data = load_results(latest_json)
        sys_obj = Sir()
        plot_professional_trajectories(data['sir'], sys_obj, radii_keys=["1.0", "2.0"])