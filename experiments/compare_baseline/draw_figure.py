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
from experiments.compare_baseline.run_pilot_experiment import load_ground_truth_data

def load_results(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_loss_landscape(sys_obj, regime, beta_range, gamma_range):
    print(f"🌍 Computing Loss Landscape for Regime: {regime.upper()}...")
    
    # 🚨 [수정 1] 하드코딩 제거: 해당 Regime에 맞는 정확한 Ground Truth와 관측치를 불러옵니다.
    t_eval, x_obs_clean, theta_true, _ = load_ground_truth_data(sys_obj, regime=regime)
    
    # 지형도용 기준 노이즈 주입 (실험과 동일하게 유지)
    noise_rng = np.random.default_rng(2024)
    noise_level = 0.05
    noise = noise_rng.normal(0, noise_level * np.abs(x_obs_clean), x_obs_clean.shape)
    x_obs_noisy = x_obs_clean + noise
    
    N_val = sum([val[0] for val in sys_obj.initial_conditions])
    x_obs_noisy = np.clip(x_obs_noisy, 0.0, N_val)
    
    # 시작 조건 복원
    y0 = [x_obs_clean[0].item(), N_val - x_obs_clean[0].item(), 0.0]
    if regime == 'hard':
        y0 = [45.0, 5.0, 0.0] # Hard regime의 초기 상태 (load_ground_truth_data와 일치)
        
    B, G = np.meshgrid(beta_range, gamma_range)
    Z = np.zeros_like(B)
    
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            beta, gamma = B[i, j], G[i, j]
            # 지형도 계산 시에는 시간을 아끼기 위해 Radau 대신 기본 솔버 허용 및 빠른 종료
            sol_grid = solve_ivp(sys_obj.ode_func, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, args=([beta, gamma],))
            
            if sol_grid.success and sol_grid.y.shape[1] == len(t_eval):
                x_pred = sol_grid.y[sys_obj.observed_var_idx].reshape(-1, 1)
                Z[i, j] = np.mean((x_pred - x_obs_noisy)**2)
            else:
                Z[i, j] = np.nan 
                
    return B, G, Z, theta_true

def plot_professional_trajectories(sys_data, sys_obj, radii_keys=["1.0", "2.0"], max_lines_per_ring=20):
    # 🚨 [수정 2] Regime별로 순회하며 플롯 생성
    for regime in sys_data.keys():
        print(f"\n--- Generating figures for Regime: {regime.upper()} ---")
        methods_data = sys_data[regime]
        
        # 지형도를 포괄할 수 있는 넓은 범위 설정
        xlim_range = (0.01, 1.0) if regime == 'hard' else (0.05, 0.6)
        ylim_range = (0.01, 1.0) if regime == 'hard' else (0.01, 0.3)
        
        beta_vals = np.linspace(xlim_range[0], xlim_range[1], 50) # 해상도 약간 증가
        gamma_vals = np.linspace(ylim_range[0], ylim_range[1], 50)
        
        B, G, Z, theta_true = compute_loss_landscape(sys_obj, regime, beta_vals, gamma_vals)
        
        for method, results in methods_data.items():
            sampled_trajectories = []
            for r_key in radii_keys:
                if r_key in results:
                    trajectories = results[r_key].get('all_trajectories', [])
                    if trajectories:
                        # 재현성을 위한 시드 고정
                        random.seed(42)
                        sampled_trajectories.extend(random.sample(trajectories, min(len(trajectories), max_lines_per_ring)))
                        
            if not sampled_trajectories: continue
            
            fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
            
            # 🌟 1. 지형도 바탕색 및 등고선 (rasterized=True 필수!)
            levels = np.logspace(np.log10(np.nanmin(Z)+1e-5), np.log10(np.nanmax(Z)), 30)
            contourf_obj = ax.contourf(B, G, Z, levels=levels, norm=LogNorm(), 
                                       cmap='viridis', alpha=0.85, extend='max', rasterized=True)
            ax.contour(B, G, Z, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
            
            cbar = fig.colorbar(contourf_obj, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('MSE Loss (Log Scale)', fontsize=14)
            
            # 🌟 2. 이론적 출발선(Log-Sphere Rings) 기하학 보정
            angles = np.linspace(0, 2 * np.pi, 200)
            for r_val in [1.0, 2.0]:
                # 🚨 [수정 3] Log-Normal 섭동에 맞춘 완벽한 동심원 수식
                ring_b = theta_true[0] * np.exp(r_val * np.cos(angles))
                ring_g = theta_true[1] * np.exp(r_val * np.sin(angles))
                ax.plot(ring_b, ring_g, color='white', linestyle=':', alpha=0.6, linewidth=1.5, zorder=3)
                
            # 🌟 3. 스타일 세팅 (Ours vs Baselines 대비 극대화)
            is_ours = 'Proposed' in method
            # 배경이 viridis(어두운 파랑~밝은 노랑)이므로 대비가 중요합니다.
            # 우리의 모델은 강력한 Vermillion(오렌지/빨강 계열), 비교군은 무채색 계열 사용
            color = '#D55E00' if is_ours else '#111111' 
            line_style = '-' if is_ours else '--'       
            alpha_line = 1.0 if is_ours else 0.6 # 비교군 투명도 조절로 복잡함 완화
            line_width = 2.5 if is_ours else 1.5
            
            for traj in sampled_trajectories:
                traj_np = np.array(traj)
                if traj_np.ndim == 3: traj_np = traj_np.squeeze(1)
                if traj_np.ndim != 2 or traj_np.shape[1] < 2: continue
                
                b_vals, g_vals = traj_np[:, 0], traj_np[:, 1]
                
                # 궤적 선 그리기
                ax.plot(b_vals, g_vals, linestyle=line_style, color=color, 
                        alpha=alpha_line, linewidth=line_width, zorder=4)
                
                # 시작점 표시
                ax.scatter(b_vals[0], g_vals[0], c='silver', edgecolors='black', 
                           alpha=0.8, s=30, zorder=5)
                
                # 화살표 그리기
                if len(b_vals) > 2:
                    mid_idx = max(1, len(b_vals) // 3) # 화살표를 약간 앞쪽에 배치하여 발산 궤적에서도 잘 보이게 함
                    dx = b_vals[mid_idx] - b_vals[mid_idx-1]
                    dy = g_vals[mid_idx] - g_vals[mid_idx-1]
                    
                    if (xlim_range[0] < b_vals[mid_idx] < xlim_range[1]) and \
                       (ylim_range[0] < g_vals[mid_idx] < ylim_range[1]):
                        
                        if np.hypot(dx, dy) > 1e-3:
                            ax.annotate('', xy=(b_vals[mid_idx], g_vals[mid_idx]), 
                                        xytext=(b_vals[mid_idx-1], g_vals[mid_idx-1]),
                                        arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.3", 
                                                        color=color, lw=line_width, alpha=alpha_line), 
                                        zorder=6)
                            
            # 정답(True Parameter) 강조 (별 크기 약간 키움)
            ax.scatter(theta_true[0], theta_true[1], c='gold', marker='*', s=600, edgecolors='black', zorder=10)
            
            ax.set_title(f"{method} ({regime.capitalize()})", fontweight='bold', pad=10)
            ax.set_xlabel(r'Infection Rate ($\beta$)')
            ax.set_ylabel(r'Recovery Rate ($\gamma$)')
            
            ax.set_xlim(xlim_range)
            ax.set_ylim(ylim_range)
            
            safe_method_name = method.replace(" ", "_").replace("(", "").replace(")", "")
            # 저장 경로에 Regime 추가
            filename = f"figures/fig_landscape_{regime}_{safe_method_name}.pdf"
            os.makedirs('figures', exist_ok=True)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close(fig) # 메모리 누수 방지

if __name__ == "__main__":
    import glob
    json_files = glob.glob('results/final_benchmark/spider_web_multi_ring*.json')
    if not json_files:
        print("No JSON files found. Run 'run_experiments1.py' first.")
    else:
        latest_json = max(json_files, key=os.path.getctime)
        print(f"Loading data from: {latest_json}")
        data = load_results(latest_json)
        sys_obj = Sir()
        plot_professional_trajectories(data['sir'], sys_obj, radii_keys=["1.0", "2.0"])