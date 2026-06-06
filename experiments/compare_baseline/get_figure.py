# plot_spider_web.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 논문용 설정
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'font.family': 'serif'})

# sys_obj (SIR) 임포트 필요
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from systems.sir import Sir


# ==========================================
# ⚙️ 사용자 옵션: Loss Landscape 배경 활성화 여부
SHOW_LANDSCAPE = True  # True로 두면 900번의 ODE를 풀어 등고선을 그립니다. (약 5~10초 소요)
# ==========================================

def load_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_loss_landscape(sys_obj, grid_size=30):
    """Loss Landscape 계산 (Log MSE)"""
    print("Computing Loss Landscape (this may take a few seconds)...")
    beta_range = np.linspace(0.01, 0.45, grid_size)
    gamma_range = np.linspace(0.01, 0.35, grid_size)
    
    B, G = np.meshgrid(beta_range, gamma_range)
    Z = np.zeros_like(B)
    
    theta_true = np.array([0.15, 0.1])
    y0 = [49.0, 1.0, 0.0]
    t_eval = sys_obj.t_points
    
    # 정답 관측치 생성
    sol_true = solve_ivp(sys_obj.ode_func, [0, 110], y0, t_eval=t_eval, args=(theta_true,))
    S_true = sol_true.y[0]
    
    # Grid 순회하며 Loss 계산
    for i in range(grid_size):
        for j in range(grid_size):
            theta_guess = [B[i, j], G[i, j]]
            sol = solve_ivp(sys_obj.ode_func, [0, 110], y0, t_eval=t_eval, args=(theta_guess,))
            if sol.success:
                mse = np.mean((sol.y[0] - S_true)**2)
                Z[i, j] = np.log(mse + 1e-5) # 가독성을 위한 Log Scale
            else:
                Z[i, j] = 10.0 # 발산 시 최대 페널티
                
    return B, G, Z

def plot_spider_web_trajectories(sys_data, radius_key="2.0", show_landscape=False):
    """거미줄 플롯 (Loss Landscape 배경 옵션 포함)"""
    methods = list(sys_data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    
    sir_sys = Sir()
    theta_true = np.array([0.15, 0.1])
    
    # 배경 등고선 미리 계산
    if show_landscape:
        B, G, Z = compute_loss_landscape(sir_sys)
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        if radius_key not in sys_data[method]:
            ax.set_title(f"{method}\n(No Data)", color='gray')
            continue
            
        # 등고선 그리기
        if show_landscape:
            contour = ax.contourf(B, G, Z, levels=30, cmap='viridis_r', alpha=0.5)
            # ax.contour(B, G, Z, levels=15, colors='black', linewidths=0.3, alpha=0.5)
            
        trajectories = sys_data[method][radius_key].get('all_trajectories', [])
        
        color = 'blue' if 'Proposed' in method else 'crimson'
        alpha_line = 0.2 if 'Proposed' in method else 0.4
        
        success_count = 0
        for traj in trajectories:
            if not traj or len(traj) == 0: continue
            
            # 🚨 [버그 수정] 단단한 차원 축소 로직
            traj_np = np.array(traj)
            if traj_np.ndim == 3: 
                traj_np = traj_np.squeeze(1) # [steps, 1, p] -> [steps, p]
            
            if traj_np.ndim != 2 or traj_np.shape[1] < 2: 
                continue
            
            beta_vals = traj_np[:, 0]
            gamma_vals = traj_np[:, 1]
            
            ax.plot(beta_vals, gamma_vals, '-', color=color, alpha=alpha_line, linewidth=1.5)
            ax.scatter(beta_vals[0], gamma_vals[0], c='black', alpha=0.3, s=15, marker='o')
            success_count += 1
            
        # 정답 위치 강조
        ax.scatter(theta_true[0], theta_true[1], c='gold', marker='*', s=400, edgecolors='black', zorder=10, label='True Params')
        
        if success_count == 0:
            ax.text(0.5, 0.5, "Failed/Diverged", ha='center', va='center', transform=ax.transAxes, color='red', fontsize=14, fontweight='bold')
            
        ax.set_title(method, fontweight='bold')
        ax.set_xlabel(r'Infection Rate ($\beta$)')
        ax.set_ylabel(r'Recovery Rate ($\gamma$)')
        
        # 축 범위 고정 (거미줄이 예쁘게 보이도록)
        ax.set_xlim(0.0, 0.45)
        ax.set_ylim(0.0, 0.35)
        
    plt.savefig('fig_spider_web_R2.pdf', format='pdf', bbox_inches='tight')
    print("Saved: fig_spider_web_R2.pdf")
    plt.close()

def plot_ode_with_legends(sys_data, radius_key="2.0"):
    """Baseline 비교가 명확한 수정된 ODE 복원 플롯"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    sir_sys = Sir()
    t_eval_dense = np.linspace(0, 110, 500)
    theta_true = np.array([0.15, 0.1])
    y0 = [49.0, 1.0, 0.0]
    
    sol_true = solve_ivp(sir_sys.ode_func, [0, 110], y0, t_eval=t_eval_dense, args=(theta_true,))
    
    model_colors = {
        "NLLS (LM)": "orange",
        "Adjoint (L-BFGS)": "green",
        "Direct Network (Naive ML)": "purple",
        "Proposed (Iterative)": "blue" 
    }
    
    for i, var_name in enumerate(['S', 'I', 'R']):
        ax = axes[i]
        
        # 1. 정답 궤적
        ax.plot(t_eval_dense, sol_true.y[i], color='black', linestyle='--', linewidth=2.5, label='Ground Truth')
        
        # 2. 관측점
        if i == 0:
            obs_S = solve_ivp(sir_sys.ode_func, [0, 110], y0, t_eval=sir_sys.t_points, args=(theta_true,)).y[0]
            ax.scatter(sir_sys.t_points, obs_S, color='black', s=120, zorder=5, label='Sparse Obs (4 pts)')
        
        # 3. 모델 예측 궤적
        for method, metrics in sys_data.items():
            if radius_key not in metrics: continue
            
            trajectories = metrics[radius_key].get('all_trajectories', [])
            if not trajectories or len(trajectories) == 0: continue
            
            # 🚨 [버그 수정] 첫 번째 궤적의 '마지막(최종 수렴)' 파라미터를 정확히 추출
            first_traj = trajectories[0]
            last_step_params = np.array(first_traj[-1]).flatten()
            
            if len(last_step_params) < 2: continue
            
            I_0 = np.clip(last_step_params[2] if len(last_step_params)>2 else 1.0, 0, 50)
            y_hat_0 = [y0[0], I_0, max(0, 50.0 - y0[0] - I_0)]
            
            sol_hat = solve_ivp(sir_sys.ode_func, [0, 110], y_hat_0, t_eval=t_eval_dense, args=(last_step_params[:2],))
            if not sol_hat.success: continue
            
            color = model_colors.get(method, "gray")
            is_proposed = 'Proposed' in method
            
            ax.plot(t_eval_dense, sol_hat.y[i], 
                    color=color, 
                    alpha=0.9 if is_proposed else 0.5, 
                    linewidth=3.0 if is_proposed else 1.5, 
                    zorder=10 if is_proposed else 3,
                    label=method if i == 0 else "")
                    
        ax.set_title(['Susceptible (S) - Partially Observed', 'Infected (I) - Hidden', 'Recovered (R) - Hidden'][i], fontweight='bold')
        ax.set_ylabel('Population')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    axes[-1].set_xlabel('Time (t)')
    plt.savefig('fig_ode_labeled_R2.pdf', format='pdf', bbox_inches='tight')
    print("Saved: fig_ode_labeled_R2.pdf")

if __name__ == "__main__":
    import glob
    # JSON 파일 경로 로드 (가장 최근 거미줄 데이터)
    json_files = glob.glob(os.path.join(project_root, 'results', 'final_benchmark', 'spider_web*.json'))
    
    if not json_files:
        print("Error: No spider_web JSON found. Please run run_spider_web.py first.")
    else:
        latest_json = max(json_files, key=os.path.getctime)
        print(f"Loading data from: {latest_json}")
        data = load_results(latest_json)
        
        if 'sir' in data:
            plot_spider_web_trajectories(data['sir'], radius_key="2.0", show_landscape=SHOW_LANDSCAPE)
            plot_ode_with_legends(data['sir'], radius_key="2.0")
            print("🎉 All figures generated successfully!")