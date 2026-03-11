import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, lagrange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from systems.sir import Sir
from systems.lotka_volterra import LotkaVolterra
from systems.ogtt_simul import OgttSimul

def plot_tangent_ablation(sys_class, params, params_dict, noise_std, dt_tangent, ax, title, ylabel):
    system = sys_class()
    y0 = system.sample_initial_conditions(params_dict)
    
    t_span = system.t_span
    t_sparse = system.t_points
    t_dense = np.linspace(t_span[0], t_span[1], 500)
    obs_idx = system.observed_var_idx
    
    # Ground Truth 연속 궤적
    sol_dense = solve_ivp(
        fun=lambda t, y: system.ode_func(t, y, params),
        t_span=t_span, y0=y0, t_eval=t_dense, method='BDF'
    )
    y_dense = sol_dense.y[obs_idx]
    
    # 희소 관측 데이터 (Clean)
    sol_sparse = solve_ivp(
        fun=lambda t, y: system.ode_func(t, y, params),
        t_span=[t_sparse[0], t_sparse[-1]], y0=y0, t_eval=t_sparse, method='BDF'
    )
    y_sparse_clean = sol_sparse.y[obs_idx]
    
    # 관측 노이즈 추가
    np.random.seed(42)
    y_sparse_noisy = y_sparse_clean + np.random.normal(0, noise_std, size=len(t_sparse))
    
    # --- 도함수(기울기) 계산 ---
    # 1. Ground Truth Vector Field
    m_gt = np.array([system.ode_func(t, y, params)[obs_idx] for t, y in zip(sol_sparse.t, sol_sparse.y.T)])
    
    # 2. Finite Differences (FD)
    m_fd = np.gradient(y_sparse_noisy, t_sparse)
    
    # 3. Lagrange Polynomial
    poly_lagrange = lagrange(t_sparse, y_sparse_noisy)
    m_lagrange = poly_lagrange.deriv()(t_sparse)
    
    # 4. Cubic Spline (Ours)
    cs = CubicSpline(t_sparse, y_sparse_noisy, bc_type='natural')
    m_spline = cs.derivative()(t_sparse)
    
    # --- Plotting ---
    ax.plot(t_dense, y_dense, 'k-', linewidth=1.5, alpha=0.3, label='True Trajectory')
    
    for i, t_val in enumerate(t_sparse):
        y_anchor = y_sparse_noisy[i]
        t_line = np.array([t_val - dt_tangent, t_val + dt_tangent])
        
        # 중간 지점의 마커에서만 범례 추가
        legend_idx = len(t_sparse) // 2
        lbl_gt = 'GT Vector Field' if i == legend_idx else ""
        lbl_fd = 'Finite Diff.' if i == legend_idx else ""
        lbl_lp = 'Lagrange Poly.' if i == legend_idx else ""
        lbl_cs = 'Cubic Spline (Ours)' if i == legend_idx else ""
        
        # 접선 그리기: y = m*(t - t0) + y0
        ax.plot(t_line, m_gt[i]*(t_line-t_val)+y_anchor, color='black', linestyle='-', linewidth=2.5, label=lbl_gt, zorder=4)
        ax.plot(t_line, m_fd[i]*(t_line-t_val)+y_anchor, color='green', linestyle='--', linewidth=1.5, label=lbl_fd, zorder=4)
        ax.plot(t_line, m_lagrange[i]*(t_line-t_val)+y_anchor, color='magenta', linestyle=':', linewidth=2, label=lbl_lp, zorder=4)
        ax.plot(t_line, m_spline[i]*(t_line-t_val)+y_anchor, color='blue', linestyle='-.', linewidth=2, label=lbl_cs, zorder=4)
        
        # 희소 관측점 마커
        if i == 0:
            ax.plot(t_val, y_anchor, 'ro', markersize=6, label='Noisy Sparse Obs.', zorder=5)
        else:
            ax.plot(t_val, y_anchor, 'ro', markersize=6, zorder=5)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time (t)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Lagrange 폭주를 숨기기 위한 y축 강제 제한
    y_range = y_dense.max() - y_dense.min()
    ax.set_ylim([max(0, y_dense.min() - y_range * 0.4), y_dense.max() + y_range * 0.6])
    ax.set_xlim([t_span[0] - dt_tangent*1.2, t_span[1] + dt_tangent*1.2])

def generate_all_tangents_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. SIR Model (Monotonic/Bell-shape)
    # R0 > 1 인 명확한 궤적을 위해 beta=0.12, gamma=0.05 사용
    plot_tangent_ablation(
        sys_class=Sir, params=[0.12, 0.05], params_dict=None, 
        noise_std=1.0, dt_tangent=7.0, 
        ax=axes[0], title='(a) SIR Model ($S$)', ylabel='Susceptible $S(t)$'
    )
    
    # 2. Lotka-Volterra Model (Periodic)
    plot_tangent_ablation(
        sys_class=LotkaVolterra, params=[0.8, 0.6, 0.4, 0.8], params_dict=None, 
        noise_std=0.5, dt_tangent=1.0, 
        ax=axes[1], title='(b) Lotka-Volterra Model ($x$)', ylabel='Prey $x(t)$'
    )
    
    # 3. OGTT Model (Non-linear Delay)
    plot_tangent_ablation(
        sys_class=OgttSimul, params=[0.5, 0.5], params_dict={'si': 0.5, 'sigma': 0.5}, 
        noise_std=3.0, dt_tangent=8.0, 
        ax=axes[2], title='(c) OGTT Model ($G$)', ylabel='Glucose (mg/dL)'
    )
    
    # 통합 범례 (오른쪽 패널 기준 또는 Figure 전체 기준)
    axes[2].legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.0, 1.0))
    # 다른 패널에서는 중복 범례 제거
    axes[0].legend().set_visible(False)
    axes[1].legend().set_visible(False)

    plt.tight_layout()
    save_path = 'all_systems_tangent_ablation.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, lagrange

# 상위 폴더 모듈 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from systems.sir import Sir
from systems.lotka_volterra import LotkaVolterra
from systems.ogtt_simul import OgttSimul

# 다항식 보간 경고 무시 (Lagrange 폭주 현상 자체를 관찰할 목적이므로)
warnings.simplefilter('ignore', np.RankWarning)

def compute_mse_distribution(sys_class, params, params_dict, noise_std, n_samples=100):
    system = sys_class()
    t_span = system.t_span
    t_sparse = system.t_points
    obs_idx = system.observed_var_idx
    
    mse_fd = []
    mse_lagrange = []
    mse_spline = []
    
    np.random.seed(42) # 재현성을 위한 시드 고정
    
    successful_samples = 0
    while successful_samples < n_samples:
        y0 = system.sample_initial_conditions(params_dict)
        
        # Ground Truth 시뮬레이션
        sol_sparse = solve_ivp(
            fun=lambda t, y: system.ode_func(t, y, params),
            t_span=[t_sparse[0], t_sparse[-1]], y0=y0, t_eval=t_sparse, method='BDF'
        )
        
        # 수치적 불안정으로 시뮬레이션 실패 시 건너뜀
        if not sol_sparse.success:
            continue
            
        y_sparse_clean = sol_sparse.y[obs_idx]
        y_sparse_noisy = y_sparse_clean + np.random.normal(0, noise_std, size=len(t_sparse))
        
        # 1. Ground Truth Vector Field
        m_gt = np.array([system.ode_func(t, y, params)[obs_idx] for t, y in zip(sol_sparse.t, sol_sparse.y.T)])
        
        # 2. Finite Differences (FD)
        m_fd = np.gradient(y_sparse_noisy, t_sparse)
        
        # 3. Lagrange Polynomial
        poly_lagrange = lagrange(t_sparse, y_sparse_noisy)
        m_lagrange = poly_lagrange.deriv()(t_sparse)
        
        # 4. Cubic Spline (Ours)
        cs = CubicSpline(t_sparse, y_sparse_noisy, bc_type='natural')
        m_spline = cs.derivative()(t_sparse)
        
        # 오차(MSE) 축적
        mse_fd.append(np.mean((m_fd - m_gt)**2))
        mse_lagrange.append(np.mean((m_lagrange - m_gt)**2))
        mse_spline.append(np.mean((m_spline - m_gt)**2))
        
        successful_samples += 1
        
    return np.array(mse_fd), np.array(mse_lagrange), np.array(mse_spline)

def generate_robust_ablation_figure():
    print("Simulating 100 trajectories per system for robust evaluation...")
    
    systems_info = [
        ('SIR', Sir, [0.12, 0.05], None, 1.0),
        ('Lotka-Volterra', LotkaVolterra, [0.8, 0.6, 0.4, 0.8], None, 0.5),
        ('OGTT', OgttSimul, [1.0, 1.0], {'si': 1.0, 'sigma': 1.0}, 3.0)
    ]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # 박스 플롯을 위한 데이터 구조화
    plot_data = []
    positions = []
    labels = []
    colors = ['mediumseagreen', 'orchid', 'royalblue']
    
    base_pos = np.arange(len(systems_info)) * 4
    
    for i, (name, sys_class, params, p_dict, noise) in enumerate(systems_info):
        dist_fd, dist_lagrange, dist_spline = compute_mse_distribution(sys_class, params, p_dict, noise, n_samples=100)
        
        plot_data.extend([dist_fd, dist_lagrange, dist_spline])
        positions.extend([base_pos[i] - 1, base_pos[i], base_pos[i] + 1])
        labels.append(name)
        
        print(f"[{name}] Median MSE -> FD: {np.median(dist_fd):.4f}, Lagr: {np.median(dist_lagrange):.4f}, Spline: {np.median(dist_spline):.4f}")

    # 박스 플롯 렌더링
    bp = ax.boxplot(plot_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor="white", color="black"),
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))

    # 박스 색상 칠하기 (FD, Lagrange, Spline 반복)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % 3])
        patch.set_alpha(0.8)

    # 범례 생성을 위한 더미 플롯
    ax.plot([], [], color=colors[0], label='Finite Differences', linewidth=6)
    ax.plot([], [], color=colors[1], label='Lagrange Polynomial', linewidth=6)
    ax.plot([], [], color=colors[2], label='Cubic Spline (Ours)', linewidth=6)
    
    ax.set_ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    ax.set_title('Robust Evaluation of Derivative Approximations ($N=100$ trajectories)', fontsize=14)
    ax.set_xticks(base_pos)
    ax.set_xticklabels(labels, fontsize=12)
    
    # y축 로그 스케일 적용
    ax.set_yscale('log')
    
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = 'derivative_mse_distribution.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")

if __name__ == "__main__":
    generate_all_tangents_figure()
    generate_robust_ablation_figure()