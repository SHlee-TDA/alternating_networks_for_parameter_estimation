import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from systems.lotka_volterra import LotkaVolterra

def V_manifold(x, y, alpha, beta, delta, gamma):
    """Equation (4): V(x,y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)"""
    return delta * x - gamma * np.log(x) + beta * y - alpha * np.log(y)

def generate_lv_figure():
    lv_system = LotkaVolterra()
    
    params = [0.8, 0.6, 0.4, 0.8]  
    y0 = [10.0, 3.0]               
    
    t_span = lv_system.t_span      
    t_points = lv_system.t_points  
    t_continuous = np.linspace(t_span[0], t_span[1], 1000)
    
    sol_continuous = solve_ivp(
        fun=lambda t, y: LotkaVolterra.ode_func(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_continuous,
        method='RK45'
    )
    x_c, y_c = sol_continuous.y
    
    sol_sparse = solve_ivp(
        fun=lambda t, y: LotkaVolterra.ode_func(t, y, params),
        t_span=[t_points[0], t_points[-1]],
        y0=y0,
        t_eval=t_points,
        method='RK45'
    )
    x_s, y_s = sol_sparse.y
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # ==========================================
    # Left Panel: Time-domain plot
    # ==========================================
    ax1 = axes[0]
    ax1.plot(sol_continuous.t, x_c, 'b-.', label=r'$x(t)$', linewidth=1.5)
    ax1.plot(sol_continuous.t, y_c, 'r-.', label=r'$y(t)$', linewidth=1.5)
    
    ax1.plot(sol_sparse.t, x_s, 'bo', markersize=6, zorder=5)
    
    ax1.set_xlabel('Time (t)', fontsize=11)
    ax1.set_ylabel('Population', fontsize=11)
    ax1.set_title('Time-domain Trajectories', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(t_span[0]-1, t_span[1]+1)
    
    # ==========================================
    # Right Panel: Phase portrait & V(x,y)
    # ==========================================
    ax2 = axes[1]
    
    # 궤적이 중앙에 오도록 동적 패딩(30%) 적용
    pad_x = (x_c.max() - x_c.min()) * 0.3
    pad_y = (y_c.max() - y_c.min()) * 0.3
    
    x_min, x_max = max(0.1, x_c.min() - pad_x), x_c.max()
    y_min, y_max = max(0.1, y_c.min() - pad_y), y_c.max()
    
    X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = V_manifold(X_grid, Y_grid, *params)
    
    # 등고선 그리기 및 라벨(V의 값) 추가
    levels = np.linspace(Z.min(), Z.max(), 12)
    contours = ax2.contour(X_grid, Y_grid, Z, levels=levels, colors='gray', alpha=0.4, linewidths=0.8)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='V=%.1f')
    
    # 위상 공간에서의 닫힌 궤도 및 희소 관측점
    ax2.plot(x_c, y_c, 'k-', linewidth=1.5, label='$\mathbf{x}(t)$')
    ax2.plot(x_s, y_s, 'bo', markersize=6, zorder=5)
    
    # sparse sample 위치에서 정확한 기울기 벡터 계산
    dx_s = np.zeros_like(x_s)
    dy_s = np.zeros_like(y_s)
    for i in range(len(x_s)):
        # ODE 모델의 정의를 이용하여 해당 좌표의 미분값을 구함
        dx_s[i], dy_s[i] = LotkaVolterra.ode_func(t_points[i], [x_s[i], y_s[i]], params)
        
    norm_s = np.hypot(dx_s, dy_s)
    dx_norm_s, dy_norm_s = dx_s / norm_s, dy_s / norm_s  # 벡터 정규화
    
    # 희소 관측점(x_s, y_s)을 시점으로 화살표 렌더링
    ax2.quiver(x_s, y_s, dx_norm_s, dy_norm_s,
               color='black', scale=20, width=0.006, headwidth=4, headlength=5, zorder=6)
    
    ax2.set_xlabel(r'$x(t)$', fontsize=11)
    ax2.set_ylabel(r'$y(t)$', fontsize=11)
    ax2.set_title('Phase Portrait', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-0.3, x_max+0.1)
    ax2.set_ylim(-0.3, y_max+0.1)

    plt.tight_layout()
    
    save_path = 'lv_trajectories.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    generate_lv_figure()