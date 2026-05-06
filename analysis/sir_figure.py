import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# analysis 디렉토리에서 실행될 때 상위 디렉토리 모듈을 임포트하기 위한 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from systems.sir import Sir

def generate_sir_figure():
    sir_system = Sir()
    
    # 시스템에 정의된 설정값 추출
    y0 = sir_system.sample_initial_conditions(None)  # [49.0, 1.0, 0.0]
    t_span = sir_system.t_span  # [0, 110]
    t_points = sir_system.t_points  # [0, 20, 40, 60, 80, 100]
    
    t_continuous = np.linspace(t_span[0], t_span[1], 1000)
    #{'beta': [0.05, 0.15], 'gamma': [0.05, 0.35]}
    params_r0_gt_1 = [0.15, 0.05]  # R0 ~ 3.0
    params_r0_le_1 = [0.05, 0.35]  # R0 ~ 0.14
    
    scenarios = [
        {'params': params_r0_gt_1, 'title': r'$\mathcal{R}_0 > 1$'},
        {'params': params_r0_le_1, 'title': r'$\mathcal{R}_0 < 1$'}
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    
    for ax, scenario in zip(axes, scenarios):
        params = scenario['params']
        
        sol_continuous = solve_ivp(
            fun=lambda t, y: Sir.ode_func(t, y, params),
            t_span=t_span,
            y0=y0,
            t_eval=t_continuous,
            method='RK45'
        )
        S_c, I_c, R_c = sol_continuous.y
        
        sol_sparse = solve_ivp(
            fun=lambda t, y: Sir.ode_func(t, y, params),
            t_span=[t_points[0], t_points[-1]],
            y0=y0,
            t_eval=t_points,
            method='RK45'
        )
        S_s = sol_sparse.y[0]
        
        ax.plot(sol_continuous.t, S_c, 'b-.', label='S(t)', linewidth=1.5)
        ax.plot(sol_continuous.t, I_c, 'r-.', label='I(t)', linewidth=1.5)
        ax.plot(sol_continuous.t, R_c, 'g-.', label='R(t)', linewidth=1.5)
        
        ax.plot(sol_sparse.t, S_s, 'bo', markersize=5)
        
        ax.set_xlabel('Time (t)', fontsize=11)
        ax.set_ylabel('Population', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 범례 대신 서브플롯 타이틀로 위상 조건 명시
        ax.set_title(scenario['title'], fontsize=12)
        
        ax.set_ylim(-2, 52)
        ax.set_xlim(-2, t_span[1] + 2)

    # 그래프 외부에 공통 범례 배치 (상단 중앙, 4열 구조)
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=10)

    plt.tight_layout()
    
    save_path = 'sir_trajectories.pdf'
    # bbox_inches='tight'가 외부 범례가 잘리지 않도록 보호합니다.
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    generate_sir_figure()