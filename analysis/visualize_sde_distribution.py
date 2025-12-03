# analysis/visualize_sde_distribution.py
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import norm

# 한글 폰트 설정 (필요시 시스템에 맞는 폰트로 변경)
plt.rcParams['axes.unicode_minus'] = False

def visualize_sde_distribution():
    print("="*60)
    print("🎨 SDE Parameter Distribution Visualization (Gaussian Process)")
    print("="*60)

    # 1. 데이터 로드
    param_path = os.path.join('data', 'parameters', 'calibrated_sde_params.json')
    if not os.path.exists(param_path):
        print(f"❌ File not found: {param_path}")
        return

    with open(param_path, 'r') as f:
        data = json.load(f)
    
    t_points = np.array(data['t_points'])
    
    # 2. 보간 (PCHIP 사용 - 안정성 확보)
    # 고해상도 시간축 생성
    t_fine = np.linspace(t_points[0], t_points[-1], 200)
    
    # 보간 함수 생성
    interp_mu_G = CubicSpline(t_points, data['mu_G'])
    interp_sigma_G = CubicSpline(t_points, data['sigma_G'])
    
    interp_mu_I = CubicSpline(t_points, data['mu_I'])
    interp_sigma_I = CubicSpline(t_points, data['sigma_I'])
    
    # 값 계산
    mu_G_fine = interp_mu_G(t_fine)
    sigma_G_fine = np.maximum(0, interp_sigma_G(t_fine)) # 음수 방지
    
    mu_I_fine = interp_mu_I(t_fine)
    sigma_I_fine = np.maximum(0, interp_sigma_I(t_fine))

    # 3. 시각화 (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    
    # --- Function: Plot Gaussian Tube ---
    def plot_gaussian_tube(ax, t, mu, sigma, title, color):
        # Mean Line
        ax.plot(t, mu, color=color, linewidth=2.5, label=r'Mean Drift ($\mu(t)$)')
        
        # Confidence Intervals
        ax.fill_between(t, mu - sigma, mu + sigma, color=color, alpha=0.3, label=r'$\pm 1\sigma$ (68%)')
        ax.fill_between(t, mu - 2*sigma, mu + 2*sigma, color=color, alpha=0.1, label=r'$\pm 2\sigma$ (95%)')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Correction Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    # --- Function: Plot Density Heatmap ---
    def plot_density_heatmap(ax, t, mu, sigma, title, cmap):
        # Y축 그리드 생성 (값의 범위에 따라 동적 조정)
        y_min = np.min(mu - 3*sigma)
        y_max = np.max(mu + 3*sigma)
        y_grid = np.linspace(y_min, y_max, 200)
        
        # Meshgrid 생성
        T, Y = np.meshgrid(t, y_grid)
        
        # 각 시점 t에 대해 Gaussian PDF 계산 (Broadcasting)
        # Mu와 Sigma를 (1, N) 형태로 맞춰줌
        Mu = mu.reshape(1, -1)
        Sigma = sigma.reshape(1, -1)
        
        # PDF 계산: N(y | mu(t), sigma(t))
        # sigma가 0인 경우 대비하여 엡실론 추가
        Z = (1 / (np.sqrt(2 * np.pi) * (Sigma + 1e-6))) * np.exp(-0.5 * ((Y - Mu) / (Sigma + 1e-6))**2)
        
        # Contourf Plot
        c = ax.contourf(T, Y, Z, levels=50, cmap=cmap, alpha=0.9)
        
        # Mean line overlay
        ax.plot(t, mu, 'k--', linewidth=1.5, alpha=0.7, label='Mean')
        
        fig.colorbar(c, ax=ax, label='Probability Density')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Correction Value')

    # Plotting
    # (1) Glucose - Tube
    plot_gaussian_tube(axes[0, 0], t_fine, mu_G_fine, sigma_G_fine, 
                       'Glucose Correction: Uncertainty Tube', 'tab:blue')
    
    # (2) Glucose - Heatmap
    plot_density_heatmap(axes[1, 0], t_fine, mu_G_fine, sigma_G_fine, 
                         'Glucose Correction: Density Evolution', 'Blues')
    
    # (3) Insulin - Tube
    plot_gaussian_tube(axes[0, 1], t_fine, mu_I_fine, sigma_I_fine, 
                       'Insulin Correction: Uncertainty Tube', 'tab:orange')
    
    # (4) Insulin - Heatmap
    plot_density_heatmap(axes[1, 1], t_fine, mu_I_fine, sigma_I_fine, 
                         'Insulin Correction: Density Evolution', 'Oranges')
    
    # 저장
    save_path = os.path.join('analysis', 'sde_distribution_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_sde_distribution()