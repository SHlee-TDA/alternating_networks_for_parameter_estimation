# analysis/compare_interpolation.py
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator, Akima1DInterpolator

# 한글 폰트 설정 (필요시)
plt.rcParams['axes.unicode_minus'] = False

def calculate_roughness(y, dt):
    """곡선의 거칠기(Roughness) 계산: 2차 미분의 제곱합"""
    dy = np.gradient(y, dt)
    ddy = np.gradient(dy, dt)
    return np.sum(ddy**2) * dt

def calculate_negative_violation(y, dt):
    """음수 위반 면적 계산"""
    negative_part = np.minimum(y, 0)
    return -np.sum(negative_part) * dt

def compare_interpolation_methods():
    print("="*60)
    print("📊 SDE Parameter Interpolation Method Comparison")
    print("="*60)

    # 1. 데이터 로드
    param_path = os.path.join('data', 'parameters', 'calibrated_sde_params.json')
    if not os.path.exists(param_path):
        print(f"❌ File not found: {param_path}")
        return

    with open(param_path, 'r') as f:
        data = json.load(f)
    
    t_points = np.array(data['t_points'])
    
    # 분석할 타겟 변수들 (Sigma G, Sigma I, Mu G, Mu I)
    targets = {
        'Sigma_G': np.array(data['sigma_G']),
        'Sigma_I': np.array(data['sigma_I']),
        'Mu_G': np.array(data['mu_G']),
        'Mu_I': np.array(data['mu_I'])
    }
    
    # 비교할 보간법 정의
    methods = {
        'Linear': lambda x, y: interp1d(x, y, kind='linear'),
        'Cubic Spline': lambda x, y: CubicSpline(x, y),
        'PCHIP': lambda x, y: PchipInterpolator(x, y), # Monotonic Cubic
        # 'Akima': lambda x, y: Akima1DInterpolator(x, y) # (옵션) Akima
    }
    
    # 시각화 설정
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # 평가용 고해상도 시간축
    t_fine = np.linspace(t_points[0], t_points[-1], 500)
    dt = t_fine[1] - t_fine[0]
    
    results = {}

    for i, (name, y_points) in enumerate(targets.items()):
        ax = axes[i]
        
        # 산점도 (원본 데이터)
        ax.plot(t_points, y_points, 'ko', markersize=8, label='Calibrated Points', zorder=10)
        
        print(f"\n[Target: {name}]")
        print(f"{'Method':<15} | {'Neg. Area':<10} | {'Roughness':<10} | {'Range [Min, Max]':<20}")
        print("-" * 65)
        
        for method_name, method_func in methods.items():
            f = method_func(t_points, y_points)
            y_fine = f(t_fine)
            
            # 지표 계산
            neg_area = calculate_negative_violation(y_fine, dt)
            roughness = calculate_roughness(y_fine, dt)
            y_min, y_max = np.min(y_fine), np.max(y_fine)
            
            # 출력
            print(f"{method_name:<15} | {neg_area:<10.4f} | {roughness:<10.4f} | [{y_min:.2f}, {y_max:.2f}]")
            
            # 플롯
            linestyle = '--' if method_name == 'Linear' else '-'
            alpha = 0.8 if method_name == 'PCHIP' else 0.5
            width = 2.5 if method_name == 'PCHIP' else 1.5
            
            ax.plot(t_fine, y_fine, linestyle=linestyle, alpha=alpha, linewidth=width, label=method_name)
            
            # Sigma의 경우 0선 표시 (음수 경고)
            if 'Sigma' in name:
                ax.axhline(0, color='r', linewidth=1, linestyle=':', alpha=0.5)
                if neg_area > 0:
                    ax.fill_between(t_fine, y_fine, 0, where=(y_fine<0), color='red', alpha=0.2, label='Negative Violation')

        ax.set_title(f"Interpolation of {name}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join('analysis', 'interpolation_comparison.png')
    plt.savefig(save_path)
    print(f"\n✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    compare_interpolation_methods()