import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# 프로젝트 모듈 임포트
from config import Config
from systems.ogtt_simul import OgttSimul
from src.data_loader import DataGenerator




def analyze_simulation_data():
    # 1. 설정 및 데이터 로드
    print("--- [Analysis] Loading Simulation Data ---")
    config = Config()
    config.USE_SDE = True
    system = OgttSimul()
    generator = DataGenerator(system, config)
    
    # generate_data()는 파일이 있으면 로드하고, 없으면 생성합니다.
    # 여기서는 저장된 데이터의 상태를 보는 것이 목적이므로 기존 파일을 로드할 것입니다.
    try:
        _, _, params_data, _ = generator.generate_data()
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # params_data shape: (N, 2) -> [si, sigma]
    print(f"\n✅ Data Loaded. Shape: {params_data.shape}")
    
    si_values = params_data[:, 0]
    sigma_values = params_data[:, 1]
    
    # 2. 통계적 분석 (텍스트 출력)
    print("\n" + "="*40)
    print(" 📊 Parameter Distribution Statistics")
    print("="*40)
    
    def print_stats(name, data):
        print(f"\n[ Parameter: {name} ]")
        print(f"  - Mean   : {np.mean(data):.6f}")
        print(f"  - Median : {np.median(data):.6f}")
        print(f"  - Std    : {np.std(data):.6f}")
        print(f"  - Min    : {np.min(data):.6f}")
        print(f"  - Max    : {np.max(data):.6f}")
        print(f"  - Zeros  : {np.sum(data == 0)} count ({(np.sum(data==0)/len(data))*100:.2f}%)")
        print(f"  - Negatives : {np.sum(data < 0)} count")
        
        # 분위수 확인 (혹시 특정 값 이하에 몰려있는지 확인)
        quantiles = np.percentile(data, [1, 5, 25, 50, 75, 95, 99])
        print(f"  - Quantiles:")
        print(f"    1% : {quantiles[0]:.6f}")
        print(f"    5% : {quantiles[1]:.6f}")
        print(f"    25%: {quantiles[2]:.6f}")
        print(f"    50%: {quantiles[3]:.6f}")
        print(f"    75%: {quantiles[4]:.6f}")
        print(f"    95%: {quantiles[5]:.6f}")
        print(f"    99%: {quantiles[6]:.6f}")

    print_stats("si (Sensitivity)", si_values)
    print_stats("sigma (Diffusion)", sigma_values)
    
    # 3. 시각화 (이미지 저장)
    print("\n--- [Analysis] Plotting Distributions ---")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (1) SI Histogram
    sns.histplot(si_values, bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of si\n(Mean: {np.mean(si_values):.2f})')
    axes[0].set_xlabel('si value')
    axes[0].axvline(np.mean(si_values), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    # (2) Sigma Histogram
    sns.histplot(sigma_values, bins=50, kde=True, ax=axes[1], color='salmon')
    axes[1].set_title(f'Distribution of sigma\n(Mean: {np.mean(sigma_values):.2f})')
    axes[1].set_xlabel('sigma value')
    axes[1].axvline(np.mean(sigma_values), color='blue', linestyle='--', label='Mean')
    axes[1].legend()
    
    # (3) Scatter Plot (Correlation)
    sns.scatterplot(x=si_values, y=sigma_values, ax=axes[2], alpha=0.3, s=10, color='purple')
    axes[2].set_title('Scatter Plot: si vs sigma')
    axes[2].set_xlabel('si')
    axes[2].set_ylabel('sigma')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # 0에 몰려있는지 확인하기 위해 로그 스케일 검토 (옵션)
    # axes[0].set_xscale('log')
    # axes[1].set_xscale('log')

    plt.tight_layout()
    save_path = os.path.join('analysis', 'simulation_data_distribution.png')
    plt.savefig(save_path)
    print(f"✅ Distribution plot saved to '{save_path}'")
    
    # 4. 결론 도출 (자동 진단)
    print("\n" + "="*40)
    print(" 🩺 Diagnostic Summary")
    print("="*40)
    
    issues_found = []
    if np.max(si_values) < 1e-3:
        issues_found.append("⚠️ 'si' values are extremely small (near zero). Check units or generation logic.")
    if np.max(sigma_values) < 1e-3:
        issues_found.append("⚠️ 'sigma' values are extremely small (near zero). Check scaling.")
    if np.sum(si_values == 0) > 0 or np.sum(sigma_values == 0) > 0:
        issues_found.append("⚠️ Zeros found in parameters. Log-Normal sampling might be failing.")
    
    if not issues_found:
        print("✅ Data distribution looks statistically normal (ranges are reasonable).")
        print("   -> If prediction is still 0, check the Model Initialization or Gradient Flow.")
    else:
        for issue in issues_found:
            print(issue)

if __name__ == "__main__":
    analyze_simulation_data()