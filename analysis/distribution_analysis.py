# analysis/analyze_distributions.py
"""
데이터 분포 분석 및 샘플링 파라미터 추출 스크립트
================================================

실제 데이터의 G0, I0, Si, Sigma 분포를 Log-Normal로 피팅합니다.
결과는 'data/parameters/distribution_params.json'에 저장되어 DataGenerator에서 사용됩니다.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from pathlib import Path

# 프로젝트 루트 경로 설정 (상위 디렉토리 모듈 import용)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config import Config
from src.data_loader import RealOGTTDataLoader

def remove_outliers(data, lower=1, upper=99):
    """1% ~ 99% 사이의 데이터만 남김"""
    if len(data) == 0: return data
    lb = np.percentile(data, lower)
    ub = np.percentile(data, upper)
    return data[(data >= lb) & (data <= ub)]

def fit_lognorm_robust(data, name, ax=None):
    """
    데이터에 Log-Normal 분포를 피팅하고 시각화합니다. (Robust ver.)
    """
    # 0 이하의 값은 lognorm 피팅 불가하므로 필터링
    data = data[data > 0]
    # 이상치 제거
    clean_data = remove_outliers(data, 1, 99)

    # s(shape), loc, scale 파라미터 추정
    s, loc, scale = stats.lognorm.fit(clean_data)
    
    print(f"[{name}] Log-Normal Fit (N={len(clean_data)}) -> s={s:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    
    if ax:
        # 히스토그램
        ax.hist(clean_data, bins=30, density=True, alpha=0.6, color='g', label='Real Data (Clean)')
        
        # PDF 라인
        xmin, xmax = clean_data.min(), clean_data.max()
        x = np.linspace(xmin, xmax, 100)
        p = stats.lognorm.pdf(x, s, loc, scale)
        ax.plot(x, p, 'k', linewidth=2, label='LogNorm Fit')
        ax.set_title(f"{name}\n(s={s:.2f}, scale={scale:.2f})")
        ax.legend()
    
    return {'s': s, 'loc': loc, 'scale': scale}

def analyze_distributions():
    print("=== Analyzing Distributions for Data-Driven Sampling ===")
    
    # 1. 데이터 로드 
    config = Config()
    # 데이터 파일 경로 (프로젝트 루트 기준)
    data_path = project_root / 'data' / 'clean_sumner_n_612.xlsx'
    if not data_path.exists():
        # 루트에 있을 경우 대비 (호환성)
        data_path = project_root / 'clean_sumner_n_612.xlsx'

    loader = RealOGTTDataLoader(str(data_path), config)
    
    # 데이터 로드 (중복 호출 제거)
    # X_obs: (N, T, D)
    X_obs, Y_hid, P_true, _ = loader.load_data()
    
    # Raw Data Extraction
    # 차원 축소: (N, 5, 1) -> (N, 5) -> (N,) (초기값은 0번째 시점)
    g0 = X_obs[:, 0, 0]
    i0 = Y_hid[:, 0, 0]
    si = P_true[:, 0]
    sigma = P_true[:, 1]
    
    # --- 1. Distribution Fitting & Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Robust Log-Normal Fitting (Outliers Removed)", fontsize=16)
    
    dist_params = {}
    dist_params['G0'] = fit_lognorm_robust(g0, "G0", axes[0, 0])
    dist_params['I0'] = fit_lognorm_robust(i0, "I0", axes[0, 1])
    dist_params['si'] = fit_lognorm_robust(si, "Si", axes[1, 0])
    dist_params['sigma'] = fit_lognorm_robust(sigma, "Sigma", axes[1, 1])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 이미지 저장 (analysis 폴더 내)
    plot_save_path = current_dir / 'distribution_analysis.png'
    plt.savefig(plot_save_path)
    print(f"Saved distribution plot to '{plot_save_path}'")
    
    # --- 2. Correlation Analysis ---
    # 데이터프레임 생성
    df_corr = pd.DataFrame({
        'G0': g0, 'I0': i0, 'Si': si, 'Sigma': sigma
    })
    
    # 상관계수 행렬 계산
    corr_matrix = df_corr.corr()
    print("\n[Correlation Matrix]")
    print(corr_matrix)
    
    # 히트맵 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Parameter Correlation Matrix")
    
    corr_save_path = current_dir / 'parameter_correlation.png'
    plt.savefig(corr_save_path)
    print(f"Saved correlation heatmap to '{corr_save_path}'")

    # --- 3. 결과 저장 (JSON) ---
    # 저장 경로: data/parameters/
    save_dir = project_root / 'data' / 'parameters'
    os.makedirs(save_dir, exist_ok=True)
    json_save_path = save_dir / 'distribution_params.json'

    with open(json_save_path, 'w') as f:
        json.dump(dist_params, f, indent=4)
    print(f"Saved robust parameters to '{json_save_path}'")

if __name__ == "__main__":
    analyze_distributions()