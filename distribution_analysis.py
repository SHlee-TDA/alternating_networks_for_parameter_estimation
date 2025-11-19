# distribution_analysis.py
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import os
from config import Config
from data_loader import RealOGTTDataLoader

def fit_lognorm(data, name, ax=None):
    """
    데이터에 Log-Normal 분포를 피팅하고 시각화합니다.
    """
    # 0 이하의 값은 lognorm 피팅 불가하므로 필터링 (데이터 품질 이슈 대비)
    data = data[data > 0]
    
    # s(shape), loc, scale 파라미터 추정
    s, loc, scale = stats.lognorm.fit(data)
    
    print(f"[{name}] Log-Normal Fit -> s={s:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    
    if ax:
        # 히스토그램
        ax.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Real Data')
        
        # PDF 라인
        xmin, xmax = data.min(), data.max()
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
    data_path = 'data/clean_sumner_n_612.xlsx' # 실제 데이터 파일 경로
    loader = RealOGTTDataLoader(data_path, config)
    
    # 모든 데이터를 한 번에 언패킹
    glucose_data, insulin_data, params_data, _ = loader.load_data()
    
    # 데이터 추출
    g0_data = glucose_data[:, 0, 0] # t=0 Glucose
    i0_data = insulin_data[:, 0, 0] # t=0 Insulin
    si_data = params_data[:, 0]     # Si
    sigma_data = params_data[:, 1]  # Sigma
    
    # 2. 분포 피팅 및 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Data-Driven Log-Normal Distribution Fitting", fontsize=16)
    
    dist_params = {}
    dist_params['G0'] = fit_lognorm(g0_data, "Initial Glucose (G0)", axes[0, 0])
    dist_params['I0'] = fit_lognorm(i0_data, "Initial Insulin (I0)", axes[0, 1])
    dist_params['si'] = fit_lognorm(si_data, "Parameter Si", axes[1, 0])
    dist_params['sigma'] = fit_lognorm(sigma_data, "Parameter Sigma", axes[1, 1])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('distribution_analysis.png')
    print("Saved plot to 'distribution_analysis.png'")
    
    # 3. 파라미터 저장
    with open('distribution_params.json', 'w') as f:
        json.dump(dist_params, f, indent=4)
    print("Saved distribution parameters to 'distribution_params.json'")

if __name__ == "__main__":
    analyze_distributions()