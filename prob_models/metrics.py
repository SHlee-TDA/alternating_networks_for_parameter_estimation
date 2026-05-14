"""
This script contains evaluation metrics for probabilistic models
"""

import numpy as np
from scipy.stats import pearsonr, multivariate_normal

def calculate_point_metrics(y_true, y_samples):
    """
    사후 분포의 평균값을 이용해 결정론적 지표(RMSE, Pearson R)를 계산합니다.
    기존 Baseline 모델과의 '사과 대 사과' 비교를 위해 사용됩니다.
    
    Args:
        y_true: [batch_size, feature_dim]
        y_samples: [num_samples, batch_size, feature_dim]
    Returns:
        rmse, pearson_r
    """
    # 샘플들의 평균을 점 추정치로 사용
    y_mean = np.mean(y_samples, axis=0)
    
    # 1. RMSE 계산
    rmse = np.sqrt(np.mean((y_true - y_mean)**2, axis=0))
    
    # 2. Pearson R 계산
    pearson_r = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        # 분산이 너무 작을 경우(모델 붕괴)의 수치적 오류 방지
        if np.var(y_true[:, i]) < 1e-12 or np.var(y_mean[:, i]) < 1e-12:
            pearson_r[i] = 0.0
        else:
            r, _ = pearsonr(y_true[:, i], y_mean[:, i])
            pearson_r[i] = r if np.isfinite(r) else 0.0
            
    return rmse, pearson_r

def calculate_prediction_interval(y_true, y_samples, alpha=0.05):
    """
    불확실성 및 신뢰도 지표인 PICP (Coverage)와 MPIW (Width)를 계산합니다.
    
    Args:
        alpha: 0.05 이면 95% 신뢰 구간 (2.5% ~ 97.5% 분위수)
    Returns:
        picp (포함 비율), mpiw (구간 평균 너비)
    """
    lower_bound = np.quantile(y_samples, q=alpha/2, axis=0)
    upper_bound = np.quantile(y_samples, q=1 - alpha/2, axis=0)
    
    # 정답이 신뢰 구간 안에 포함되었는지 검사 (Boolean mask)
    is_captured = (y_true >= lower_bound) & (y_true <= upper_bound)
    
    # PICP: 포함된 비율 (1.0에 가까울수록, 최소 1-alpha 이상일수록 좋음)
    picp = np.mean(is_captured, axis=0)
    
    # MPIW: 구간의 평균 너비 (정답을 놓치지 않는 선에서 좁을수록 좋음)
    mpiw = np.mean(upper_bound - lower_bound, axis=0)
    
    return picp, mpiw

def calculate_crps(y_true, y_samples):
    """
    에너지 스코어(Energy Score) 기반의 1D CRPS 근사치를 계산합니다.
    분포의 형태적 정확도와 불확실성을 동시에 평가하는 핵심 지표입니다.
    
    Returns:
        crps (낮을수록 좋음)
    """
    num_samples = y_samples.shape[0]
    
    # 1. 정답과 샘플들 간의 오차 평균: E|X - y_true|
    diff_true = np.mean(np.abs(y_samples - y_true[np.newaxis, :, :]), axis=0)
    
    # 2. 샘플들 간의 내부 오차 평균: E|X - X'|
    diff_samples = np.zeros_like(y_true)
    for i in range(num_samples):
        diff_samples += np.sum(np.abs(y_samples - y_samples[i:i+1]), axis=0)
    diff_samples = diff_samples / (num_samples ** 2)
    
    crps = diff_true - 0.5 * diff_samples
    
    return np.mean(crps, axis=0) # batch 방향 평균

def calculate_parametric_nll(theta_true, theta_samples):
    """
    (선택적 사용) 가우시안 분포 가정을 통한 NLL (Negative Log-Likelihood) 계산.
    이전에 문제가 되었던 KDE 방식 대신 안정적인 다변량 정규분포 근사를 사용합니다.
    
    Args:
        theta_true: [feature_dim] (특정 샘플 1개에 대해)
        theta_samples: [num_samples, feature_dim]
    """
    mean_vec = np.mean(theta_samples, axis=0)
    cov_matrix = np.cov(theta_samples, rowvar=False)
    
    try:
        # allow_singular=True 로 설정하여 비식별성 계곡에서의 수치적 붕괴 방지
        mvn = multivariate_normal(mean=mean_vec, cov=cov_matrix, allow_singular=True)
        return -mvn.logpdf(theta_true)
    except Exception:
        # 공분산 행렬이 극도로 불안정할 경우
        return np.nan