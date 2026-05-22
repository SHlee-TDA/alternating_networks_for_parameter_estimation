import warnings

import numpy as np
from scipy.stats import pearsonr, gaussian_kde


def calculate_point_metrics(y_true, y_samples):
    """
    y_true: [batch_size, feature_dim]
    y_samples: [batch_size, num_samples, feature_dim]
    """
    # 1. num_samples 축(axis=1)을 따라 평균을 구해 점 추정치 생성
    y_mean = np.mean(y_samples, axis=1) # 형태: [batch_size, feature_dim]
    
    # 2. RMSE (batch 방향으로 평균)
    rmse = np.sqrt(np.mean((y_true - y_mean)**2, axis=0))
    
    # 3. Pearson R (각 Feature 별로 계산)
    pearson_r = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        if np.var(y_true[:, i]) < 1e-12 or np.var(y_mean[:, i]) < 1e-12:
            pearson_r[i] = 0.0
        else:
            r, _ = pearsonr(y_true[:, i], y_mean[:, i])
            pearson_r[i] = r if np.isfinite(r) else 0.0
            
    return rmse, pearson_r

def calculate_prediction_interval(y_true, y_samples, alpha=0.05):
    """
    y_true: [batch_size, feature_dim]
    y_samples: [batch_size, num_samples, feature_dim]
    """
    # num_samples 축(axis=1)에 대해 분위수 계산
    lower_bound = np.quantile(y_samples, q=alpha/2, axis=1)
    upper_bound = np.quantile(y_samples, q=1 - alpha/2, axis=1)
    
    is_captured = (y_true >= lower_bound) & (y_true <= upper_bound)
    
    picp = np.mean(is_captured, axis=0)
    mpiw = np.mean(upper_bound - lower_bound, axis=0)
    
    return picp, mpiw

def calculate_crps(y_true, y_samples):
    """
    y_true: [batch_size, feature_dim]
    y_samples: [batch_size, num_samples, feature_dim]
    """
    num_samples = y_samples.shape[1]
    
    # E|X - y_true|: y_true를 [batch_size, 1, feature_dim]으로 팽창시켜 오차 계산
    diff_true = np.mean(np.abs(y_samples - y_true[:, np.newaxis, :]), axis=1)
    
    # E|X - X'|: 모든 샘플 쌍 간의 오차 계산
    diff_samples = np.zeros_like(y_true)
    for i in range(num_samples):
        # i번째 샘플 [batch_size, 1, feature_dim]과 전체 샘플들 간의 오차
        diff_samples += np.sum(np.abs(y_samples - y_samples[:, i:i+1, :]), axis=1)
    diff_samples = diff_samples / (num_samples ** 2)
    
    crps = diff_true - 0.5 * diff_samples
    return np.mean(crps, axis=0) 

def calculate_kde_nll(theta_true, theta_samples):
    """
    각 샘플(Batch)별로 가우시안 커널 밀도 추정(KDE)을 수행하여 NLL을 계산합니다.
    초승달 모양, Bimodal 등 비선형적인 사후 분포(Posterior) 평가에 필수적입니다.
    
    theta_true: [batch_size, feature_dim]
    theta_samples: [batch_size, num_samples, feature_dim]
    """
    batch_size, num_samples, feature_dim = theta_samples.shape
    nll_list = []
    
    for i in range(batch_size):
        # gaussian_kde는 [차원 수, 샘플 수] 형태를 요구하므로 전치(Transpose)합니다.
        samples_i = theta_samples[i].T 
        true_i = theta_true[i]
        
        try:
            # 모든 샘플이 한 점으로 붕괴했을 때 KDE가 터지는 것을 막기 위한 미세 노이즈
            jitter = np.random.normal(0, 1e-6, samples_i.shape)
            kde = gaussian_kde(samples_i + jitter)
            
            # logpdf는 배열을 반환하므로 [0]으로 스칼라 값을 추출
            log_prob = kde.logpdf(true_i)[0]
            
            if not np.isfinite(log_prob):
                nll_list.append(100.0) 
            else:
                nll_list.append(-log_prob)
                
        except Exception:
            # 행렬 연산 실패 시 안전한 패널티 부여
            nll_list.append(100.0)
            
    # 전체 Batch에 대한 평균 NLL 반환
    avg_nll = np.mean(nll_list)
    
    # 다른 메트릭과 반환 형태를 맞추기 위해 배열 형태로 반환
    return np.array([avg_nll] * feature_dim)
    