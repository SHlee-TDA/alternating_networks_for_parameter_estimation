import numpy as np
from scipy.stats import pearsonr, multivariate_normal

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

def calculate_parametric_nll(theta_true, theta_samples):
    mean_vec = np.mean(theta_samples, axis=0)
    cov_matrix = np.cov(theta_samples, rowvar=False)
    try:
        mvn = multivariate_normal(mean=mean_vec, cov=cov_matrix, allow_singular=True)
        return -mvn.logpdf(theta_true)
    except Exception:
        return np.nan
    