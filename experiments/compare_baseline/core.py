import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseEstimator(ABC):
    """
    모든 매개변수 추정 방법론(NLLS, PINN, Ours)의 공통 인터페이스입니다.
    """
    def __init__(self, system_name: str = "sir"):
        self.system_name = system_name

    @abstractmethod
    def fit(self, 
            t_eval: np.ndarray, 
            x_obs: np.ndarray, 
            theta_init: np.ndarray, 
            x_hid_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        주어진 관측 데이터와 초기 추정치를 바탕으로 매개변수를 최적화합니다.
        
        Args:
            t_eval (np.ndarray): 관측 시간 그리드, shape (N,)
            x_obs (np.ndarray): 부분 관측된 상태 데이터, shape (N, d_obs)
            theta_init (np.ndarray): 섭동된 매개변수 초기값, shape (p,)
            x_hid_init (np.ndarray): 섭동된 은닉 상태 초기값, shape (d_hid,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 
                - 최종 추정된 매개변수 (theta_hat)
                - 최종 추정된 은닉 상태 초기값 (x_hid_hat_0)
                - 알고리즘 실행 시간 (exec_time_seconds)
        """
        pass

def calculate_relative_error(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    """상대 오차(L2 Norm 기준)를 계산합니다."""
    return np.linalg.norm(theta_hat - theta_true) / np.linalg.norm(theta_true)

def evaluate_success(error: float, threshold: float = 0.05) -> bool:
    """오차가 임계값 미만인지 확인하여 수렴 성공 여부를 반환합니다."""
    return error < threshold

def perturb_initial_values(true_values, radius, rng=None):
    """
    주어진 참값에 반경(radius) 비례하는 섭동(perturbation)을 가합니다.
    
    Args:
        true_values (np.ndarray): 참값 배열
        radius (float): 섭동 반경 (상대적 크기)
        rng (np.random.Generator, optional): 난수 생성기. 
                                             None이면 전역 np.random 사용
    """
    if rng is None:
        # rng가 제공되지 않으면 기존처럼 전역 상태 사용 (하위 호환성 유지)
        noise = np.random.uniform(-radius, radius, size=true_values.shape)
    else:
        # rng가 제공되면 독립적인 지역 난수 생성기 사용
        noise = rng.uniform(-radius, radius, size=true_values.shape)
        
    perturbed = true_values * (1 + noise)
    
    # 물리적 타당성을 위해 모든 값은 0보다 크도록 강제 클리핑
    return np.clip(perturbed, a_min=1e-6, a_max=None)