from abc import ABC, abstractmethod
import numpy as np

class System(ABC):
    """
    Abstract base class 
    모든 ODE 시스템의 기본 구조를 정의하는 추상 베이스 클래스
    SDE 확장을 위해 drift와 diffusion 메서드를 분리함
    """
    
    @property
    @abstractmethod
    def name(self):
        """시스템의 이름 (예: 'lotka_volterra')"""
        pass

    @property
    @abstractmethod
    def param_names(self):
        """파라미터 이름 리스트 (예: ['alpha', 'beta', ...])"""
        pass

    @property
    @abstractmethod
    def param_ranges(self):
        """파라미터 샘플링 범위 딕셔너리"""
        pass
    
    @property
    @abstractmethod
    def initial_conditions(self):
        """초기 조건 [y0_min, y0_max]"""
        pass

    @property
    @abstractmethod
    def t_span(self):
        """ODE 솔버를 위한 시간 범위 [t_start, t_end]"""
        pass
    
    @property
    @abstractmethod
    def t_points(self):
        """데이터를 관측할 시간 지점 numpy 배열"""
        pass
    
    @property
    @abstractmethod
    def observed_var_idx(self):
        """관측 변수의 인덱스 (예: x는 0)"""
        pass

    @property
    @abstractmethod
    def hidden_var_idx(self):
        """숨겨진 변수의 인덱스 (예: y는 1)"""
        pass

    @abstractmethod
    def sample_initial_conditions(self, params_dict):
        """각 시스템별 초기값 샘플링 메서드"""
        pass

    @staticmethod
    @abstractmethod
    def ode_func(t, y, params):
        """시스템의 ODE 함수"""
        pass

    # ---- SDE Extension ----
    def drift_func(self, t, y, params):
        """SDE의 drift 함수 (기본적으로 ODE 함수와 동일)"""
        return self.ode_func(t, y, params)
    
    def diffusion_func(self, t, y, params):
        """SDE의 diffusion 함수 (기본적으로 0 행렬 반환)"""
        return np.zeros_like(y)