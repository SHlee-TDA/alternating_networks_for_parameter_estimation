import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from experiments.compare_baseline.core import BaseEstimator

class NLLSEstimator(BaseEstimator):
    def __init__(self, system_obj):
        """
        기존 systems 패키지의 시스템 객체(Sir, LotkaVolterra 등)를 그대로 주입받습니다.
        시스템 코드는 전혀 수정할 필요가 없습니다.
        """
        super().__init__(system_obj.name)
        self.sys = system_obj
        
        # 동적으로 파라미터 차원(p)을 계산합니다.
        self.p = len(self.sys.param_names)
        
    def _reconstruct_initial_state(self, x_obs_0: float, x_hid_0: np.ndarray):
        """
        기존 시스템 코드를 수정하지 않고, Estimator 내부에서 시스템 이름에 따라
        전체 초기 상태 벡터 y0를 재구성합니다.
        """
        name = self.sys.name.lower()
        if name == 'sir':
            # SIR: initial_conditions ([49.0], [1.0], [0.0]) 를 합하여 총 인구수 N 도출
            N = sum([val[0] for val in self.sys.initial_conditions])
            I0 = x_hid_0[0]
            R0 = N - x_obs_0 - I0
            return [x_obs_0, I0, R0]
            
        elif name == 'lotka_volterra':
            # Lotka-Volterra: [prey, predator]
            y0 = x_hid_0[0]
            return [x_obs_0, y0]
            
        elif name == 'ogtt':
            # 추후 OGTT가 추가되면 여기에 y0 재구성 로직을 추가하면 됩니다.
            raise NotImplementedError("OGTT 초기 상태 재구성 로직이 필요합니다.")
        else:
            raise ValueError(f"Unknown system name: {name}")

    def _get_optimization_bounds(self):
        """각 시스템의 파라미터와 은닉 상태에 대한 최적화 경계값을 설정합니다."""
        name = self.sys.name.lower()
        # 모든 동역학계 파라미터(rate)는 기본적으로 0 이상
        lower_bounds = [0.0] * self.p
        upper_bounds = [np.inf] * self.p
        
        if name == 'sir':
            # SIR의 은닉 상태 I(0)는 0 ~ N 사이
            N = sum([val[0] for val in self.sys.initial_conditions])
            lower_bounds.append(0.0)
            upper_bounds.append(N)
        elif name == 'lotka_volterra':
            # LV의 은닉 상태 y(0)는 0 ~ inf
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
        else:
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
            
        return lower_bounds, upper_bounds

    def _simulate_forward(self, t_eval: np.ndarray, theta_opt: np.ndarray, x_obs_0: float):
        """시스템에 독립적인 Forward ODE 적분"""
        params = theta_opt[:self.p]
        x_hid_0 = theta_opt[self.p:]

        y0 = self._reconstruct_initial_state(x_obs_0, x_hid_0)

        # 물리적 제약 페널티 방지
        bounds_lower, bounds_upper = self._get_optimization_bounds()
        if np.any(theta_opt < bounds_lower) or np.any(theta_opt > bounds_upper):
            return np.full(len(t_eval), 1e6) 

        sol = solve_ivp(
            fun=self.sys.ode_func,
            t_span=(t_eval[0], t_eval[-1]),
            y0=y0,
            t_eval=t_eval,
            args=(params,),
            method='RK45'
        )
        
        if not sol.success:
            return np.full(len(t_eval), 1e6)
            
        # 관측 인덱스(observed_var_idx)를 사용하여 동적으로 관측 데이터 추출
        return sol.y[self.sys.observed_var_idx]

    def _residual(self, theta_opt: np.ndarray, t_eval: np.ndarray, x_obs: np.ndarray, x_obs_0: float):
        """관측값과 예측값의 잔차 계산"""
        x_obs_pred = self._simulate_forward(t_eval, theta_opt, x_obs_0)
        return x_obs_pred - x_obs.flatten()

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        """범용 NLLS 최적화 수행"""
        theta_opt_init = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]

        start_time = time.time()

        lower_bounds, upper_bounds = self._get_optimization_bounds()
        
        result = least_squares(
            fun=self._residual,
            x0=theta_opt_init,
            args=(t_eval, x_obs, x_obs_0),
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            max_nfev=200
        )

        exec_time = time.time() - start_time

        theta_hat = result.x[:self.p]
        x_hid_hat_0 = result.x[self.p:]

        return theta_hat, x_hid_hat_0, exec_time