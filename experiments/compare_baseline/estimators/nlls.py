import time
import warnings
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
        self.p = len(self.sys.param_names)
        
    def _reconstruct_initial_state(self, x_obs_0: float, x_hid_0: np.ndarray):
        name = self.sys.name.lower()
        if name == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            I0 = x_hid_0[0]
            R0 = N - x_obs_0 - I0
            return [x_obs_0, I0, R0]
            
        elif name == 'lotka_volterra':
            y0 = x_hid_0[0]
            return [x_obs_0, y0]
            
        elif name == 'ogtt':
            raise NotImplementedError("OGTT 초기 상태 재구성 로직이 필요합니다.")
        else:
            raise ValueError(f"Unknown system name: {name}")

    def _get_optimization_bounds(self):
        name = self.sys.name.lower()
        lower_bounds = [0.0] * self.p
        upper_bounds = [np.inf] * self.p
        
        if name == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            lower_bounds.append(0.0)
            upper_bounds.append(N)
        elif name == 'lotka_volterra':
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
        else:
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
            
        return lower_bounds, upper_bounds

    def _simulate_forward(self, t_eval: np.ndarray, theta_opt: np.ndarray, x_obs_0: float):
        params = theta_opt[:self.p]
        x_hid_0 = theta_opt[self.p:]

        y0 = self._reconstruct_initial_state(x_obs_0, x_hid_0)

        # 🚨 [수정 1] 수동 물리적 제약 페널티 로직 삭제
        # trf 알고리즘이 Bounds를 완벽히 통제하므로 부동소수점 오차로 인한 강제 페널티를 막습니다.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 🚨 [수정 2] RK45 대신 Radau 사용 (유한차분 탐색 시 강건성 확보)
            sol = solve_ivp(
                fun=self.sys.ode_func,
                t_span=(t_eval[0], t_eval[-1]),
                y0=y0,
                t_eval=t_eval,
                args=(params,),
                method='Radau' 
            )
        
        # 🚨 [수정 3] 발산 페널티 완화 및 오류 검출 강화
        if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            # 극단적인 상수(1e6) 대신, 기울기 폭발을 막는 적당한 상수(1e4) 사용
            return np.full(len(t_eval), 1e4)
            
        return sol.y[self.sys.observed_var_idx]

    def _residual(self, theta_opt: np.ndarray, t_eval: np.ndarray, x_obs: np.ndarray, x_obs_0: float):
        x_obs_pred = self._simulate_forward(t_eval, theta_opt, x_obs_0)
        return x_obs_pred - x_obs.flatten()

    def _reconstruct_initial_state(self, x_obs_0: float, x_hid_0: np.ndarray):
        name = self.sys.name.lower()
        if name == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            I0 = x_hid_0[0]
            # 🚨 [수정] 음수 인구수 방지
            R0 = max(0.0, N - x_obs_0 - I0) 
            return [x_obs_0, I0, R0]
        elif name == 'lotka_volterra':
            return [x_obs_0, x_hid_0[0]]
        else:
            raise ValueError(f"Unknown system name: {name}")

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        theta_opt_init = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]
        
        lower_bounds, upper_bounds = self._get_optimization_bounds()
        
        # 동적 Bounds 설정 (I_0가 N - S_0 를 넘지 못하도록)
        if self.sys.name.lower() == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            max_I0 = max(0.001, N - x_obs_0)
            upper_bounds[-1] = max_I0
            
        # 🚨 [안전장치 추가] 초기값이 Bounds를 조금이라도 벗어나면 완벽하게 클리핑
        theta_opt_init = np.clip(theta_opt_init, lower_bounds, upper_bounds)
            
        # 🚨 [버그 수정 완료] 파라미터(beta, gamma) 부분만 잘라서 저장 (2차원)
        p_history = [theta_opt_init[:self.p].copy()]
        
        start_time = time.time()
        
        def tracked_residuals(theta_opt, *args):
            current_theta = theta_opt[:self.p].copy()
            # 이제 2차원 vs 2차원으로 정상 비교됩니다.
            if len(p_history) == 0 or not np.allclose(p_history[-1], current_theta, atol=1e-5):
                p_history.append(current_theta)
            return self._residual(theta_opt, *args)    
        
        result = least_squares(
            fun=tracked_residuals, 
            x0=theta_opt_init,
            args=(t_eval, x_obs, x_obs_0),
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            loss='linear', 
            max_nfev=200
        )
        
        exec_time = time.time() - start_time
        return result.x[:self.p], result.x[self.p:], exec_time, p_history