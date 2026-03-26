import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from experiments.compare_baseline.core import BaseEstimator

class AdjointLBFGSEstimator(BaseEstimator):
    def __init__(self, system_obj):
        """
        Adjoint Method의 최적화 궤적을 모사하는 L-BFGS 기반 추정기입니다.
        (기울기 계산의 효율성만 다를 뿐, 최적화 궤적은 Adjoint와 수학적으로 동일합니다.)
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
            return [x_obs_0, x_hid_0[0]]
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
            
        return list(zip(lower_bounds, upper_bounds))

    def _simulate_forward(self, t_eval: np.ndarray, theta_opt: np.ndarray, x_obs_0: float):
        params = theta_opt[:self.p]
        x_hid_0 = theta_opt[self.p:]
        y0 = self._reconstruct_initial_state(x_obs_0, x_hid_0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sol = solve_ivp(
                    fun=self.sys.ode_func,
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=y0,
                    t_eval=t_eval,
                    args=(params,),
                    method='Radau' # RK45보다 Stiff/Exploding 시스템에 훨씬 강건함
                )
                
                # 적분에 실패했거나, 결과에 NaN/Inf가 섞여 있다면 실패로 간주
                if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                    return None
                    
                return sol.y[self.sys.observed_var_idx]
            except Exception:
                # 그 외의 모든 에러(예: ODE Solver 내부 에러)도 None으로 처리
                return None
            
        return sol.y[self.sys.observed_var_idx]

    def _objective(self, theta_opt: np.ndarray, t_eval: np.ndarray, x_obs: np.ndarray, x_obs_0: float):
        """목적 함수: 잔차 제곱합 (Sum of Squared Residuals)"""
        x_obs_pred = self._simulate_forward(t_eval, theta_opt, x_obs_0)
        
        # ODE 발산 시 옵티마이저에 강한 페널티 부여
        if x_obs_pred is None:
            return 1e6 
            
        residuals = x_obs_pred - x_obs.flatten()
        return 0.5 * np.sum(residuals**2)

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        theta_opt_init = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]
        bounds = self._get_optimization_bounds()

        start_time = time.time()

        # L-BFGS-B 옵티마이저 실행
        # jac='3-point'를 통해 Adjoint가 제공할 정확한 그래디언트를 모사합니다.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # L-BFGS-B 옵티마이저 실행
            result = minimize(
                fun=self._objective,
                x0=theta_opt_init,
                args=(t_eval, x_obs, x_obs_0),
                method='L-BFGS-B',
                bounds=bounds,
                jac='3-point', 
                # maxiter(최대 반복 횟수)와 maxfun(최대 함수 호출 횟수)을 작게 제한하여 
                # 절벽에 갇혀도 빨리 포기하고 다음 몬테카를로 시도로 넘어가게 합니다.
                options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-4} 
            )

        exec_time = time.time() - start_time

        theta_hat = result.x[:self.p]
        x_hid_hat_0 = result.x[self.p:]

        return theta_hat, x_hid_hat_0, exec_time