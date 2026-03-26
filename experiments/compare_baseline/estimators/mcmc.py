import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp

from experiments.compare_baseline.core import BaseEstimator

class MCMCEstimator(BaseEstimator):
    def __init__(self, system_obj, n_iters=3000, step_size=0.05):
        """
        Metropolis-Hastings 알고리즘을 이용한 MCMC 추정기.
        Args:
            system_obj: systems 패키지의 시스템 객체
            n_iters: MCMC 체인의 샘플링 횟수
            step_size: 제안 분포(Random Walk)의 표준편차 비율
        """
        super().__init__(system_obj.name)
        self.sys = system_obj
        self.p = len(self.sys.param_names)
        self.n_iters = n_iters
        self.step_size = step_size

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
        else:
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
            
        return np.array(lower_bounds), np.array(upper_bounds)

    def _simulate_forward(self, t_eval: np.ndarray, theta_opt: np.ndarray, x_obs_0: float):
        params = theta_opt[:self.p]
        x_hid_0 = theta_opt[self.p:]
        y0 = self._reconstruct_initial_state(x_obs_0, x_hid_0)

        # 1. 모든 경고 억제 및 예외 처리 (Adjoint와 동일한 보호막)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sol = solve_ivp(
                    fun=self.sys.ode_func,
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=y0,
                    t_eval=t_eval,
                    args=(params,),
                    method='Radau'
                )
                
                # 적분 실패 또는 발산 시 즉시 None 반환
                if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                    return None
                    
                return sol.y[self.sys.observed_var_idx]
            except Exception:
                return None
    def _log_prior(self, theta_opt):
        lower, upper = self._get_optimization_bounds()
        if np.any(theta_opt < lower) or np.any(theta_opt > upper):
            return -np.inf
        return 0.0

    def _log_likelihood(self, theta_opt, t_eval, x_obs, x_obs_0):
        x_obs_pred = self._simulate_forward(t_eval, theta_opt, x_obs_0)
        if x_obs_pred is None:
            return -np.inf
        residuals = x_obs_pred - x_obs.flatten()
        return -0.5 * np.sum(residuals**2)

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        theta_opt_curr = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]

        start_time = time.time()

        if self._log_prior(theta_opt_curr) == -np.inf:
            lower, upper = self._get_optimization_bounds()
            theta_opt_curr = np.clip(theta_opt_curr, lower, upper)
            
        # 2. 초기값 자체의 ODE 폭발 확인 (빠른 포기 로직)
        log_prior_curr = self._log_prior(theta_opt_curr)
        log_like_curr = self._log_likelihood(theta_opt_curr, t_eval, x_obs, x_obs_0)
        log_prob_curr = log_prior_curr + log_like_curr

        # 만약 초기 섭동값이 ODE를 터뜨리는 값이면, 
        # 3000번 반복해봐야 의미가 없으므로 즉시 실패 처리 (시간 낭비 방지)
        if log_prob_curr == -np.inf:
            return theta_init, x_hid_init, time.time() - start_time

        best_theta_opt = theta_opt_curr.copy()
        best_log_prob = log_prob_curr

        # Metropolis-Hastings 
        for _ in range(self.n_iters):
            # Scale-aware Random Walk 제안
            proposal_std = self.step_size * (np.abs(theta_opt_curr) + 1e-3)
            theta_opt_prop = np.random.normal(theta_opt_curr, proposal_std)
            
            log_prior_prop = self._log_prior(theta_opt_prop)
            if log_prior_prop == -np.inf:
                log_prob_prop = -np.inf
            else:
                log_prob_prop = log_prior_prop + self._log_likelihood(theta_opt_prop, t_eval, x_obs, x_obs_0)
                
            # Metropolis 채택 기준
            if log_prob_prop > log_prob_curr:
                accept = True
            else:
                accept = np.log(np.random.rand()) < (log_prob_prop - log_prob_curr)

            if accept:
                theta_opt_curr = theta_opt_prop
                log_prob_curr = log_prob_prop
                
                # MAP 갱신
                if log_prob_curr > best_log_prob:
                    best_log_prob = log_prob_curr
                    best_theta_opt = theta_opt_curr.copy()

        exec_time = time.time() - start_time

        theta_hat = best_theta_opt[:self.p]
        x_hid_hat_0 = best_theta_opt[self.p:]

        return theta_hat, x_hid_hat_0, exec_time