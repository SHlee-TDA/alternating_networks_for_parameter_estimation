import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp

from experiments.compare_baseline.core import BaseEstimator

class MCMCEstimator(BaseEstimator):
    def __init__(self, system_obj, n_iters=1000, init_step_size=0.1, temperature=10.0, max_time_sec=5.0):
        """
        Adaptive Metropolis-Hastings 알고리즘을 이용한 MCMC 추정기.
        Args:
            n_iters: MCMC 체인의 샘플링 횟수 (3000은 너무 많아 1000으로 조정)
            max_time_sec: 한 번의 fit 과정이 이 시간을 넘기면 강제 종료 (무한 루프 방어)
        """
        super().__init__(system_obj.name)
        self.sys = system_obj
        self.p = len(self.sys.param_names)
        self.n_iters = n_iters
        self.init_step_size = init_step_size
        self.temperature = temperature 
        self.max_time_sec = max_time_sec # 🚨 [안전장치] 최대 실행 시간

    def _reconstruct_initial_state(self, x_obs_0: float, x_hid_0: np.ndarray):
        name = self.sys.name.lower()
        if name == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            I0 = x_hid_0[0]
            # 🚨 음수 방지
            R0 = max(0.0, N - x_obs_0 - I0)
            return [x_obs_0, I0, R0]
        elif name == 'lotka_volterra':
            return [x_obs_0, x_hid_0[0]]
        else:
            raise ValueError(f"Unknown system name: {name}")

    def _get_optimization_bounds(self):
        name = self.sys.name.lower()
        # 🚨 [수정 1] NLLS와 동일하게 무한대와 0 방지
        lower_bounds = [1e-5] * self.p
        upper_bounds = [10.0] * self.p
        
        if name == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            lower_bounds.append(1e-5)
            upper_bounds.append(N)
        elif name == 'lotka_volterra':
            lower_bounds.append(1e-5)
            upper_bounds.append(np.inf)
        else:
            lower_bounds.append(1e-5)
            upper_bounds.append(np.inf)
            
        return np.array(lower_bounds), np.array(upper_bounds)

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
                    method='Radau' 
                )
                
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
        return -0.5 * np.sum(residuals**2) / self.temperature

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        theta_opt_curr = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]

        lower_bounds, upper_bounds = self._get_optimization_bounds()
        
        # 동적 Bounds 설정
        if self.sys.name.lower() == 'sir':
            N = sum([val[0] for val in self.sys.initial_conditions])
            max_I0 = max(0.001, N - x_obs_0)
            upper_bounds[-1] = max_I0
        
        # 🚨 초기값 클리핑
        theta_opt_curr = np.clip(theta_opt_curr, lower_bounds, upper_bounds)

        start_time = time.time()
            
        log_prior_curr = self._log_prior(theta_opt_curr)
        log_like_curr = self._log_likelihood(theta_opt_curr, t_eval, x_obs, x_obs_0)
        log_prob_curr = log_prior_curr + log_like_curr

        if log_prob_curr == -np.inf:
            return theta_init, x_hid_init, time.time() - start_time, [theta_init.copy()]

        best_theta_opt = theta_opt_curr.copy()
        best_log_prob = log_prob_curr
        p_history = [theta_opt_curr[:self.p].copy()]
        
        for i in range(self.n_iters): 
            # 🚨 [수정 2] 타임아웃 체크: 지정된 시간(예: 5초) 넘으면 미련 없이 종료
            if time.time() - start_time > self.max_time_sec:
                break
                
            # 🚨 [수정 3] 감쇠(Decay)하되 최소 보폭(0.01)은 유지하여 탐색이 얼어붙지 않게 함
            decay_factor = max(0.1, np.exp(-2.0 * i / self.n_iters))
            adaptive_step = self.init_step_size * decay_factor
            proposal_std = adaptive_step * (np.abs(theta_opt_curr) + 0.01)
            
            theta_opt_prop = np.random.normal(theta_opt_curr, proposal_std)
            theta_opt_prop = np.clip(theta_opt_prop, lower_bounds, upper_bounds)
            
            log_prior_prop = self._log_prior(theta_opt_prop)
            log_prob_prop = log_prior_prop + self._log_likelihood(theta_opt_prop, t_eval, x_obs, x_obs_0)
                
            if log_prob_prop > log_prob_curr:
                accept = True
            else:
                u = np.random.rand()
                accept = np.log(u + 1e-10) < (log_prob_prop - log_prob_curr)

            if accept:
                theta_opt_curr = theta_opt_prop
                log_prob_curr = log_prob_prop
                
                if log_prob_curr > best_log_prob:
                    best_log_prob = log_prob_curr
                    best_theta_opt = theta_opt_curr.copy()

            if i % 10 == 0:
                # 같은 값만 계속 기록되는 것을 방지
                if not np.allclose(p_history[-1], theta_opt_curr[:self.p], atol=1e-5):
                    p_history.append(theta_opt_curr[:self.p].copy())
                
        exec_time = time.time() - start_time
        return best_theta_opt[:self.p], best_theta_opt[self.p:], exec_time, p_history