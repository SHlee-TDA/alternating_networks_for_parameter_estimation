import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from experiments.compare_baseline.core import BaseEstimator

class AdjointLBFGSEstimator(BaseEstimator):
    def __init__(self, system_obj):
        super().__init__(system_obj.name)
        self.sys = system_obj
        self.p = len(self.sys.param_names)

    def _augmented_ode(self, t, y_aug, beta, gamma, N):
        """SIR ODE (3차원) + 민감도 ODE (9차원) = 12차원 시스템"""
        S, I, R = y_aug[0:3]
        
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        
        J11, J12 = -beta * I / N, -beta * S / N
        J21, J22 = beta * I / N,  beta * S / N - gamma
        
        Sb, Ib, Rb = y_aug[3:6]
        dSb_dt = J11 * Sb + J12 * Ib - (S * I / N)
        dIb_dt = J21 * Sb + J22 * Ib + (S * I / N)
        dRb_dt = gamma * Ib
        
        Sg, Ig, Rg = y_aug[6:9]
        dSg_dt = J11 * Sg + J12 * Ig
        dIg_dt = J21 * Sg + J22 * Ig - I
        dRg_dt = gamma * Ig + I
        
        Si0, Ii0, Ri0 = y_aug[9:12]
        dSi0_dt = J11 * Si0 + J12 * Ii0
        dIi0_dt = J21 * Si0 + J22 * Ii0
        dRi0_dt = gamma * Ii0
        
        return [dSdt, dIdt, dRdt, 
                dSb_dt, dIb_dt, dRb_dt, 
                dSg_dt, dIg_dt, dRg_dt, 
                dSi0_dt, dIi0_dt, dRi0_dt]

    def _objective_and_gradient(self, theta_opt: np.ndarray, t_eval: np.ndarray, x_obs: np.ndarray, x_obs_0: float):
        beta, gamma = theta_opt[0], theta_opt[1]
        I_0 = theta_opt[2]
        N = sum([val[0] for val in self.sys.initial_conditions])
        
        S_0 = float(x_obs_0)
        R_0 = max(0.0, N - S_0 - I_0)
        
        y_aug_0 = [S_0, I_0, R_0,  
                   0.0, 0.0, 0.0,  
                   0.0, 0.0, 0.0,  
                   0.0, 1.0, -1.0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(
                fun=self._augmented_ode,
                t_span=(t_eval[0], t_eval[-1]),
                y0=y_aug_0,
                t_eval=t_eval,
                args=(beta, gamma, N),
                method='Radau' # Stiff 대응 솔버 유지
            )
            
        # 🚨 [수정 1] 발산 시, 옵티마이저가 확실하게 '이쪽 방향은 아니다'라고 느끼도록 스케일이 큰 패널티 부여
        if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            return 1e6, np.full_like(theta_opt, 1e4)
            
        S_pred = sol.y[0]
        residuals = S_pred - x_obs.flatten()
        loss = 0.5 * np.sum(residuals**2)
        
        dS_dbeta  = sol.y[3]
        dS_dgamma = sol.y[6]
        dS_dI0    = sol.y[9]
        
        grad_beta  = np.sum(residuals * dS_dbeta)
        grad_gamma = np.sum(residuals * dS_dgamma)
        grad_I0    = np.sum(residuals * dS_dI0)
        
        return loss, np.array([grad_beta, grad_gamma, grad_I0])

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        theta_opt_init = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]
        
        N = sum([val[0] for val in self.sys.initial_conditions])
        
        # 🚨 [수정 2] 무한대 방지 (10.0), 0 방지 (1e-5), 동적 I_0 바운드
        max_I0 = max(0.001, N - x_obs_0)
        bounds = [(1e-5, 10.0), (1e-5, 10.0), (1e-5, max_I0)]
        
        # 🚨 [수정 3] L-BFGS-B는 x0가 bounds 바깥에 있으면 ValueError를 냅니다. 철저히 클리핑!
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        theta_opt_init = np.clip(theta_opt_init, lower_bounds, upper_bounds)
        
        p_history = [theta_opt_init.copy()[:self.p]]

        def callback(xk):
            p_history.append(xk[:self.p].copy())

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                fun=self._objective_and_gradient,
                x0=theta_opt_init,
                args=(t_eval, x_obs, x_obs_0),
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,  # 우리가 직접 계산한 Gradient 사용
                callback=callback,
                # 🚨 [수정 4] maxfun을 명시하여 무한 평가 방지, ftol로 조기종료 확보
                options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-6}
            )
            
        exec_time = time.time() - start_time

        theta_hat = result.x[:self.p]
        x_hid_hat_0 = result.x[self.p:]

        return theta_hat, x_hid_hat_0, exec_time, p_history