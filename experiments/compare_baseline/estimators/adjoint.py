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
        """SIR ODE (3차원) + 민감도 ODE (9차원) = 12차원 시스템 (완벽 정상)"""
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
        
        # 🚨 [수정 1] S_0는 관측값 그대로! R_0를 나머지 인원으로 계산
        S_0 = float(x_obs_0)
        R_0 = max(0.0, N - S_0 - I_0)
        
        # 🚨 [수정 2] I_0에 대한 초기 민감도 설정
        # S(0)는 고정된 관측값이므로 dS(0)/dI_0 = 0.0
        # I(0) = I_0 이므로 dI(0)/dI_0 = 1.0
        # R(0) = N - S(0) - I_0 이므로 dR(0)/dI_0 = -1.0
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
                method='Radau'
            )
            
        # 만약 발산 시, 기울기가 0이 되어 멈추는 것을 방지하기 위해 가짜 기울기 반환
        if not sol.success or np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            return 1e6, np.ones_like(theta_opt)
            
        S_pred = sol.y[0]
        residuals = S_pred - x_obs.flatten()
        loss = 0.5 * np.sum(residuals**2)
        
        dS_dbeta  = sol.y[3]
        dS_dgamma = sol.y[6]
        dS_dI0    = sol.y[9]
        
        # 연쇄 법칙(Chain Rule)을 통한 정확한 Gradient
        grad_beta  = np.sum(residuals * dS_dbeta)
        grad_gamma = np.sum(residuals * dS_dgamma)
        grad_I0    = np.sum(residuals * dS_dI0)
        
        return loss, np.array([grad_beta, grad_gamma, grad_I0])

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        theta_opt_init = np.concatenate([theta_init, x_hid_init])
        x_obs_0 = x_obs[0] if x_obs.ndim == 1 else x_obs[0, 0]
        
        N = sum([val[0] for val in self.sys.initial_conditions])
        bounds = [(0.0, np.inf), (0.0, np.inf), (0.0, N)]
        
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
                jac=True,  # 해석적 기울기 사용
                callback=callback,
                options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-6}
            )
            
        exec_time = time.time() - start_time

        theta_hat = result.x[:self.p]
        x_hid_hat_0 = result.x[self.p:]

        return theta_hat, x_hid_hat_0, exec_time, p_history