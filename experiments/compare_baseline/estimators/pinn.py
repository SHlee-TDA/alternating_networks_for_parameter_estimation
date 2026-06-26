import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from experiments.compare_baseline.core import BaseEstimator

class PINNEstimator(BaseEstimator):
    def __init__(self, system_obj, hidden_dim=64, num_layers=4):
        super().__init__(system_obj.name)
        self.sys = system_obj
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def _build_net(self, out_dim):
        layers = []
        layers.append(nn.Linear(1, self.hidden_dim))
        layers.append(nn.Tanh()) 
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        
        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        return net

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        obs_dim = 1 if len(x_obs.shape) == 1 else x_obs.shape[1] 
        x_obs_pure = x_obs[:, 0].reshape(-1, 1) if obs_dim > 1 else x_obs.reshape(-1, 1)
        
        t_max = float(t_eval[-1])
        t_eval_scaled = t_eval / t_max
        
        t_torch = torch.tensor(t_eval_scaled, dtype=torch.float32, device=self.device).view(-1, 1).requires_grad_(True)
        x_obs_torch = torch.tensor(x_obs_pure, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # 🚨 [수정 1] Random Sampling 대신 균등 격자(Grid) 배치로 좁은 피크(Peak)를 놓치지 않게 함
        num_colloc = 200
        t_colloc = np.linspace(0, 1.0, num_colloc).reshape(-1, 1)
        t_physics = np.vstack((t_eval_scaled.reshape(-1, 1), t_colloc))
        t_physics = np.unique(t_physics, axis=0) # 중복 제거
        t_phys_torch = torch.tensor(t_physics, dtype=torch.float32, device=self.device).requires_grad_(True)
        
        state_dim = len(self.sys.initial_conditions) if hasattr(self.sys, 'initial_conditions') else 2
        net = self._build_net(state_dim).to(self.device)
        
        # 🚨 [수정 2] 파라미터 무한대 발산을 막기 위해 Sigmoid 변환 사용 (최대 10.0으로 제한)
        # theta_actual = 10.0 * sigmoid(theta_raw)
        def inv_sigmoid(val, max_val=10.0):
            val = np.clip(val, 1e-4, max_val - 1e-4)
            return np.log(val / (max_val - val))
            
        theta_raw_init = [inv_sigmoid(th, 10.0) for th in theta_init]
        theta_raw = nn.Parameter(torch.tensor(theta_raw_init, dtype=torch.float32, device=self.device))
        
        # I_0의 상한선은 (N - S_0)
        x_obs_0 = float(x_obs_pure[0])
        N_val = sum([val[0] for val in self.sys.initial_conditions]) if self.sys.name.lower() == 'sir' else 100.0
        max_I0 = max(0.001, N_val - x_obs_0)
        
        x_hid_raw_init = [inv_sigmoid(x, max_I0) for x in x_hid_init]
        x_hid_raw = nn.Parameter(torch.tensor(x_hid_raw_init, dtype=torch.float32, device=self.device))

        start_time = time.time()
        
        def compute_loss():
            # 🚨 물리적 상한/하한이 강제된 안전한 파라미터 생성
            theta_actual = 10.0 * torch.sigmoid(theta_raw)
            x_hid_actual = max_I0 * torch.sigmoid(x_hid_raw)
            
            # 🚨 [수정 3] 상태 변수가 절대로 음수가 되지 않도록 Softplus 래핑 (매우 중요!!!)
            # 이걸 안 하면 -S * -I 가 되어 ODE가 대폭발합니다.
            u_pred_obs = F.softplus(net(t_torch))
            u_pred_phys = F.softplus(net(t_phys_torch))
            
            # (1) Data Loss
            obs_idx = self.sys.observed_var_idx
            loss_data = torch.mean((u_pred_obs[:, obs_idx:obs_idx+1] - x_obs_torch)**2)
            
            # (2) Physics Loss
            dudt = []
            for i in range(state_dim):
                grad = torch.autograd.grad(
                    u_pred_phys[:, i].sum(), t_phys_torch, 
                    create_graph=True, retain_graph=True
                )[0]
                dudt.append(grad) 
            dudt = torch.cat(dudt, dim=1)
            
            rhs = self.sys.ode_func_torch(u_pred_phys, theta_actual)
            
            # 🚨 [수정 4] grad / t_max 대신 우변에 t_max를 곱하여 Vanishing Gradient 방지
            rhs_scaled = rhs * t_max
            loss_phys = torch.mean((dudt - rhs_scaled)**2)
            
            # (3) Initial Condition Loss
            hid_indices = self.sys.hidden_var_idx
            loss_ic = torch.mean((u_pred_obs[0, hid_indices] - x_hid_actual)**2)
            
            # SIR 모델 질량 보존
            if self.sys.name.lower() == 'sir':
                target_R0 = max(0.0, N_val - x_obs_0 - x_hid_actual[0].item())
                loss_ic += torch.mean((u_pred_obs[0, 2] - target_R0)**2)
                # 관측치(S)의 초기값도 꽉 잡아줍니다.
                loss_ic += torch.mean((u_pred_obs[0, 0] - x_obs_0)**2)
            
            # 가중치: 데이터가 스파스하므로 데이터 로스에 절대적인 신뢰를 부여
            return 100.0 * loss_data + 1.0 * loss_phys + 10.0 * loss_ic

        try:
            p_history = [theta_init.copy()] 
            
            # Phase 1: Adam Optimizer
            optimizer_adam = torch.optim.Adam(
                list(net.parameters()) + [theta_raw, x_hid_raw], 
                lr=2e-3 # 약간 속도업
            )
            for epoch in range(1500): # 안정화를 위해 epoch 살짝 늘림
                optimizer_adam.zero_grad()
                loss = compute_loss()
                loss.backward()
                
                # 🚨 [수정 5] Gradient Clipping으로 갑작스런 폭발 완벽 차단
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer_adam.step()

                if epoch % 50 == 0:
                    with torch.no_grad():
                        current_theta = (10.0 * torch.sigmoid(theta_raw)).detach().cpu().numpy()
                        p_history.append(current_theta.copy())
                        
            # Phase 2: L-BFGS Optimizer 
            optimizer_lbfgs = torch.optim.LBFGS(
                list(net.parameters()) + [theta_raw, x_hid_raw],
                lr=0.1, # LBFGS 보폭을 줄여서 안정성 도모
                max_iter=100, 
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=50,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer_lbfgs.zero_grad()
                loss = compute_loss()
                loss.backward()
                
                with torch.no_grad():
                    current_theta = (10.0 * torch.sigmoid(theta_raw)).detach().cpu().numpy()
                    if not np.allclose(p_history[-1], current_theta, atol=1e-5):
                        p_history.append(current_theta.copy())
                return loss

            optimizer_lbfgs.step(closure)

        except Exception as e:
            import traceback
            print(f"\n[PINN Failed]: {e}")

        exec_time = time.time() - start_time
        
        theta_hat = (10.0 * torch.sigmoid(theta_raw)).detach().cpu().numpy()
        x_hid_hat_0 = (max_I0 * torch.sigmoid(x_hid_raw)).detach().cpu().numpy()
        
        del net, optimizer_adam, optimizer_lbfgs, t_torch, x_obs_torch, t_phys_torch, theta_raw, x_hid_raw
        
        return theta_hat, x_hid_hat_0, exec_time, p_history