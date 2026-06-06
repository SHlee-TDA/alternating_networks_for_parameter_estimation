import torch
import torch.nn as nn
import numpy as np
import time

from experiments.compare_baseline.core import BaseEstimator

class PINNEstimator(BaseEstimator):
    def __init__(self, system_obj, hidden_dim=64, num_layers=4): # 🚨 레이어 살짝 증가
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
        
        # 네트워크 가중치 초기화 (Xavier) - 학습 안정화
        net = nn.Sequential(*layers)
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        return net

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        obs_dim = 1 if len(x_obs.shape) == 1 else x_obs.shape[1] 
        x_obs_pure = x_obs[:, 0].reshape(-1, 1) if obs_dim > 1 else x_obs.reshape(-1, 1)
        
        # 시간 스케일링 (0~110 -> 0~1) - PINN 학습 안정성을 위한 필수 테크닉
        t_max = t_eval[-1]
        t_eval_scaled = t_eval / t_max
        
        t_torch = torch.tensor(t_eval_scaled, dtype=torch.float32, device=self.device).view(-1, 1).requires_grad_(True)
        x_obs_torch = torch.tensor(x_obs_pure, dtype=torch.float32, device=self.device).view(-1, 1)
        
        num_colloc = 200
        t_colloc = np.random.uniform(0, 1.0, (num_colloc, 1))
        t_physics = np.vstack((t_eval_scaled.reshape(-1, 1), t_colloc))
        t_phys_torch = torch.tensor(t_physics, dtype=torch.float32, device=self.device).requires_grad_(True)
        
        state_dim = len(self.sys.initial_conditions) if hasattr(self.sys, 'initial_conditions') else 2
        net = self._build_net(state_dim).to(self.device)
        
        # 🚨 [수정 1] 파라미터 양수 강제를 위해 Inverse Softplus로 초기화
        def inv_softplus(val):
            return np.log(np.exp(val) - 1.0) if val > 0 else -5.0
            
        theta_raw_init = [inv_softplus(th) for th in theta_init]
        theta_raw = nn.Parameter(torch.tensor(theta_raw_init, dtype=torch.float32, device=self.device))
        
        x_hid_raw_init = [inv_softplus(x) for x in x_hid_init]
        x_hid_raw = nn.Parameter(torch.tensor(x_hid_raw_init, dtype=torch.float32, device=self.device))

        start_time = time.time()
        
        # 파라미터 변환 함수 (Raw -> Actual Positive Values)
        softplus = nn.Softplus()
        
        def compute_loss():
            # 실제 양수 파라미터 도출
            theta_actual = softplus(theta_raw)
            x_hid_actual = softplus(x_hid_raw)
            
            # (1) Data Loss
            u_pred_obs = net(t_torch)
            obs_idx = self.sys.observed_var_idx
            loss_data = torch.mean((u_pred_obs[:, obs_idx:obs_idx+1] - x_obs_torch)**2)
            
            # (2) Physics Loss
            u_pred_phys = net(t_phys_torch)
            dudt = []
            for i in range(state_dim):
                grad = torch.autograd.grad(
                    u_pred_phys[:, i].sum(), t_phys_torch, 
                    create_graph=True, retain_graph=True
                )[0]
                # 시간 스케일링을 했으므로 체인룰에 의해 미분값 보정
                dudt.append(grad / t_max) 
            dudt = torch.cat(dudt, dim=1)
            
            rhs = self.sys.ode_func_torch(u_pred_phys, theta_actual)
            loss_phys = torch.mean((dudt - rhs)**2)
            
            # (3) Initial Condition Loss
            # (3) Initial Condition Loss
            hid_indices = self.sys.hidden_var_idx
            loss_ic = torch.mean((u_pred_obs[0, hid_indices] - x_hid_actual)**2)
            
            # 🚨 [수정] SIR 모델의 경우 R(0) 고정 추가 (질량 보존 법칙 강제)
            if self.sys.name.lower() == 'sir':
                N_val = sum([val[0] for val in self.sys.initial_conditions])
                # R(0) = N - S(0) - I(0)
                target_R0 = N_val - x_obs_torch[0].item() - x_hid_actual[0]
                # R은 상태 벡터에서 3번째 인덱스(idx=2)
                loss_ic += torch.mean((u_pred_obs[0, 2] - target_R0)**2)
            
            # 🚨 [수정 2] Data Loss에 압도적인 가중치 (100배) 부여하여 원점 붕괴 방지
            lambda_data = 100.0
            lambda_phys = 1.0
            lambda_ic = 10.0
            
            return lambda_data * loss_data + lambda_phys * loss_phys + lambda_ic * loss_ic

        try:
            p_history = [theta_init.copy()] 
            
            # Phase 1: Adam Optimizer
            optimizer_adam = torch.optim.Adam(
                list(net.parameters()) + [theta_raw, x_hid_raw], 
                lr=1e-3
            )
            for epoch in range(1000):
                optimizer_adam.zero_grad()
                loss = compute_loss()
                loss.backward()
                optimizer_adam.step()

                if epoch % 20 == 0:
                    with torch.no_grad():
                        current_theta = softplus(theta_raw).detach().cpu().numpy()
                        p_history.append(current_theta.copy())
                        
            # Phase 2: L-BFGS Optimizer 
            optimizer_lbfgs = torch.optim.LBFGS(
                list(net.parameters()) + [theta_raw, x_hid_raw],
                lr=0.5,
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
                    current_theta = softplus(theta_raw).detach().cpu().numpy()
                    if not np.allclose(p_history[-1], current_theta, atol=1e-5):
                        p_history.append(current_theta.copy())
                return loss

            optimizer_lbfgs.step(closure)

        except Exception as e:
            import traceback
            print(f"\n[PINN Failed]: {e}")

        exec_time = time.time() - start_time
        
        theta_hat = softplus(theta_raw).detach().cpu().numpy()
        x_hid_hat_0 = softplus(x_hid_raw).detach().cpu().numpy()
        
        del net, optimizer_adam, optimizer_lbfgs, t_torch, x_obs_torch, t_phys_torch, theta_raw, x_hid_raw
        
        return theta_hat, x_hid_hat_0, exec_time, p_history