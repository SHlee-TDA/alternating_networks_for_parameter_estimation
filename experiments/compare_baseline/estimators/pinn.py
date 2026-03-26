import torch
import torch.nn as nn
import numpy as np
import time

from experiments.compare_baseline.core import BaseEstimator

class PINNEstimator(BaseEstimator):
    def __init__(self, system_obj, hidden_dim=64, num_layers=3):
        super().__init__(system_obj.name)
        self.sys = system_obj
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def _build_net(self, out_dim):
        layers = []
        # 입력은 시간 t (1차원)
        layers.append(nn.Linear(1, self.hidden_dim))
        layers.append(nn.Tanh()) # PINN에서는 미분 가능성이 높은 Tanh를 선호함
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def fit(self, t_eval: np.ndarray, x_obs: np.ndarray, theta_init: np.ndarray, x_hid_init: np.ndarray):
        # 1. 관측 데이터 준비 (Data Loss 용도)
        # x_obs의 형태 보정 (앞선 차원 충돌 문제 해결)
        # 만약 x_obs가 미분값을 포함하고 있다면, 순수 상태 변수만 사용하도록 슬라이싱
        obs_dim = 1 if len(x_obs.shape) == 1 else x_obs.shape[1] 
        x_obs_pure = x_obs[:, 0].reshape(-1, 1) if obs_dim > 1 else x_obs.reshape(-1, 1)
        
        t_torch = torch.tensor(t_eval, dtype=torch.float32, device=self.device).view(-1, 1).requires_grad_(True)
        x_obs_torch = torch.tensor(x_obs_pure, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # 2. Collocation Points 생성 (Physics Loss 용도)
        # t_eval 범위 내에서 촘촘하게 N개의 점을 무작위로 생성
        num_colloc = 200
        t_colloc = np.random.uniform(t_eval[0], t_eval[-1], (num_colloc, 1))
        # 관측 지점과 병합하여 Physics Loss를 계산할 전체 시간 포인트 구성
        t_physics = np.vstack((t_eval.reshape(-1, 1), t_colloc))
        t_phys_torch = torch.tensor(t_physics, dtype=torch.float32, device=self.device).requires_grad_(True)
        
        # 3. 모델 및 파라미터 초기화
        state_dim = len(self.sys.initial_conditions) if hasattr(self.sys, 'initial_conditions') else 2
        net = self._build_net(state_dim).to(self.device)
        
        theta_param = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32, device=self.device))
        x_hid_0_param = nn.Parameter(torch.tensor(x_hid_init, dtype=torch.float32, device=self.device))
        
        optimizer = torch.optim.LBFGS(
            list(net.parameters()) + [theta_param, x_hid_0_param],
            lr=1.0,
            max_iter=1000, 
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        start_time = time.time()
        
        # Loss 계산 로직을 별도 함수로 분리 (Adam과 L-BFGS가 공유)
        def compute_loss():
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
                dudt.append(grad)
            dudt = torch.cat(dudt, dim=1)
            
            rhs = self.sys.ode_func_torch(u_pred_phys, theta_param)
            loss_phys = torch.mean((dudt - rhs)**2)
            
            # (3) Initial Condition Loss
            hid_indices = self.sys.hidden_var_idx
            loss_ic = torch.mean((u_pred_obs[0, hid_indices] - x_hid_0_param)**2)
            
            return loss_data + loss_phys + loss_ic

        try:
            # ==========================================
            # Phase 1: Adam Optimizer (빠른 전역 탐색)
            # ==========================================
            optimizer_adam = torch.optim.Adam(
                list(net.parameters()) + [theta_param, x_hid_0_param], 
                lr=1e-3
            )
            for _ in range(800):  # 빠른 속도로 800 에포크 진행
                optimizer_adam.zero_grad()
                loss = compute_loss()
                loss.backward()
                optimizer_adam.step()

            # ==========================================
            # Phase 2: L-BFGS Optimizer (정밀 타격)
            # ==========================================
            optimizer_lbfgs = torch.optim.LBFGS(
                list(net.parameters()) + [theta_param, x_hid_0_param],
                lr=1.0,
                max_iter=100,  # 1000에서 100으로 대폭 축소! (어차피 Adam이 다 와있음)
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=50,
                line_search_fn="strong_wolfe"
            )

            def closure():
                optimizer_lbfgs.zero_grad()
                loss = compute_loss()
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

        except Exception as e:
            import traceback
            print(f"\n[PINN Failed]: {e}")
            # print(traceback.format_exc()) 

        exec_time = time.time() - start_time
        
        theta_hat = theta_param.detach().cpu().numpy()
        x_hid_hat_0 = x_hid_0_param.detach().cpu().numpy()
        
        del net, optimizer, t_torch, x_obs_torch, t_phys_torch, theta_param, x_hid_0_param
        
        return theta_hat, x_hid_hat_0, exec_time