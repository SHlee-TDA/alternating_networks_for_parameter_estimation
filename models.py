# models.py
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def get_activation(name):
    """문자열 이름으로부터 활성화 함수 객체를 반환합니다."""
    if name == 'Tanh':
        return nn.Tanh()
    elif name == 'ReLU':
        return nn.ReLU()
    elif name == 'SiLU':
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def init_weights_xavier(m, dist='uniform'):
    """
    nn.Linear 레이어의 가중치를 Xavier Uniform 방식으로 초기화하고,
    편향은 0으로 초기화합니다.
    """
    if isinstance(m, nn.Linear):
        if dist == 'uniform':
            torch.nn.init.xavier_uniform_(m.weight)
        elif dist == 'normal':
            torch.nn.init.xavier_normal_(m.weight)
        else:
            raise ValueError(f"Unknown distribution for Xavier initialization: {dist}")
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    """Skip Connection + LayerNorm이 적용된 블록"""
    def __init__(self, hidden_dim, activation, use_layer_norm=False, use_spectral_norm=False, dropout_rate=0.0):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        # 선형 레이어
        linear = nn.Linear(hidden_dim, hidden_dim)
        if use_spectral_norm:
            linear = spectral_norm(linear)
        self.linear = linear
        
        # 활성화 함수
        self.activation = activation
        
        # Layer Norm (선택)
        if use_layer_norm:
            self.ln = nn.LayerNorm(hidden_dim)
            
        # Dropout (선택 - 현재는 0.0 권장)
        #self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. Linear -> Activation
        out = self.linear(x)
        if self.use_layer_norm:
            out = self.ln(out)
        out = self.activation(out)
        #out = self.dropout(out)
        
        # 2. Skip Connection (x + F(x))
        return x + out
    

class BaseNetwork(nn.Module):
    """f_theta와 g_phi가 공유하는 기본 네트워크 구조"""
    def __init__(self, input_dim, output_dim, model_config, use_spectral_norm):
        super().__init__()
        hidden_dims = model_config['hidden_dims']
        activation = get_activation(model_config['activation'])
        
        layers = []
        
        # 1. Input Layer (차원 맞추기: Input -> Hidden)
        first_layer = nn.Linear(input_dim, hidden_dims[0])
        if use_spectral_norm: first_layer = spectral_norm(first_layer)
        layers.append(first_layer)
        layers.append(activation)
        
        # 2. Hidden Layers (Residual Blocks)
        # Skip Connection을 쓰려면 입력/출력 차원이 같아야 하므로 Hidden끼리 연결
        for i in range(len(hidden_dims) - 1):
            # 차원이 같을 때만 Residual Block 사용 가능
            if hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ResidualBlock(
                    hidden_dims[i], 
                    activation, 
                    use_layer_norm=True,  # [추천] LayerNorm 켜기
                    use_spectral_norm=use_spectral_norm
                ))
            else:
                # 차원이 다르면 일반 Linear 사용 (Skip Connection 불가)
                l = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                if use_spectral_norm: l = spectral_norm(l)
                layers.append(l)
                layers.append(activation)
                
        # 3. Output Layer
        last_layer = nn.Linear(hidden_dims[-1], output_dim)
        if use_spectral_norm: last_layer = spectral_norm(last_layer)
        layers.append(last_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class HiddenVarPredictor(nn.Module):
    def __init__(self, flat_x_dim, flat_y_dim, num_params, model_config, use_spectral_norm=False, initialization_config=None):
        super().__init__()
        input_dim = flat_x_dim + num_params
        output_dim = flat_y_dim
        
        self.net = BaseNetwork(input_dim, output_dim, model_config, use_spectral_norm)
        # (초기화 로직은 그대로 적용하거나 BaseNetwork 내부로 이동 가능)

    def forward(self, x, p):
        combined = torch.cat([x, p], dim=1)
        return self.net(combined)

class ParameterEstimator(nn.Module):
    def __init__(self, flat_x_dim, flat_y_dim, num_params, model_config, use_spectral_norm=False, initialization_config=None):
        super().__init__()
        input_dim = flat_x_dim + flat_y_dim
        output_dim = num_params
        
        self.net = BaseNetwork(input_dim, output_dim, model_config, use_spectral_norm)
        self.final_act = nn.Tanh() # [필수] 정규화된 파라미터 범위

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        return self.final_act(self.net(combined))
    
# class HiddenVarPredictor(nn.Module):
#     """f_theta: 숨겨진 변수를 예측하는 범용 네트워크"""
#     def __init__(self, 
#                  flat_x_dim,        # (T * num_features)
#                  flat_y_dim,        # (T * num_hidden) 
#                  num_params,
#                  model_config,
#                  use_spectral_norm=False,
#                  initialization_config=None): 
#         super().__init__()
        
#         hidden_dims = model_config['hidden_dims']
#         activation = get_activation(model_config['activation'])
#         init_config = initialization_config or {'type': 'xavier', 'distribution': 'uniform'}
        
#         layers = []
#         input_dim = flat_x_dim + num_params
        
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             layers.append(activation)
#             input_dim = hidden_dim
            
#         layers.append(nn.Linear(input_dim, flat_y_dim))
        
#         if use_spectral_norm:
#             # 마지막 레이어에는 spectral_norm을 적용하지 않음
#             layers[:-1] = [spectral_norm(l, n_power_iterations=15) if isinstance(l, nn.Linear) else l for l in layers[:-1]]
            

#         self.network = nn.Sequential(*layers)
#         self.network.apply(lambda m: init_weights_xavier(m, dist=init_config['distribution']))

#     def forward(self, x_observed, params):
#         # x_observed: (B, flat_x_dim), params: (B, num_params)
#         # network : (B, flat_x_dim + num_params) -> (B, flat_y_dim)
#         combined_input = torch.cat((x_observed, params), dim=1)
#         return self.network(combined_input)

# class ParameterEstimator(nn.Module):
#     """g_phi: 파라미터를 예측하는 범용 네트워크"""
#     def __init__(self, 
#                  flat_x_dim,      # (T * n_features) 값
#                  flat_y_dim,      # (T * n_hidden) 값
#                  num_params,      # 파라미터 개수
#                  model_config,    # config.MODEL_CONFIG['g_phi']
#                  use_spectral_norm=False,
#                  initialization_config=None):
#         super().__init__()
        
#         hidden_dims = model_config['hidden_dims']
#         activation = get_activation(model_config['activation'])
#         init_config = initialization_config or {'type': 'xavier', 'distribution': 'uniform'}

#         # 입력: 평탄화된 X + 평탄화된 Y
#         input_dim = flat_x_dim + flat_y_dim
        
#         layers = []
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             layers.append(activation)
#             input_dim = hidden_dim
            
#         # 출력: 파라미터 개수
#         layers.append(nn.Linear(input_dim, num_params))
#         layers.append(nn.Tanh())
#         #layers.append(nn.Sigmoid()) # 출력을 [0, 1]로 제한
        
#         if use_spectral_norm:
#             layers = [spectral_norm(l, n_power_iterations=15) if isinstance(l, nn.Linear) else l for l in layers]
            
#         self.network = nn.Sequential(*layers)
#         self.network.apply(lambda m: init_weights_xavier(m, dist=init_config['distribution']))
        

#     def forward(self, x_observed, y_hidden):
#         combined_input = torch.cat((x_observed, y_hidden), dim=1)
#        return self.network(combined_input)
    
    