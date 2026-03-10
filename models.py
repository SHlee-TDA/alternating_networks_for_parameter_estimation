# models.py
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def get_activation(name):
    if name == 'Tanh' or name == 'tanh':
        return nn.Tanh()
    elif name == 'ReLU' or name == 'relu':
        return nn.ReLU()
    elif name == 'SiLU' or  name == 'silu':
        return nn.SiLU()
    elif name == 'Sigmoid' or name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def init_weights(m, activation='Tanh'):
    """
    Initializes weights based on the activation type.
    
    Note:
        - Tanh/Sigmoid: Xavier Uniform
        - ReLU/SiLU: Kaiming Normal
        - Bias: 0
    """
    # if isinstance(m, nn.Linear):
    #     if activation in ['Tanh', 'Sigmoid']:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias, 0)
                
    #     elif activation in ['ReLU', 'SiLU', 'Softplus']:
    #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias, 0)
    
    """
    Orthogonal Initialization for Spectral Normalized Networks.
    Achieves 'Edge of Chaos' initialization state.
    """
    if isinstance(m, nn.Linear):
        # 1. Gain 설정 (활성화 함수에 맞게 신호 보존)
        if activation in ['ReLU', 'SiLU', 'ELU']:
            # 이론적으로 sqrt(2)지만, SN 환경에선 조금 더 공격적으로 잡음
            gain = torch.nn.init.calculate_gain('relu') # sqrt(2) approx 1.414
        elif activation == 'Tanh':
            gain = torch.nn.init.calculate_gain('tanh') # 5/3 approx 1.67
        else:
            gain = 1.0

        # 2. Orthogonal Initialization
        target_weight = getattr(m, 'weight_orig', m.weight)
        torch.nn.init.orthogonal_(target_weight, gain=gain)
        
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.constant_(m.bias, 0)
                
class ExcludeLambda(nn.Module):
    """
    Scales the output by a constant factor.
    
    Note:
        Used to enforce strict contraction mapping (Lipschitz constant < 1) 
        and compensate for numerical approximation errors in spectral norm.
    """    
    def __init__(self, scale=0.95):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale
  
            
class HiddenVarPredictor(nn.Module):
    """
    f_theta: Predicts hidden variables (Y) from observed variables (X) and parameters (P).
    
    Args:
        flat_x_dim (int): Dimension of flattened observed variables.
        flat_y_dim (int): Dimension of flattened hidden variables.
        num_params (int): Number of system parameters.
        model_config (dict): Configuration for hidden dims and activation.
        use_spectral_norm (bool): Whether to apply spectral normalization.
    """
    def __init__(self, 
                 flat_x_dim, 
                 flat_y_dim, 
                 num_params,
                 model_config,
                 use_spectral_norm=False,
                 ): 
        super().__init__()
        
        hidden_dims = model_config['hidden_dims']
        activation = get_activation(model_config['activation'])
        
        layers = []
        input_dim = flat_x_dim + num_params
        spectral_scale = 0.99
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(input_dim, hidden_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear, n_power_iterations=1)
            layers.append(linear)
            
            if use_spectral_norm:
                layers.append(ExcludeLambda(spectral_scale))
                
            layers.append(activation)
            input_dim = hidden_dim
            
        # Output Layer
        last_linear = nn.Linear(input_dim, flat_y_dim)
        if use_spectral_norm:
            last_linear = spectral_norm(last_linear, n_power_iterations=1)
        layers.append(last_linear)
                
        self.network = nn.Sequential(*layers)
        self.network.apply(lambda m: init_weights(m, activation))

    def forward(self, x_observed, params):
        combined_input = torch.cat((x_observed, params), dim=1)
        return self.network(combined_input)

class ParameterEstimator(nn.Module):
    """
    g_phi: Estimates parameters (P) from observed (X) and hidden variables (Y).
    
    Note:
        The output is bounded by Tanh to match the normalized parameter range [-1, 1].
    """    
    def __init__(self, 
                 flat_x_dim, 
                 flat_y_dim,
                 num_params, 
                 model_config, 
                 use_spectral_norm=False,
                 ):
        super().__init__()
        
        hidden_dims = model_config['hidden_dims']
        activation = get_activation(model_config['activation'])

        input_dim = flat_x_dim + flat_y_dim
        layers = []
        spectral_scale = 0.99

        for hidden_dim in hidden_dims:
            linear = nn.Linear(input_dim, hidden_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear, n_power_iterations=1)
            layers.append(linear)
            
            if use_spectral_norm:
                layers.append(ExcludeLambda(spectral_scale))
                
            layers.append(activation)
            input_dim = hidden_dim
            
        # Output Layer
        final_linear = nn.Linear(input_dim, num_params)
        if use_spectral_norm:
            final_linear = spectral_norm(final_linear, n_power_iterations=1)
        layers.append(final_linear)
        
        # FIX: 2025-12-15 
        # if use_spectral_norm:
        #      layers.append(ExcludeLambda(spectral_scale))
        
        # Enforce range [-1, 1]
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.network.apply(lambda m: init_weights(m, activation))

    def forward(self, x_observed, y_hidden):
        combined_input = torch.cat((x_observed, y_hidden), dim=1)
        return self.network(combined_input)

                


# class ResidualBlock(nn.Module):
#     """Skip Connection + LayerNorm이 적용된 블록"""
#     def __init__(self, hidden_dim, activation, use_layer_norm=False, use_spectral_norm=False, dropout_rate=0.0):
#         super().__init__()
#         self.use_layer_norm = use_layer_norm
        
#         # 선형 레이어
#         linear = nn.Linear(hidden_dim, hidden_dim)
#         if use_spectral_norm:
#             linear = spectral_norm(linear)
#         self.linear = linear
        
#         # 활성화 함수
#         self.activation = activation
        
#         # Layer Norm (선택)
#         if use_layer_norm:
#             self.ln = nn.LayerNorm(hidden_dim)
            
#         # Dropout (선택 - 현재는 0.0 권장)
#         #self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         # 1. Linear -> Activation
#         out = self.linear(x)
#         if self.use_layer_norm:
#             out = self.ln(out)
#         out = self.activation(out)
#         #out = self.dropout(out)
        
#         # 2. Skip Connection (x + F(x))
#         return x + out
    

# class BaseNetwork(nn.Module):
#     """f_theta와 g_phi가 공유하는 기본 네트워크 구조"""
#     def __init__(self, input_dim, output_dim, model_config, use_spectral_norm):
#         super().__init__()
#         hidden_dims = model_config['hidden_dims']
#         activation = get_activation(model_config['activation'])
        
#         layers = []
        
#         # 1. Input Layer (차원 맞추기: Input -> Hidden)
#         first_layer = nn.Linear(input_dim, hidden_dims[0])
#         if use_spectral_norm: first_layer = spectral_norm(first_layer)
#         layers.append(first_layer)
#         layers.append(activation)
        
#         # 2. Hidden Layers (Residual Blocks)
#         # Skip Connection을 쓰려면 입력/출력 차원이 같아야 하므로 Hidden끼리 연결
#         for i in range(len(hidden_dims) - 1):
#             # 차원이 같을 때만 Residual Block 사용 가능
#             if hidden_dims[i] == hidden_dims[i+1]:
#                 layers.append(ResidualBlock(
#                     hidden_dims[i], 
#                     activation, 
#                     use_layer_norm=True,  
#                     use_spectral_norm=use_spectral_norm
#                 ))
#             else:
#                 # 차원이 다르면 일반 Linear 사용 (Skip Connection 불가)
#                 l = nn.Linear(hidden_dims[i], hidden_dims[i+1])
#                 if use_spectral_norm: l = spectral_norm(l)
#                 layers.append(l)
#                 layers.append(activation)
                
#         # 3. Output Layer
#         last_layer = nn.Linear(hidden_dims[-1], output_dim)
#         if use_spectral_norm: last_layer = spectral_norm(last_layer)
#         layers.append(last_layer)
        
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)
    

# class HiddenVarPredictor(nn.Module):
#     def __init__(self, flat_x_dim, flat_y_dim, num_params, model_config, use_spectral_norm=False, initialization_config=None):
#         super().__init__()
#         input_dim = flat_x_dim + num_params
#         output_dim = flat_y_dim
        
#         self.net = BaseNetwork(input_dim, output_dim, model_config, use_spectral_norm)
#         # (초기화 로직은 그대로 적용하거나 BaseNetwork 내부로 이동 가능)

#     def forward(self, x, p):
#         combined = torch.cat([x, p], dim=1)
#         return self.net(combined)

# class ParameterEstimator(nn.Module):
#     def __init__(self, flat_x_dim, flat_y_dim, num_params, model_config, use_spectral_norm=False, initialization_config=None):
#         super().__init__()
#         input_dim = flat_x_dim + flat_y_dim
#         output_dim = num_params
        
#         self.net = BaseNetwork(input_dim, output_dim, model_config, use_spectral_norm)
#         self.final_act = nn.Tanh() # [필수] 정규화된 파라미터 범위
#         #self.final_act = nn.Softplus()
#         self._initialize_last_layer()
        
#     def forward(self, x, y):
#         combined = torch.cat([x, y], dim=1)
#         return self.final_act(self.net(combined))
    
#     def _initialize_last_layer(self):
#         # 네트워크의 마지막 Linear 레이어를 찾아 Bias를 0.5로 설정
#         # (BaseNetwork -> Sequential -> ... -> Linear)
#         last_layer = None
#         for module in self.net.modules():
#             if isinstance(module, nn.Linear):
#                 last_layer = module
        
#         if last_layer is not None:
#             print(f"[ParameterEstimator] Initializing output bias to 0.5")
#             nn.init.constant_(last_layer.bias, 0.5)
#             # 가중치는 작게 하여 초기 출력이 Bias에 의존하도록 함
#             nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)