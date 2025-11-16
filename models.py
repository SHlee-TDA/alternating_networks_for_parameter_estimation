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

class HiddenVarPredictor(nn.Module):
    """f_theta: 숨겨진 변수를 예측하는 범용 네트워크"""
    def __init__(self, 
                 flat_x_dim,        # (T * num_features)
                 flat_y_dim,        # (T * num_hidden) 
                 num_params,
                 model_config,
                 use_spectral_norm=False,
                 initialization_config=None): 
        super().__init__()
        
        hidden_dims = model_config['hidden_dims']
        activation = get_activation(model_config['activation'])
        init_config = initialization_config or {'type': 'xavier', 'distribution': 'uniform'}
        
        layers = []
        input_dim = flat_x_dim + num_params
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, flat_y_dim))
        
        if use_spectral_norm:
            # 마지막 레이어에는 spectral_norm을 적용하지 않음
            layers[:-1] = [spectral_norm(l, n_power_iterations=15) if isinstance(l, nn.Linear) else l for l in layers[:-1]]
            

        self.network = nn.Sequential(*layers)
        self.network.apply(lambda m: init_weights_xavier(m, dist=init_config['distribution']))

    def forward(self, x_observed, params):
        # x_observed: (B, flat_x_dim), params: (B, num_params)
        # network : (B, flat_x_dim + num_params) -> (B, flat_y_dim)
        combined_input = torch.cat((x_observed, params), dim=1)
        return self.network(combined_input)

class ParameterEstimator(nn.Module):
    """g_phi: 파라미터를 예측하는 범용 네트워크"""
    def __init__(self, 
                 flat_x_dim,      # (T * n_features) 값
                 flat_y_dim,      # (T * n_hidden) 값
                 num_params,      # 파라미터 개수
                 model_config,    # config.MODEL_CONFIG['g_phi']
                 use_spectral_norm=False,
                 initialization_config=None):
        super().__init__()
        
        hidden_dims = model_config['hidden_dims']
        activation = get_activation(model_config['activation'])
        init_config = initialization_config or {'type': 'xavier', 'distribution': 'uniform'}

        # 입력: 평탄화된 X + 평탄화된 Y
        input_dim = flat_x_dim + flat_y_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim
            
        # 출력: 파라미터 개수
        last_layer = nn.Linear(input_dim, num_params)
        #layers.append(nn.Sigmoid()) # 출력을 [0, 1]로 제한
        
        if use_spectral_norm:
            layers = [spectral_norm(l, n_power_iterations=15) if isinstance(l, nn.Linear) else l for l in layers]
            
        self.network = nn.Sequential(*layers, last_layer)
        self.network.apply(lambda m: init_weights_xavier(m, dist=init_config['distribution']))
        

    def forward(self, x_observed, y_hidden):
        combined_input = torch.cat((x_observed, y_hidden), dim=1)
        return self.network(combined_input)