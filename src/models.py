# models.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# Helper functions
def get_activation(name):
    """Returns the corresponding PyTorch activation module."""
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
    Orthogonal Initialization for Spectral Normalized Networks.
    Achieves the 'Edge of Chaos' initialization state to ensure signal 
    propagation without vanishing or exploding gradients.
    """
    if isinstance(m, nn.Linear):
        # Set gain based on the activation function to preserve signal variance
        if activation in ['ReLU', 'SiLU', 'ELU']:
            gain = torch.nn.init.calculate_gain('relu') # sqrt(2) approx 1.414
        elif activation == 'Tanh':
            gain = torch.nn.init.calculate_gain('tanh') # 5/3 approx 1.67
        else:
            gain = 1.0

        # Orthogonal init
        target_weight = getattr(m, 'weight_orig', m.weight)
        torch.nn.init.orthogonal_(target_weight, gain=gain)
        
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.constant_(m.bias, 0)
                
class ExcludeLambda(nn.Module):
    """
    Scales the output by a constant factor.
    
    Note:
        Used to enforce a strict contraction mapping (Lipschitz constant < 1) 
        and to compensate for numerical approximation errors in spectral norm.
    """   
    def __init__(self, scale=0.95):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return x * self.scale
  
# Proposed Architectures
class HiddenVarPredictor(nn.Module):
    """
    H_phi: Predicts hidden variables (Y) from observed variables (X) and parameters (P).
    
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
        spectral_scale = model_config.get('spectral_scale', 0.99)
        
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
    P_psi: Estimates parameters (P) from observed (X) and hidden variables (Y).
    
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
        spectral_scale = model_config.get('spectral_scale', 0.99)

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

        # Enforce range [-1, 1] for normalized parameters
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.network.apply(lambda m: init_weights(m, activation))

    def forward(self, x_observed, y_hidden):
        combined_input = torch.cat((x_observed, y_hidden), dim=1)
        return self.network(combined_input)


# Baseline Architecture
class SingleNetworkBaseline(nn.Module):
    """
    Baseline model that directly maps observed variables (X) to hidden variables (Y) 
    without parameter estimation.
    
    Note:
        This serves as a simple benchmark to evaluate the benefits of the 
        proposed architecture.
    """
    def __init__(self, flat_x_dim, flat_y_dim, model_config, use_spectral_norm=False):
        super().__init__()
        
        hidden_dims = model_config['hidden_dims']
        activation = get_activation(model_config['activation'])
        
        layers = []
        input_dim = flat_x_dim
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

    def forward(self, x_observed):
        return self.network(x_observed)

