# losses.py
"""
This script containts abstract class of loss functions used for training the models.
It provides a base structure for implementing various loss functions that can be extended as needed.
"""

import torch
import torch.nn as nn

class BaseLoss(nn.Module):
    """
    Abstract base class for loss functions.
    All custom loss functions should inherit from this class and implement the forward method.
    """
    def __init__(self, hidden_net, param_net, config):
        super(BaseLoss, self).__init__()
        self.H_phi = hidden_net
        self.P_psi = param_net
        self.config = config
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x_true, y_true, p_true):
        """
        Compute the loss given the true inputs and outputs.
        
        Args:
            x_true (torch.Tensor): Input observed variable.
            y_true (torch.Tensor): True hidden variables.
            p_true (torch.Tensor): True parameters.
        
        Returns:
            total_loss (torch.Tensor): Computed loss value.
            metrics (dict): Dictionary of individual loss components for logging.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")
    
class SupervisedLoss(BaseLoss):
    """
    Supervised loss function that computes MSE loss for both hidden variables and parameters.
    """
    def forward(self, x_true, y_true, p_true):
        # Predict hidden variables using H_phi
        y_pred = self.H_phi(x_true, p_true)
        hidden_state_loss = self.mse_loss(y_pred, y_true)
        
        # Predict parameters using P_psi
        p_pred = self.P_psi(x_true, y_true)
        param_loss = self.mse_loss(p_pred, p_true)
        
        total_loss = hidden_state_loss + param_loss
        
        metrics = {
            'hidden_state_loss': hidden_state_loss.item(),
            'param_loss': param_loss.item()
        }
        
        return total_loss, metrics
    
class RecurrentLoss(BaseLoss):
    """
    Recurrent loss function that computes multi-step prediction loss for parameters.
    """
    def __init__(self, hidden_net, param_net, config, iter=2, gamma=1.0):
        super(RecurrentLoss, self).__init__(hidden_net, param_net, config)
        self.iter = iter # Number of recurrent steps
        self.gamma = gamma # Discount factor
        
    def forward(self, x_true, y_true, p_true):
        total_loss = torch.tensor(0.0, device=x_true.device)
        metrics = {}
        
        p_curr = p_true
        
        first_step_loss = None
        final_step_loss = None
        
        # Unrolled Optimization Loop (BPTT)
        for k in range(1, self.iter + 1):
            y_pred = self.H_phi(x_true, p_curr)
            p_next = self.P_psi(x_true, y_pred)
            step_loss = self.mse_loss(p_next, p_true)
            
            weight = self.gamma ** (k - 1)
            total_loss += step_loss * weight
            
            if k == 1:
                first_step_loss = step_loss.item()
            if k == self.iter:
                final_step_loss = step_loss.item()
            
            p_curr = p_next
        
        metrics['loss_recur'] = total_loss.item()
        metrics['loss_recur_first'] = first_step_loss
        metrics['loss_recur_final'] = final_step_loss 
        
        return total_loss, metrics
    
class CompositeLoss(BaseLoss):
    def __init__(self, hidden_net, param_net, config, components_with_weights):
        super(CompositeLoss, self).__init__(hidden_net, param_net, config)
        
        # components_with_weights: [(module, weight), (module, weight), ...]
        self.components = nn.ModuleList([c for c, w in components_with_weights])
        self.weights = [w for c, w in components_with_weights]
        
    def forward(self, x_true, y_true, p_true):
        total_loss = 0
        merged_metrics = {}
        
        for component, weight in zip(self.components, self.weights):
            loss, metrics = component(x_true, y_true, p_true)
            
            total_loss += loss * weight
            
            merged_metrics.update(metrics)
        
        merged_metrics['total_loss'] = total_loss.item()
            
        return total_loss, merged_metrics
    
# Factory function
def get_loss_function(hidden_net, param_net, config):
    loss_config = getattr(config, 'LOSS_CONFIG', [('supervised', 1.0)])
    
    component_map = {
        'supervised': SupervisedLoss,
        'recurrent': RecurrentLoss
    }
    
    active_components_with_weights = []
    
    for item in loss_config:
        if isinstance(item, tuple) or isinstance(item, list):
            name, weight = item
        else:
            name, weight = item, 1.0 
            
        if name not in component_map:
            raise ValueError(f"Unknown loss type: {name}")
            
        comp_instance = component_map[name](hidden_net, param_net, config)
        active_components_with_weights.append((comp_instance, weight))
        
    return CompositeLoss(hidden_net, param_net, config, active_components_with_weights)