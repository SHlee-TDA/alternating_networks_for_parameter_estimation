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
    def __init__(self, f_theta, g_phi, config):
        super(BaseLoss, self).__init__()
        self.f_theta = f_theta
        self.g_phi = g_phi
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
        # Predict hidden variables using f_theta
        y_pred = self.f_theta(x_true, p_true)
        loss_f = self.mse_loss(y_pred, y_true)
        
        # Predict parameters using g_phi
        p_pred = self.g_phi(x_true, y_true)
        loss_g = self.mse_loss(p_pred, p_true)
        
        total_loss = loss_f + loss_g
        
        metrics = {
            'loss_f': loss_f.item(),
            'loss_g': loss_g.item()
        }
        
        return total_loss, metrics
    
class CompositeLoss(BaseLoss):
    def __init__(self, f_theta, g_phi, config, components_with_weights):
        super(CompositeLoss, self).__init__(f_theta, g_phi, config)
        
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
def get_loss_function(f_theta, g_phi, config):
    loss_config = getattr(config, 'LOSS_CONFIG', [('supervised', 1.0)])
    
    component_map = {
        'supervised': SupervisedLoss,
        #'consistency': ConsistencyComponent,
        #'recurrent': RecurrentComponent
    }
    
    active_components_with_weights = []
    
    for item in loss_config:
        if isinstance(item, tuple) or isinstance(item, list):
            name, weight = item
        else:
            name, weight = item, 1.0 
            
        if name not in component_map:
            raise ValueError(f"Unknown loss type: {name}")
            
        comp_instance = component_map[name](f_theta, g_phi, config)
        active_components_with_weights.append((comp_instance, weight))
        
    return CompositeLoss(f_theta, g_phi, config, active_components_with_weights)