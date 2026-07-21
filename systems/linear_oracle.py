import numpy as np
import torch
from .base_system import System


class LinearOracle(System):
    """
    G0 sanity-oracle system.

    A linear, fully identifiable 2-state ODE used purely to validate that the
    machinery (data pipeline, iterative operator, NLS baseline) can recover
    ground-truth parameters to high precision on a clean, well-posed problem.

        dx/dt = -a * x + y      (observed)
        dy/dt = -b * y          (hidden)

    Closed form (x0 = 1):
        x(t) = (1 - y0/(a-b)) e^{-a t} + (y0/(a-b)) e^{-b t}

    So x(t) is a sum of two decaying exponentials with distinct rates a != b and
    amplitudes fixed by (a, b, y0). Observing x on a time grid identifies
    (a, b, y0) whenever a != b, hence the pair (a, b) is fully identifiable.

    Design notes:
    - Param ranges are chosen inside (-1, 1). For non-OGTT systems the pipeline
      runs with use_normalization=False, and P_psi ends in a Tanh that bounds its
      raw output to [-1, 1]; keeping targets inside that interval avoids the cap.
    - Rates are well separated (a in [0.1, 0.35], b in [0.6, 0.9], gap >= 0.25)
      so the two exponentials are distinguishable and the map x_obs -> (a, b) is
      well conditioned.
    - Deterministic ODE only (no SDE / noise) so both estimators can, in
      principle, reach <1e-3 relative error.
    """
    name = 'linear_oracle'
    param_names = ['a', 'b']

    param_ranges = {'a': [0.10, 0.35], 'b': [0.60, 0.90]}

    # x0 fixed at 1.0, y0 sampled in [0.5, 1.5] (see sample_initial_conditions)
    initial_conditions = ([1.0], [1.0])

    t_span = [0, 5]
    t_points = np.linspace(0, 5, 8)

    observed_var_idx = 0  # x
    hidden_var_idx = 1    # y

    def sample_initial_conditions(self, params_dict):
        x0 = 1.0
        y0 = np.random.uniform(0.5, 1.5)
        return [x0, y0]

    @staticmethod
    def ode_func(t, y, params):
        x, yv = y
        a, b = params
        dxdt = -a * x + yv
        dydt = -b * yv
        return [dxdt, dydt]

    @staticmethod
    def ode_func_torch(u, theta):
        """
        u: (Batch, 2) -> [x, y]
        theta: (2,) -> [a, b]
        """
        x, yv = u[:, 0], u[:, 1]
        a, b = theta[0], theta[1]
        dxdt = -a * x + yv
        dydt = -b * yv
        return torch.stack([dxdt, dydt], dim=1)
