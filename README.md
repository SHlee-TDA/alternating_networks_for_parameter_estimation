# Alternating Networks for Parameter Estimation in Dynamical Systems

This repository contains the official implementation of Alternating Networks for Parameter Estimation, a neural framework designed to simultaneously infer hidden states and estimate governing parameters in nonlinear dynamical systems.

## 🌟 Key Features

- **Dual-Network Architecture**: Implements a coupled system of $H_\phi$ (Hidden State Estimator) and $P_\psi$ (Parameter Estimator) to handle partially observed dynamics.
- **Alternating Minimization**: A robust training protocol that iterates between latent state reconstruction and parameter optimization.


 
├── main.py                # Main pipeline (Setup -> Data -> Train -> Eval)
├── config.py              # Centralized hyperparameter & experiment registry
├── data_loader.py         # SDE/ODE generation and clinical data (NIH) loading
├── models.py              # $f_\theta$ and $g_\phi$ architectures with Spectral Norm
├── trainer.py             # Alternating minimization training loop logic
├── analyzer.py            # Phase portraits, loss curves, and clinical validation
├── utils.py               # Euler-Maruyama solver and data normalizers
├── systems/               # Dynamical system definitions (Lotka-Volterra, SIR, OGTT)
└── docs/                  # Technical documentation and mathematical rationales