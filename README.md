# An Iterative Network for Parameter Estimation in Nonlinear Dynamical Systems from Sparse and Partial Observations
This repository contains the official PyTorch implementation of *"An Iterative Network for Parameter Estimation in Nonlinear Dynamical Systems from Sparse and Partial Observations"*.
Estimating parameters from sparse and partial observations is a highly ill-posed inverse problem. 
Traditional joint optimization often yields a non-convex loss landscape, causing solvers to diverge or trap in local minima. 
To overcome this, we decouple the joint inverse problem into two conditional tasks and formulate the inference as an alternating update between two trained neural networks ($H_\phi$ and $P_\psi$).

## 🌟 Key Contributions
- **Decoupled Iterative Inference**: Separates hidden-state estimation ($H_\phi$) and parameter estimation ($P_\psi$) to avoid highly non-convex joint optimization.
- **Guaranteed Convergence (Contraction Mapping)**: By enforcing Lipschitz bounds via Spectral Normalization, we mathematically guarantee that the composition of the two networks forms a contraction mapping, ensuring exponential convergence to a unique fixed point.
- **Teacher-Forced Supervised Learning**: Aligns the unique fixed point with the ground-truth parameter during the training phase.

## 🏗️ Architecture Overview

### 📁 Repository Structure
We strictly adhere to object-oriented and modular design principles to ensure extensibility:
```
├── main.py                # Pipeline orchestrator (Phase 1 to Phase 3)
├── data_loader.py         # ODE/SDE parallel simulation & dataset generation
├── infer.py               # Alternating Inference Engine (Fixed-point iteration)
├── analyzer.py            # BaseAnalyzer and System-specific visualization modules
├── models.py              # H_phi (Hidden Net) & P_psi (Param Net) with Spectral Norm
├── trainer.py             # Supervised training loop using teacher forcing
└── systems/               # Definitions for SIR, Lotka-Volterra, and OGTT models
```

### 🚀 Getting Started
1. Installation
    ```
    pip install torch numpy scipy pandas matplotlib seaborn tqdm scikit-learn
    ```
2. Run Experiments
The entire pipeline (Data Generation $\rightarrow$ Training $\rightarrow$ Inference & Evaluation) can be executed with a single command:
    ```
    # Run the pipeline with default configuration
    python main.py

    # Override epochs for a quick test
    python main.py --epochs 100
    ```

## 🧪 Benchmark Dynamical Systems & Results
We evaluate our framework on three diverse dynamical systems. 
The `analyzer.py` module automatically generates the following evaluation plots.
1. **Epidemiological Model (SIR)** 
    
    Demonstrates the network's ability to robustly recover the basic reproduction number ($\mathcal{R}_0$) across the epidemic bifurcation threshold.
2. **Ecological Model (Lotka-Volterra)**

    Challenges the network to implicitly reconstruct the global nonlinear invariant manifold solely from highly aliased, sparse local observations.
3. **Physiological Model (OGTT)**

    Applied to the Oral Glucose Tolerance Test (OGTT) model to highlight practical non-identifiability issues when relying on glucose-only observations.
    
    
# 📝 Citation
If you find this work useful in your research, please consider citing our paper:
```
@article{lee2026iterative,
  title={An Iterative Network for Parameter Estimation in Nonlinear Dynamical Systems from Sparse and Partial Observations},
  author={Lee, Seong-Heon and Lee, Dongjin and Gu, Jiaxi and Ha, Joon and Jung, Jae-Hun},
  journal={arXiv preprint},
  year={2026}
}
```