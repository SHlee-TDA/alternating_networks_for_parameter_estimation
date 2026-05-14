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
├── play.py                # Interactive Web Dashboard (Easiest way to run)
├── main.py                # Pipeline orchestrator (Phase 1 to Phase 3)
├── config.py              # Configuration registry and CLI argument parser
├── data_loader.py         # ODE/SDE parallel simulation & dataset generation
├── infer.py               # Alternating Inference Engine (Fixed-point iteration)
├── analyzer.py            # BaseAnalyzer and System-specific visualization modules
├── models.py              # H_phi (Hidden Net) & P_psi (Param Net) with Spectral Norm
├── trainer.py             # Supervised training loop using teacher forcing
└── systems/               # Definitions for SIR, Lotka-Volterra, and OGTT models
```

### 🚀 Getting Started
1. Environment Setup
We strongly recommend using an isolated virtual environment to avoid version conflicts.
- Using `conda` (Recommended):
    ```
    conda create -n param_estim python=3.9 -y
    conda activate param_estim
    ```
- Using Python `venv`:
    ```
    python -m venv venv
    source venv/bin/activate  
    # On Windows use: venv\Scripts\activate
    ```
2. Install Dependencies

    ```
    pip install -r requirements.txt
    ```

#### 💻 Quick Start: Interactive Web Dashboard (Recommended)
We provide a user-friendly Web GUI powered by *Gradio*. 
This is the easiest way to explore the framework without digging into the codebase.

```
python play.py
```

After running the command, open the provided local URL (e.g., http://127.0.0.1:7860) in your web browser.

**How to use the dashboard**:

- Select the target dynamical system (SIR, Lotka-Volterra, OGTT).

- Set basic hyperparameters (Epochs, Batch Size, LR, Seed).

- Configure advanced data features like Derivative Methods (Lagrangian, Spline, ).

- Design your neural network architecture (Hidden layers & Activations) or toggle Spectral Normalization.

- Click 'Run Experiment' and monitor the live logs and evaluation plots directly on your screen!

#### 🛠️ Advanced Usage: Programmatic Control

For deeper, fine-grained control over the experiments (e.g., batch processing, custom loss weighting, or remote server execution), you can directly interact with the orchestrator.

For the highest level of control, you can directly modify `config.py`.
This allows you to set up complex multi-scenario experiments, modify data-generation bounds, or change the default logging behaviors.

```
python main.py
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