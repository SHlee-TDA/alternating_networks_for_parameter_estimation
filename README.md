# Convergence Without Correctness: Limits of Contractive Fixed-Point Parameter Estimation from Sparse Partial Observations

This repository contains the official PyTorch implementation accompanying the paper *"Convergence Without Correctness: Limits of Contractive Fixed-Point Parameter Estimation from Sparse Partial Observations"* (paper source: [`paper/simods/`](paper/simods/main.tex)).

Estimating parameters of nonlinear dynamical systems from sparse and partial observations is a severely ill-posed inverse problem. A natural strategy is to *decouple* it: learn one network ($H_\phi$) that reconstructs the unobserved (hidden) state from the observations and a current parameter guess, and another ($P_\psi$) that updates the parameter from the observations and the reconstructed state, then iterate the composition to a fixed point. This repository implements that decoupled fixed-point estimator and the experiments used to characterize what it can and cannot do.

## 🌟 Key Findings
- **Decoupled Iterative Inference**: separates hidden-state estimation ($H_\phi$) and parameter estimation ($P_\psi$) to avoid a highly non-convex joint optimization.
- **Guaranteed Convergence (Contraction Mapping)**: enforcing Lipschitz bounds via spectral normalization makes the composed update a contraction, guaranteeing convergence to a unique fixed point at a geometric rate from any initialization.
- **Convergence does not imply correctness**: under observational ambiguity, a strict contraction cannot preserve two distinct admissible ground truths as zero-residual fixed points — teacher-forced correctness and global contractivity are structurally incompatible on a non-identifiable fiber. Because the fixed point is a deterministic function of the observation, it cannot beat the conditional-mean (Bayes) estimator; reconstructing the hidden state confers no information advantage at inference.
- **A priori identifiability diagnosis**: Hermann–Krener Lie-derivative rank (SIR, Lotka–Volterra) and a sampled-sensitivity/scaling-symmetry ablation (OGTT) predict which parameters are recoverable *before* training; the predictions are corroborated empirically across all three benchmark systems.
- **Test-time noise reveals a stability–accuracy trade-off**: under a pre-registered observation-perturbation stress test, a matched-branch direct regressor is more accurate on clean data but degrades sharply and leaves the admissible parameter range under noise, while the contractive iterative estimator stays bounded at a higher error floor — this is stability, not superiority, as shown against a constant prior-mean control.

## 🏗️ Architecture Overview

### 📁 Repository Structure
We strictly adhere to object-oriented and modular design principles to ensure extensibility:
```
├── play.py                    # Interactive Web Dashboard (Easiest way to run)
├── main.py                    # Pipeline orchestrator (Phase 1 to Phase 3)
├── config.py                  # Configuration registry and CLI argument parser
├── src/
│   ├── data_loader.py         # ODE/SDE parallel simulation & dataset generation
│   ├── infer.py                # Alternating Inference Engine (Fixed-point iteration)
│   ├── analyzer.py             # BaseAnalyzer and System-specific visualization modules
│   ├── models.py                # H_phi (Hidden Net), P_psi (Param Net) & direct-regression baseline
│   ├── trainer.py               # Supervised training loop using teacher forcing
│   ├── losses.py                # Loss functions (teacher forcing, consistency)
│   └── utils.py                  # Normalizer, SDE solver, derivative estimators
├── systems/                    # Definitions for SIR, Lotka-Volterra, and OGTT models
├── experiments/det_meanlocus/  # Post-hoc analysis: identifiability, contraction certification,
│                               # conditional-mean locus, and the test-time noise stress test
└── paper/simods/               # SIAM-style paper source (main.tex)
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
We evaluate the decoupled estimator against a matched-branch direct regressor on three systems chosen to
span the identifiability spectrum. Each system's local identifiability is diagnosed *a priori* (Hermann–Krener
Lie-derivative rank for SIR/Lotka–Volterra; a sampled-sensitivity analysis with a scaling-symmetry ablation
for OGTT) and corroborated empirically. The `src/analyzer.py` module generates the evaluation plots; the
identifiability diagnostics and the noise stress test are reproduced by standalone scripts under
`experiments/det_meanlocus/`.

1. **Epidemiological Model (SIR)** — locally observable from the susceptible trajectory alone. Both
   estimators recover $(\beta,\gamma)$ reasonably well; the residual gap between them isolates the
   *operator-approximation* cost of the decoupled construction rather than an identifiability limit.
2. **Ecological Model (Lotka–Volterra)** — structurally *not* observable from prey-only trajectories
   (Hermann–Krener rank $5<6$): an explicit joint $(y_0,\beta)$ scaling symmetry leaves $\beta$
   unrecoverable, while $\alpha,\delta,\gamma$ are recovered almost exactly by both estimators.
3. **Physiological Model (OGTT)** — weakly identifiable from a sparse 5-point glucose record: a
   sampled-sensitivity analysis restricted to the data-generating manifold shows the record resolves only
   the product $S_I\sigma$ well, with a $\sim\!10^3$-fold weaker scaling direction. Both estimators collapse
   toward this product ridge, exactly as predicted a priori.

We additionally run a pre-registered test-time observation-perturbation sweep (noiseless training, noisy
evaluation) comparing the iterative estimator, the direct regressor, and a constant prior-mean predictor —
see `experiments/det_meanlocus/noise_stress_test.py` and the corresponding section of the paper for the full
stability–accuracy analysis.

# 📝 Citation
If you find this work useful in your research, please consider citing our paper:
```
@article{lee2026convergence,
  title={Convergence Without Correctness: Limits of Contractive Fixed-Point Parameter Estimation from Sparse Partial Observations},
  author={Lee, Seong-Heon and Lee, Dongjin and Gu, Jiaxi and Ha, Joon and Jung, Jae-Hun},
  journal={arXiv preprint},
  year={2026}
}
```
