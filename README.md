# Project Structure

```bash
├── main.py                # Main experiment pipeline (Setup -> Data -> Train -> Eval)
├── config.py              # Configuration hyperparameters
├── data_loader.py         # Synthetic data generation & Real data loading
├── models.py              # Neural network architectures
├── trainer.py             # Alternating minimization training loop
├── analyzer.py            # Evaluation & Visualization suite
├── utils.py               # Utilities
├── data/                  # Store cached simulation data sets and real data (NIH)
└── systems/               # Dynamical system definitions
    ├── ogtt_simul.py      # OGTT ODE/SDE implementation
    ├── 

```

# Dependencies

```bash
pip install torch numpy scipy pandas matplotlib seaborn tqdm
```


# Running Experiments
The `main.py` script handles the entire pipeline:
- Phase 1: Generates synthetic training data (cached for reuse).
- Phase 2: Trains $f_\theta$ and $g_\phi$ using alternating minimization.
- Phase 3: Evaluates the model on both synthetic test sets and real clinical data.

To run with your custom settings:
1. Modify `config.py`.
2. Run `python main.py`.

