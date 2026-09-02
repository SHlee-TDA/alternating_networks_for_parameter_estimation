"""Shared configuration for the G0 sanity-oracle gate.

Both run_g0.py (training driver) and evaluate_g0.py (metrics) import these so the
train split, seeds, and hyperparameters stay perfectly consistent between the two
stages. Per the repo convention, these values are pushed into config.py before
each main.py run and the effective config is written next to the results.
"""
import os

SYSTEM_NAME = 'linear_oracle'
RESULTS_DIR = 'experiments/g0_sanity_oracle/results'
OUTPUT_DIR = 'experiments/g0_sanity_oracle/results'  # metrics + figures land here

SEEDS = [0, 1, 2, 3, 4]

# Data / training hyperparameters (clean, deterministic ODE => low noise).
NUM_SAMPLES = 20000
EPOCHS = 800
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
ITERATIONS = 10           # fixed-point unroll depth K at inference (headline iterative K)
USE_SPECTRAL_NORM = False  # best iterative accuracy on the oracle (note: not contractive)
RECURRENT_ITER = 2         # BPTT unroll depth for the recurrent training loss

# Inference K-sweep recorded to characterize the fixed-point behavior.
K_SWEEP = [1, 2, 3, 5, 8, 10, 15, 20, 30]

# Pass gate: NLS must recover the fully-identifiable oracle to < this (pipeline check).
# The iterative estimator's error is RECORDED as a documented method characterization
# (its ~3% floor is a fixed-point bias, not a pipeline bug — see FINDINGS.md), not gated.
REL_ERR_THRESHOLD = 1e-3

# NLS evaluation subset size (per seed) — NLS is slow, evaluate on a sample.
NLS_SUBSET = 300


def experiment_name(seed):
    return f'g0_seed{seed}'


def results_path(seed):
    """Directory where trainer saves Hnet.pth / Pnet.pth for a given seed."""
    return os.path.join(RESULTS_DIR, SYSTEM_NAME, experiment_name(seed))
