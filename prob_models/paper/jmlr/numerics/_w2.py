"""Wasserstein-2 helpers for the theory numerical illustrations (T2-3).

Self-contained (numpy only). Used by e_a_rate.py, e_b_bounds.py, e_c_ridge.py.
- gaussian_w2_1d : exact W2 between two 1-D Gaussians.
- bures_w2       : exact W2 between two (multivariate) Gaussians (Bures metric).
- emp_w2_1d      : exact 1-D W2 between two equal-size empirical samples.
- sliced_w2      : dependency-free sliced W2 for d-D empirical samples
                   (mirrors Track E's prob_models/paper/experiments/_metrics.py).
"""
import numpy as np


def gaussian_w2_1d(m1, v1, m2, v2):
    """Exact W2 between N(m1, v1) and N(m2, v2); v = variance."""
    return float(np.sqrt((m1 - m2) ** 2 + (np.sqrt(v1) - np.sqrt(v2)) ** 2))


def _spd_sqrt(S):
    """Square root of a symmetric PSD matrix via eigendecomposition."""
    w, V = np.linalg.eigh(np.asarray(S, float))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def bures_w2(m1, S1, m2, S2):
    """Exact W2 between N(m1, S1) and N(m2, S2) via the Bures metric on covariances."""
    m1 = np.asarray(m1, float); m2 = np.asarray(m2, float)
    S1 = np.asarray(S1, float); S2 = np.asarray(S2, float)
    S1h = _spd_sqrt(S1)
    cross = _spd_sqrt(S1h @ S2 @ S1h)
    bures2 = np.trace(S1 + S2 - 2.0 * cross)
    mean2 = float(np.sum((m1 - m2) ** 2))
    return float(np.sqrt(max(mean2 + bures2, 0.0)))


def emp_w2_1d(a, b):
    """Exact 1-D W2 between two equal-size empirical samples (quantile matching)."""
    a = np.sort(np.asarray(a, float).ravel())
    b = np.sort(np.asarray(b, float).ravel())
    n = min(a.size, b.size)
    # subsample the larger to the common quantile grid
    qa = np.quantile(a, np.linspace(0, 1, n))
    qb = np.quantile(b, np.linspace(0, 1, n))
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def sliced_w2(a, b, n_proj=400, seed=0):
    """Sliced W2 between empirical samples a (n,d), b (m,d). Dependency-free.

    Exact 1-D W2 per random projection, averaged over ``n_proj`` unit directions.
    A proper metric that lower-bounds the true W2; stable and estimator-consistent
    with Track E's b7_ablation eps_inc.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    thetas = rng.normal(size=(n_proj, d))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12
    q = np.linspace(0.0, 1.0, 512)
    tot = 0.0
    for th in thetas:
        tot += np.mean((np.quantile(a @ th, q) - np.quantile(b @ th, q)) ** 2)
    return float(np.sqrt(tot / n_proj))
