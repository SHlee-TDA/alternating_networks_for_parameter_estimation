"""Distribution-comparison metrics used across the experiment track.

POT (``ot``) is not installed in the ``vision_task`` env, so we implement a
self-contained sliced 2-Wasserstein estimator (exact 1-D W2 averaged over random
projections) plus helpers for sampling a gridded density and computing coverage.
"""
import numpy as np


def sliced_w2(a, b, n_proj=400, seed=0):
    """Sliced 2-Wasserstein distance between two 2-D empirical samples.

    a, b : (n,2) / (m,2) arrays. Exact 1-D W2 per projection (quantile matching),
    averaged over ``n_proj`` random unit directions. This is a proper metric and
    lower-bounds the true W2; it is stable and dependency-free.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    thetas = rng.normal(size=(n_proj, d))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12
    # common quantile grid
    q = np.linspace(0.0, 1.0, 512)
    tot = 0.0
    for th in thetas:
        pa = np.quantile(a @ th, q)
        pb = np.quantile(b @ th, q)
        tot += np.mean((pa - pb) ** 2)
    return float(np.sqrt(tot / n_proj))


def w1_1d(x, y):
    """Exact 1-D 1-Wasserstein via sorted quantile matching."""
    q = np.linspace(0.0, 1.0, 1024)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def sample_grid_density(gx, gy, dens, n, rng):
    """Sample ``n`` points from a density defined on a regular grid.

    gx, gy : 1-D grid axes (len Nx, Ny). dens : (Ny, Nx) non-negative density.
    Returns (n,2) samples with uniform jitter inside each selected cell.
    """
    dens = np.asarray(dens, float)
    p = dens.ravel()
    p = p / p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    iy, ix = np.unravel_index(idx, dens.shape)
    dx = gx[1] - gx[0]; dy = gy[1] - gy[0]
    xs = gx[ix] + (rng.random(n) - 0.5) * dx
    ys = gy[iy] + (rng.random(n) - 0.5) * dy
    return np.stack([xs, ys], axis=1)


def hpd_mask_2d(dens, level=0.95):
    """Boolean mask of the highest-posterior-density region at ``level`` on a grid."""
    dens = np.asarray(dens, float)
    flat = dens.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]) / flat.sum()
    k = np.searchsorted(csum, level) + 1
    thresh = flat[order][k - 1]
    return dens >= thresh


def coverage_in_grid_hpd(points, gx, gy, dens, level=0.95):
    """Fraction of ``points`` (n,2) falling inside the grid HPD region."""
    mask = hpd_mask_2d(dens, level)
    dx = gx[1] - gx[0]; dy = gy[1] - gy[0]
    ix = np.clip(((points[:, 0] - gx[0]) / dx).round().astype(int), 0, len(gx) - 1)
    iy = np.clip(((points[:, 1] - gy[0]) / dy).round().astype(int), 0, len(gy) - 1)
    return float(mask[iy, ix].mean())
