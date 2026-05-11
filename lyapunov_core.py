"""
lyapunov_core.py
================

Substrate-agnostic estimator of the largest Lyapunov exponent (λ) for
trajectories in a reconstructed phase space.

The same code works for:
    - 6D brain trajectories  (gradient + derivative phase space, original use case)
    - LLM hidden-state trajectories
    - Token-sample ensembles from a fixed prompt
    - Any X ∈ R^(T x d) where T is time and d is embedding dimension

Method:  Rosenstein-style nearest-neighbour divergence (Rosenstein et al. 1993),
which is the practical, finite-data, noise-robust estimator that turned the
Wolf et al. (1985) algorithm from a textbook curiosity into something you can
run on real data.

References
----------
Wolf et al. (1985) Physica D 16: 285-317.
Rosenstein et al. (1993) Physica D 65: 117-134.
Eckmann & Ruelle (1985) Rev. Mod. Phys. 57: 617.
Takens (1981) Dynamical Systems and Turbulence, LNM 898.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "largest_lyapunov",
    "reconstruct_phase_space",
    "divergence_curve",
]


def reconstruct_phase_space(
    x: ArrayLike,
    embedding_dim: int = 6,
    delay: int = 1,
) -> np.ndarray:
    """
    Takens delay-coordinate embedding of a 1-D time series.

    Used when you have a scalar signal and need to lift it into a phase space.
    Not needed when your data is already multi-dimensional, for example brain
    parcels or LLM hidden states. In that case pass the array directly to
    `largest_lyapunov`.
    """
    x = np.asarray(x, dtype=float).ravel()
    T = len(x)
    span = (embedding_dim - 1) * delay
    if T <= span:
        raise ValueError(
            f"Time series too short for embedding_dim={embedding_dim}, delay={delay}."
        )
    rows = T - span
    X = np.empty((rows, embedding_dim))
    for i in range(embedding_dim):
        X[:, i] = x[i * delay : i * delay + rows]
    return X


def divergence_curve(
    X: ArrayLike,
    tau_max: int = 20,
    k_neighbors: int = 1,
    min_separation: int | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Mean log-divergence curve d(τ) = <log ||x_{i+τ} - x_{j(i)+τ}||>.

    j(i) is a nearest neighbour of x_i in phase space after excluding neighbours
    that are too close in time using a Theiler window.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D: (T, d).")
    T, _ = X.shape
    if T <= tau_max + 1:
        raise ValueError(f"Trajectory too short ({T} steps) for tau_max={tau_max}.")
    if min_separation is None:
        min_separation = tau_max

    last_valid = T - tau_max
    base = X[:last_valid]

    n_search = min(k_neighbors + 2 * min_separation + 1, last_valid)
    nbrs = NearestNeighbors(n_neighbors=n_search, metric=metric).fit(base)
    _, idx = nbrs.kneighbors(base)

    chosen = np.full((last_valid, k_neighbors), -1, dtype=int)
    for i in range(last_valid):
        kept = 0
        for j in idx[i]:
            if j == i or abs(int(j) - i) <= min_separation:
                continue
            chosen[i, kept] = j
            kept += 1
            if kept == k_neighbors:
                break

    valid = (chosen >= 0).all(axis=1)
    if not valid.any():
        raise RuntimeError(
            "No valid neighbour pairs found. Try increasing T, decreasing "
            "min_separation, or decreasing k_neighbors."
        )

    base_i = np.where(valid)[0]
    nbr_j = chosen[valid]

    eps = np.finfo(float).tiny
    d_curve = np.zeros(tau_max + 1)
    for tau in range(tau_max + 1):
        future_i = X[base_i + tau]
        future_j = X[nbr_j + tau]
        diff = future_i[:, None, :] - future_j
        dist = np.linalg.norm(diff, axis=-1)
        d_curve[tau] = np.mean(np.log(dist + eps))
    return d_curve


def largest_lyapunov(
    X: ArrayLike,
    tau: int = 3,
    k_neighbors: int = 5,
    fit_range: tuple[int, int] | None = None,
    return_diagnostics: bool = False,
):
    """
    Estimate the largest Lyapunov exponent λ from a phase-space trajectory.

    If `fit_range` is None, this uses the single-shot distance ratio:
        λ = (d(tau) - d(0)) / tau

    If `fit_range=(t0, t1)`, this fits the slope of the Rosenstein divergence
    curve over t0..t1 and returns that slope.
    """
    X = np.asarray(X, dtype=float)
    if fit_range is None:
        d_curve = divergence_curve(X, tau_max=tau, k_neighbors=k_neighbors)
        lam = (d_curve[tau] - d_curve[0]) / tau
        if return_diagnostics:
            return lam, {
                "d_curve": d_curve,
                "fit_slope": None,
                "fit_intercept": None,
                "fit_range": None,
            }
        return float(lam)

    t0, t1 = fit_range
    if t0 < 0 or t1 <= t0:
        raise ValueError("fit_range must be (t0, t1) with 0 <= t0 < t1.")
    d_curve = divergence_curve(X, tau_max=t1, k_neighbors=k_neighbors)
    taus = np.arange(t0, t1 + 1)
    slope, intercept = np.polyfit(taus, d_curve[t0 : t1 + 1], 1)
    lam = float(slope)
    if return_diagnostics:
        return lam, {
            "d_curve": d_curve,
            "fit_slope": slope,
            "fit_intercept": intercept,
            "fit_range": (t0, t1),
        }
    return lam
