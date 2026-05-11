"""
examples/llm_bridge_demo.py
============================

The neuro -> LLM bridge in code.

Two synthetic d-dimensional trajectories, both length T, both standardised
to unit per-dim variance so that distances live on comparable scales:

    stable  prompt analog  -> damped AR(1) (contractive linear dynamics)
    chaotic prompt analog  -> Henon map (discrete chaos) padded with noise dims

Both systems are observed through the same kNN-divergence estimator that
the original brain-trajectory pipeline used.

The point is not that prompts literally generate Henon orbits. The point
is that the same lambda estimator that produced meaningful numbers on
6-D brain trajectories produces meaningful numbers on any d-dimensional
state sequence, including ones that mimic the structural pathology we
worry about in LLM inference: small initial perturbations leading to
exponentially divergent reasoning paths.

Run
---
    python examples/llm_bridge_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyapunov_core import largest_lyapunov  # noqa: E402


def stable_prompt_trajectory(
    T: int = 6000,
    d: int = 6,
    decay: float = 0.5,
    noise_sigma: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """
    x_{t+1} = decay * x_t + small_noise.

    Deterministic lambda = log(decay) = log(0.5) = -0.693 per step.
    Small noise keeps trajectories non-degenerate without dominating dynamics.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((T, d))
    x[0] = rng.standard_normal(d)
    for t in range(1, T):
        x[t] = decay * x[t - 1] + noise_sigma * rng.standard_normal(d)
    return x


def chaotic_prompt_trajectory(
    T: int = 6000,
    d: int = 6,
    a: float = 1.4,
    b: float = 0.3,
    noise_sigma: float = 0.02,
    seed: int = 1,
) -> np.ndarray:
    """
    Henon map in the canonical chaotic regime, padded with noise dimensions
    for a high-dimensional hidden-state analogue.
    """
    if d < 2:
        raise ValueError("d must be at least 2.")
    rng = np.random.default_rng(seed)
    xy = np.zeros((T, 2))
    xy[0] = np.array([0.1, 0.1])
    for t in range(1, T):
        xy[t, 0] = 1.0 - a * xy[t - 1, 0] ** 2 + xy[t - 1, 1]
        xy[t, 1] = b * xy[t - 1, 0]
    if d == 2:
        return xy
    aux = noise_sigma * rng.standard_normal((T, d - 2))
    return np.concatenate([xy, aux], axis=1)


def step_entropy(X: np.ndarray, n_bins: int = 32) -> float:
    """Mean per-dim Shannon entropy of the marginal distribution."""
    eps = 1e-12
    H = 0.0
    for j in range(X.shape[1]):
        hist, _ = np.histogram(X[:, j], bins=n_bins, density=True)
        p = hist / (hist.sum() + eps)
        H += -np.sum(p * np.log(p + eps))
    return H / X.shape[1]


def main() -> None:
    T = 6000
    d = 6

    Xs = stable_prompt_trajectory(T=T, d=d)
    Xc = chaotic_prompt_trajectory(T=T, d=d)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)

    fit_window = (2, 10)
    lam_stable, diag_s = largest_lyapunov(
        Xs, k_neighbors=5, fit_range=fit_window, return_diagnostics=True
    )
    lam_chaotic, diag_c = largest_lyapunov(
        Xc, k_neighbors=5, fit_range=fit_window, return_diagnostics=True
    )
    d_curve_s = diag_s["d_curve"]
    d_curve_c = diag_c["d_curve"]

    H_stable = step_entropy(Xs)
    H_chaotic = step_entropy(Xc)

    bar = "=" * 72
    print()
    print(bar)
    print(" Lyapunov for synthetic LLM-style trajectories")
    print(bar)
    print(f"  trajectory length T = {T},  state dim d = {d}")
    print(f"  fit window for slope estimate = tau in {fit_window}")
    print()
    print("  Prompt A  (stable region)   ->  damped AR(1),  decay = 0.5")
    print("     true deterministic lambda  = log(0.5) = -0.693 per step")
    print(f"     per-dim entropy        H_A = {H_stable:+.3f}")
    print(f"     measured lambda            = {lam_stable:+.4f}  per step")
    print()
    print("  Prompt B  (chaotic region)  ->  Henon map + noise dims")
    print("     true largest lambda        = +0.418 per step (Henon a=1.4,b=0.3)")
    print(f"     per-dim entropy        H_B = {H_chaotic:+.3f}")
    print(f"     measured lambda            = {lam_chaotic:+.4f}  per step")
    print()
    print("  Punchline")
    print("  ---------")
    print(
        f"     entropy ratio   H_B / H_A  = "
        f"{H_chaotic / max(H_stable, 1e-9):+.2f}   (marginal spread)"
    )
    print(
        f"     lambda ratio    lam_B/lam_A = "
        f"{lam_chaotic / max(lam_stable, 1e-9):+.2f}   (trajectory divergence)"
    )
    print()
    print("  Marginal entropy alone under-explains the difference: both")
    print("  trajectories are standardised onto comparable coordinate scales,")
    print("  but the structural risk is in how neighbouring states evolve.")
    print()
    print("  Lambda separates them clearly. Same estimator, same code, same")
    print("  data shape (T, d). The signal lives in trajectory geometry,")
    print("  not in pointwise marginals.")
    print()
    print("  Translation to LLM eval:")
    print("    * Prompt A = an arithmetic question - the model has one")
    print("      attractor it pulls toward. Reasoning paths converge.")
    print("    * Prompt B = an open-ended creative continuation - tiny shifts")
    print("      in the early state route the generation completely differently.")
    print("  Pointwise confidence can look acceptable per step; lambda over")
    print("  trajectory geometry exposes whether perturbations amplify.")
    print(bar)

    print()
    print("  Divergence curves d(tau) = <log ||x_i+tau - x_j+tau||>")
    print("  tau   stable AR(1)      Henon chaos")
    print("  ---   ------------      -----------")
    for tau in range(0, 11):
        print(f"  {tau:3d}   {d_curve_s[tau]:+.4f}          {d_curve_c[tau]:+.4f}")
    print(bar)


if __name__ == "__main__":
    main()
