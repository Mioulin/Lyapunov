# Theory note

A two-page summary of what the estimator does and why.

## The continuous-time picture

For a smooth dynamical system

    dx/dt = f(x),    x ∈ R^d

linearise around a trajectory x(t). A perturbation δ(t) evolves as

    dδ/dt = J(x(t)) δ,    J = ∂f/∂x

Over a finite time window the perturbation is mapped by a **monodromy matrix**
M(t). The d eigenvalues of (M^T M)^{1/(2t)} as t → ∞ are the
**Lyapunov spectrum** {λ_1, λ_2, …, λ_d}, ordered from largest to smallest.

Operationally only the **largest** one tends to matter:

    λ_1 = lim_{t→∞} (1/t) · ln( ||δ(t)|| / ||δ(0)|| )

A positive λ_1 is the textbook definition of sensitive dependence on initial
conditions, i.e. chaos.

## Why we can estimate this from a single time series

By **Takens' embedding theorem** (1981), a single scalar observable y(t) of
a deterministic dynamical system carries enough information to reconstruct
a topologically faithful copy of the original phase space, via delay
coordinates:

    z(t) = ( y(t), y(t+τ), y(t+2τ), …, y(t+(m-1)τ) )

For a high-dimensional system (brain parcels, hidden states, multi-channel
imaging) we usually already have a multi-dimensional observable and can
skip delay embedding entirely - just pass the (T, d) array.

## The Rosenstein estimator

Wolf et al. (1985) gave the first practical numerical method for λ_1 from a
time series. It works but is fragile on short, noisy data: it requires
re-normalising the perturbation vector whenever it grows too large, which
introduces a lot of free parameters.

Rosenstein et al. (1993) gave the version we use here:

1. For each point x_i in phase space, find its nearest neighbour x_{j(i)}
   in space, **excluding** temporally nearby points (Theiler window). This
   prevents counting the same trajectory segment as its own neighbour, which
   would force λ → 0.
2. Track the distance ||x_{i+τ} - x_{j(i)+τ}|| as a function of τ.
3. Average the log-distance over all valid pairs i.
4. Fit a straight line to the early linear regime of the resulting curve
   d(τ). The slope is λ_1.

This estimator is what `largest_lyapunov` and `divergence_curve` in
`lyapunov_core.py` implement.

## What happens for stochastic data

Strictly speaking, the textbook Lyapunov spectrum is defined for
deterministic systems. Real data is always stochastic to some degree.

For a stochastic process, what the nearest-neighbour estimator measures is
better named the **largest local Lyapunov exponent** (Yamaguti & Sakai,
1991), or the **finite-time Lyapunov exponent (FTLE)**.

It combines two effects:

- the deterministic stretching rate of the underlying dynamics
- the rate at which independent noise injects fresh distance between
  initially nearby trajectories

For most applied purposes (regime detection, stability scoring, prompt-risk
scoring) this is the more useful quantity than the textbook λ. It is well-
defined on finite, noisy, irregularly-sampled data; it is sensitive to local
regime changes; and it gives a single scalar score per trajectory.

## Practical reading of measured λ

A few rules of thumb when applying this to high-dimensional data:

- **Always inspect d(τ) before reading λ.** A clean linear regime followed
  by a saturation plateau is the diagnostic of "the estimator worked".
  If d(τ) is jagged, the answer is unreliable.

- **The early-τ slope is what you want.** Saturation appears once distances
  reach the attractor's natural scale. Past that, the slope drops to zero
  for trivial geometric reasons. The Rosenstein fit window must lie in the
  linear regime.

- **λ is in nats per step.** If your time axis has physical meaning (Hz,
  tokens, generation steps), report λ in those units.

- **Compare λ across conditions, not λ to zero.** With finite-data and
  stochasticity, even purely contractive systems often yield small positive
  numbers. The signal is in differences: λ(condition A) vs λ(condition B).

## Why pointwise entropy is not enough

Entropy and λ measure orthogonal things.

- **Entropy** asks: at this moment, how spread is the distribution over
  possible next states? A single-timestep quantity.

- **λ** asks: how does spread *evolve*? Does a small perturbation today
  grow or shrink tomorrow? A trajectory-level quantity.

This is the same reason "the model is confident at each token" does not
imply "the chain of thought is stable".

## References

- Eckmann, J.-P. & Ruelle, D. (1985). Ergodic theory of chaos and strange
  attractors. *Rev. Mod. Phys.* 57: 617-656.
- Rosenstein, M. T., Collins, J. J., De Luca, C. J. (1993). A practical
  method for calculating largest Lyapunov exponents from small data sets.
  *Physica D* 65: 117-134.
- Takens, F. (1981). Detecting strange attractors in turbulence. In
  *Dynamical Systems and Turbulence* (LNM 898).
- Wolf, A., Swift, J. B., Swinney, H. L., Vastano, J. A. (1985).
  Determining Lyapunov exponents from a time series. *Physica D* 16:
  285-317.
- Yamaguti, M. & Sakai, K. (1991). Local Lyapunov exponents and noise.
  *Prog. Theor. Phys.* 86: 757-761.
