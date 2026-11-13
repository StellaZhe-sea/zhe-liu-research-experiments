# Research Framework

## Problem

The project studies whether PDE residuals can guide a sequence of small learning problems instead of asking one network to fit all scales at once. A single network often learns the dominant low-frequency components first, while high-frequency residual components become very small, noisy, and difficult to optimize. Near-resonant Helmholtz operators make this problem worse because small eigenvalues amplify numerical and optimization errors.

## Core loop

For a PDE operator

\[
\mathcal{L}[u]=f,
\]

the current approximation \(u_n\) produces the residual

\[
e_n=\mathcal{L}[u_n]-f.
\]

A linearized or shifted preconditioner estimates the required correction. A spectral or multiresolution transform then decomposes the residual, selects the currently important modes or atoms, and constructs features for the active grade. The grade network predicts a correction, the candidate solution is evaluated, and the grade is frozen only if it improves the accepted state.

```text
initial solution
  -> physical residual
  -> preconditioning / linearization
  -> DST / FFT / Chebyshev / DWT / WPT representation
  -> adaptive mode or atom selection
  -> grade network predicts a correction
  -> spectral and physical loss
  -> candidate evaluation and accept/reject
  -> freeze current grade
  -> repeat with the new residual
```

This is a **multi-grade learning framework**, not a geometric multigrid method. Different grades may use different active sets, basis functions, network capacities, learning rates, or residual normalization factors.

## Main components

### Operators and representations

- DST-I: exact spectral treatment of one-dimensional Dirichlet Helmholtz and Poisson problems.
- FFT and FFT×FFT: periodic one-dimensional and two-dimensional problems.
- FFT×Chebyshev: mixed periodic and non-periodic directions, including boundary-layer problems.
- Haar DWT and general wavelet bases: localized spatial and multiresolution residual components.
- Wavelet packets: entropy-selected adaptive time-frequency partitions.
- Gaussian random field sampling: operator-eigenbasis sampling for synthetic solutions and tests.

### Models

- **GradeNet**: symmetric frequency mixing followed by a small MLP.
- **GradeDSTFNO / Recurrent DST-FNO**: local plus spectral residual blocks with physical-state channels and hard frequency gates.
- **GRBD / AtomMLP**: per-coefficient weights for Fourier, wavelet, lifting, or wavelet-packet atoms.
- **SpectralGNN / PI-GNN**: frequency nodes with eigenvalue or dispersion embeddings and physics-inspired edge weights.
- **AS-MGDL**: Snake activation, hidden-state filtering, SVD energy truncation, and PCGrad for conflicting frequency objectives.

### Training and evaluation

- Previous grades are strictly frozen; later grades learn only from the residual left by earlier grades.
- Grade-local residual normalization prevents late-stage objectives from vanishing.
- Losses combine spectral error, PDE residual, H1 regularization, null-space control, and monotonicity where applicable.
- Learning-rate warmup, plateau adaptation, best-checkpoint rollback, and adaptive early stopping improve late-grade stability.
- Candidate grades can be rejected without contaminating the accumulated solution.

## Three experiment types

1. **Single synthetic PDE**: validates whether the method can reduce error on a known manufactured problem.
2. **Multi-PDE operator learning**: trains on many offline PDE samples and evaluates on unseen right-hand sides and wave numbers.
3. **Data-driven field reconstruction**: studies basis choice for coarse-to-fine vorticity reconstruction.

These settings answer different questions and must not be merged into one accuracy ranking.

## Current interpretation

The strongest evidence is not a single best number. It is the repeated observation that residual-guided grade selection, grade-local losses, frozen training, and rejection control lead to stable error reduction across Helmholtz, NLS, 2D, flow-reconstruction, and basis-comparison experiments. Negative GNN results and unresolved long-running experiments are also part of the current state.
