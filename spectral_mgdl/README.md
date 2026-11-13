# Spectral-Residual Multi-Grade Deep Learning for PDEs

This repository documents a research framework that combines PDE residuals, spectral analysis, and multi-grade deep learning. At each grade, the method analyzes the current residual, selects important frequencies or atoms, trains a small correction network, evaluates the candidate solution, and freezes the accepted grade before moving to the next one.

The experiments cover high-frequency Helmholtz equations, nonlinear Schrödinger equations (NLS), 2D spectral extensions, Kolmogorov flow reconstruction, steady Navier–Stokes residuals, SpectralGNN ablations, Newton methods, and reference MGDL/SGDL baselines.

## Key results

These numbers come from saved execution logs or executed notebooks. They do not all measure the same task and should not be ranked directly.

| Experiment | Result | Scope |
|---|---|---|
| Helmholtz H1-8, single synthetic PDE | Relative L2 error `2.2656e-14` | Fixed equation and manufactured solution |
| Frozen 3-grade Helmholtz | Relative L2 error `1.1613e-10` | Fixed high-frequency multi-scale problem |
| Helmholtz operator learning | Unseen-test mean relative L2 error `2.014e-4`, max `5.627e-3` | 160 train / 40 validation / 40 unseen test PDEs |
| 2D FFT×FFT Helmholtz | Best accepted relative L2 error `1.6404e-9` | Synthetic 2D periodic problem |
| 1D cubic NLS, historical run | Relative L2 error `7.1940e-9` after 3 grades | Manufactured nonlinear problem |
| 2D NLS split-step validation | Mass relative error `4.29e-13`; Hamiltonian relative error `4.56e-12` | Physical conservation check |
| Kolmogorov flow reconstruction | Fourier validation relative squared error `0.009904`; historical serial Wavelet about `2e-6` | Basis-comparison experiments with different training policies |
| 2D NLS ablation | Baseline `0.049309`, Method-B `0.049301`, PI-GNN `0.048256` | 8 grades; PI-GNN improves by about 2.14% |
| Third-party MGDL regression | Test RSE `5.5488e-6` | Reference implementation, not an original contribution |
| SpectralGNN full ablation | Baseline final error `6.8709e-3`; all tested GNN variants were worse under this setup | Negative result retained |

See [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md) for the full result matrix and [docs/RESULT_BOUNDARIES.md](docs/RESULT_BOUNDARIES.md) for interpretation limits.

## Repository layout

```text
src/                     Canonical, runnable implementations
  helmholtz_multigrade/  Frozen multi-grade DST-FNO operator learning
  helmholtz_single/      Single-PDE spectral multigrade implementations
  grbd/                  Basis-agnostic GRBD framework
  nls2d/                 2D NLS operator, grade solver, and PI-GNN
  kolmogorov/            Kolmogorov pseudo-spectral data and GRBD training
  gnn/                   SpectralGNN implementation and ablation runner
  spectral_tools/        Fourier, wavelet, wavelet-packet, and 2D operators
  asmgdl/                AS-MGDL prototype with Snake and PCGrad
docs/                    English framework, results, and reproducibility notes
results/                 Curated figures and machine-readable metrics
```

The public tree intentionally excludes local Chinese historical archives, third-party source archives, installers, books, personal files, and machine-specific virtual environments.

## Reproduce

Create an environment from [requirements.txt](requirements.txt), preferably Python 3.10.11. GPU execution is optional, but CUDA FFT and linear algebra were validated with PyTorch `2.9.1+cu128` on an RTX 5090 Laptop GPU.

```bash
pip install -r requirements.txt

# Single frozen Helmholtz experiment
python src/helmholtz_multigrade_single_pde/main.py

# Multi-PDE operator learning
python src/helmholtz_multigrade/main.py

# 2D FFT and FFT-Chebyshev runs
python src/spectral_tools/run_2d.py --mode fft2d
python src/spectral_tools/run_2d.py --mode fftcheb

# Kolmogorov Fourier/Wavelet comparison
python src/kolmogorov/train.py

# SpectralGNN ablation
python src/gnn/run_ablation.py --device auto

```

Representative outputs are already archived under `results/`. Historical Chinese exploratory notebooks are retained locally and are not part of this public tree.

## Scientific boundaries

- A low error on one manufactured PDE does not establish generalization; unseen-test results are reported separately.
- Several Kolmogorov implementations define RSE as relative squared error, not relative L2 error.
- Experiments using an exact spectral inverse, ground-truth checkpoint selection, or target residuals are identified as synthetic or training-mechanism studies.
- Failed, rejected, timed-out, and negative results are reported rather than hidden.
- Third-party MGDL/SGDL code is excluded from this public tree; only reproducibility results are summarized in the documentation.

## License

A license has not been selected by the repository owner yet. Public viewing is enabled, but reuse rights remain reserved until an explicit license is added.
