# Experiment Results

All values below come from saved logs or executed notebooks. `Relative L2` means \(\lVert u-u^*\rVert_2/\lVert u^*\rVert_2\) unless a table explicitly says relative squared error.

## 1. High-frequency Helmholtz, single synthetic problem

The main manufactured problem uses \(N=256\), \(\kappa=200\pi\), homogeneous Dirichlet boundary conditions, and a complex spectrum with 14 modes whose amplitudes range from `1.0` to `1.5e-5`. The resonant zero-space issue is explicitly regularized.

| Implementation | Grades | Final relative L2 | Evidence |
|---|---:|---:|---|
| Reviewed spectral multigrade | 3 | `4.9660e-06` | Current rerun; historical result `4.8803e-06` |
| Recurrent DST-FNO | 8 | `1.153697e-05` | Current and historical logs |
| Frozen 3-grade, warmup + dual-plateau LR | 3 | `1.1612746e-10` | Current and historical logs |
| Spectral adaptive dual, 4 grades | 4 | `2.5995406e-12` | Historical log |
| H1 + PDE, 8 grades | 8 | **`2.2656e-14`** | Current run |

The frozen 3-grade experiment discovers the following frequency groups:

- Grade 1: `[3, 7, 11]`
- Grade 2: `[19, 31, 47, 73, 103]`
- Grade 3: `[137, 173, 199, 201, 229, 247]`

This supports the interpretation that later grades progressively discover weaker high-frequency residual components.

## 2. Helmholtz operator learning

The canonical dataset contains 160 training PDEs, 40 validation PDEs, and 40 unseen test PDEs. Wave numbers cover \(\kappa/\pi\in[170,230]\), including near-resonant samples.

| Version | Unseen mean RSE | Unseen max RSE | Mean relative residual |
|---|---:|---:|---:|
| Frozen-operator historical version | `1.347329e-04` | `3.413798e-03` | `1.209998e-05` |
| Recurrent historical version | `1.912487e-04` | `5.309937e-03` | `1.505732e-05` |
| Recurrent current rerun | `2.013989e-04` | `5.626741e-03` | `1.827555e-05` |

This is the main generalization result, but it remains limited to the designed synthetic data distribution.

## 3. Nonlinear Schrödinger and 2D extensions

| Experiment | Saved result | Interpretation |
|---|---|---|
| 1D cubic NLS, historical | Relative L2 `7.1940e-09` after 3 grades; CPU version `6.6469e-08` | Newton-style linearized inverse iteration plus hard spectral projection |
| Early NLS notebook | Relative L2 about `2.745e-02` | Shows the accuracy limit of the earlier soft out-of-band strategy |
| 2D FFT×FFT, current | Best accepted relative L2 `1.6404e-09` | Grades 7 and 8 were rejected, demonstrating rollback protection |
| 2D FFT×Chebyshev, current | Relative L2 `0.16405` after accepted Grade 1; Grade 3 rejected | The implementation runs, but this route is not yet competitive |
| 2D NLS split-step validation | Mass error `4.29e-13`; Hamiltonian error `4.56e-12` | A half-step phase fix reduced Hamiltonian drift from `1.16` to `4.56e-12` |
| 2D NLS grade ablation | Baseline `0.049309`; Method-B `0.049301`; PI-GNN `0.048256` | PI-GNN improves by about 2.14%, a useful but limited gain |

## 4. Kolmogorov flow and steady Navier–Stokes

Data generation uses a pseudo-spectral vorticity-streamfunction solver with RK4 time integration, 2/3 dealiasing, and Kolmogorov forcing.

| Experiment | Saved result | Note |
|---|---:|---|
| Python Fourier-GRBD, current validation | Relative squared error `0.009904` | Fixed plateau in this setup |
| Python Wavelet-GRBD, current | Printed as `0.000000` | The saved format has six decimals; this is not proof of exact zero error |
| Learnable Lifting, historical | Relative squared error `0.002431` | Higher training cost |
| Serial Wavelet-GRBD, historical versions | About `2e-6`, `4e-6`, `5e-6` | Different training policies |
| Steady NS Fourier, historical | Best PDE RMS `3.16e-04` | Residual optimization result |
| Steady NS Wavelet, historical | Best PDE RMS `2.47e-04` | Slightly better than Fourier in this setup |

Some validation implementations still use target-derived residuals during the forward pass. These experiments demonstrate training and representation behavior, not fully blind inference.

## 5. Multi-basis and SpectralGNN comparison

The full basis suite compares Fourier, Haar wavelet, wavelet packets, MLP, and GNN variants on smooth, multi-band, and locally concentrated Helmholtz problems.

Key observations from the completed full run:

- Wavelet packets provide the strongest final errors on the local and multi-band examples in the current implementation.
- Fourier can converge quickly on smooth synthetic problems but stagnates on some flow-reconstruction tasks.
- GNN adds cross-mode coupling, but its benefit is problem-dependent and can be outweighed by optimization and memory cost.

The completed 1D SpectralGNN ablation reports:

| Variant | Final error | Relative to baseline |
|---|---:|---:|
| Baseline GradeNet | `6.8709e-03` | reference |
| GNN energy, two layers | `4.0477e-02` | 5.891× worse |
| GNN distance, two layers | `3.5753e-02` | 5.203× worse |
| GNN learned MLP edges | `4.6213e-02` | 6.726× worse |
| GNN selection edges | `3.4302e-02` | 4.992× worse |
| GNN cascade edges | `1.6010e-02` | 2.330× worse |
| Energy GNN without message passing | `2.9611e-02` | 4.310× worse |

This negative result is retained because it is scientifically informative.

## 6. Newton methods

The Newton notebook implements and validates six algorithms from Deuflhard's framework.

| Algorithm | Saved behavior |
|---|---|
| Ordinary Newton | 5 iterations; residual `1.11e-16` |
| Simplified Newton | 17 iterations; residual `7.96e-11`; one Jacobian evaluation |
| Broyden rank-1 update | 9 iterations; residual `1.60e-13`; one exact Jacobian |
| Damped Newton with Armijo | 10 iterations; zero residual in the test |
| Natural monotonicity test | 100 iterations; residual about `4.73e+02`; did not converge |
| Gauss–Newton | 28 iterations; residual norm `9.71e-02`; fitted `a=0.494695`, `b=1.958341` |

## 7. Third-party MGDL and SGDL references

The original third-party source archive is not uploaded. Reproducibility results are summarized here.

| Reference experiment | Saved result | Caution |
|---|---:|---|
| MGDL synthetic regression | Test RSE `5.5488e-06`; validation RSE `2.0093e-04` | 20/20/40 epochs across three grades |
| SGDL synthetic regression | Test RSE `0.997109` | Only five epochs in the current configuration |
| MGDL noisy-label MNIST, historical | Validation accuracy `0.9649` | 20% label noise |
| SGDL MNIST, historical | `0.9654` after 5 epochs; `0.8146` after 100 epochs | Shows training-budget sensitivity |

## 8. Machine-readable metrics

Curated metrics are stored in [results/metrics.json](../results/metrics.json) and [results/metrics.csv](../results/metrics.csv). Raw historical logs are not uploaded to the public English tree.
