# Reproducibility Notes

## Validated environment

- Python 3.10.11
- PyTorch 2.9.1+cu128
- CUDA runtime 12.8
- GPU used for validation: NVIDIA GeForce RTX 5090 Laptop GPU, compute capability `sm_120`
- NumPy 1.26.4
- SciPy 1.14.1
- Matplotlib 3.9.4
- PyWavelets 1.8.0
- TensorFlow CPU 2.15.1, only for legacy dataset loading in reference experiments

CUDA FFT backward propagation, SVD, convolution, TensorFlow import, and `pip check` passed in the validated environment.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, the project was developed under Git Bash. Commands below also work from PowerShell after replacing activation with `.venv\Scripts\Activate.ps1`.

## Run canonical experiments

```bash
# Frozen 3-grade Helmholtz
python src/helmholtz_multigrade_single_pde/main.py

# Multi-PDE Helmholtz operator learning
python src/helmholtz_multigrade/main.py

# Recurrent single-PDE DST-FNO
python src/helmholtz_single/1D_H_recurrent_dst_fno_cpu.py

# Reviewed spectral multigrade script
python src/helmholtz_single/1D_H_spectral_v9_k200pi_reviewed.py

# H1+PDE 8-grade experiment
python src/helmholtz_single/1D_H_spectral_v9_k200pi_h1_pde_cpu.py

# 2D FFT and FFT-Chebyshev
python src/spectral_tools/run_2d.py --mode fft2d
python src/spectral_tools/run_2d.py --mode fftcheb

# 2D NLS validation and ablation
python src/nls2d/nls2d_operator.py
python src/nls2d/nls2d_gnn_module.py
python src/nls2d/nls2d_grade_solver.py --device cuda

# Kolmogorov Fourier/Wavelet comparison
python src/kolmogorov/train.py

# Full basis suite and AS-MGDL prototype
python src/asmgdl/asmgdl.py
python src/basis_mgdl_full.py

# SpectralGNN ablation
python src/gnn/run_ablation.py --device auto
```

Some full experiments are intentionally expensive. Start with the smaller canonical Helmholtz runs before launching the full basis suite or SpectralGNN ablation.

## Outputs

- `src/helmholtz_multigrade_single_pde/` writes grade diagnostics and a frozen checkpoint under its configured output directory.
- `src/helmholtz_multigrade/` writes operator-learning metrics and visualizations.
- Kolmogorov training writes figures to `./results` relative to the execution directory.
- The 2D NLS ablation accepts `--output_dir`.
- Curated result figures and metrics are already provided under [results/](../results/).

## Large files

Git LFS is configured for `.pt`, `.npz`, and `.pptx` files. After cloning:

```bash
git lfs pull
```
