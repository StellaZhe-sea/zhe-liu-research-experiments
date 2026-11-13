# Result Boundaries and Known Limitations

## Metric definitions

- Most Helmholtz and NLS experiments report relative L2 error.
- Most Kolmogorov GRBD notebooks define RSE as relative squared error.
- Different metrics and training budgets are labeled separately and are not combined into unsupported speedup factors.

## Data and supervision boundaries

The following signals are allowed in synthetic algorithm studies but must not be described as fully unsupervised deployment:

- Exact spectral inverse used to generate correction targets.
- Ground-truth solution used only for evaluation.
- Ground-truth RSE used for checkpoint selection.
- Validation forward passes that still receive target fields to form residual inputs.

The final frozen Helmholtz implementation separates training/checkpoint selection from ground-truth supervision more carefully than the earlier exploratory versions, but the original Chinese historical archives remain local and are not part of this public tree.

## Generalization boundaries

- `2.2656e-14` is a fixed manufactured-PDE result. It is not an unseen-test generalization claim.
- The operator-learning experiments use a designed synthetic distribution of right-hand sides and wave numbers.
- Kolmogorov results currently validate representation and training mechanisms under the documented settings.
- Cross-grid, cross-boundary, real-data, and out-of-distribution robustness remain future work.

## Negative and unresolved results

These results are retained deliberately:

- 2D FFT×Chebyshev currently reaches only `0.16405` under the documented route.
- Several candidate grades are rejected by design; rejection is not a failure of the solver.
- The full 1D SpectralGNN ablation is worse than the simple baseline under the current setup.
- Early SpectralGNN runs ran out of memory on a 4 GB GPU. The current hardware run is complete, but it still reports worse final errors than the baseline.
- Some long or exploratory experiments remain queued or failed locally and are not presented as completed results.

## Reproducibility notes

- Python 3.10.11 and PyTorch 2.9.1+cu128 were validated with CUDA FFT backward propagation, SVD, and convolution.
- Git LFS tracks `.pt`, `.npz`, and `.pptx` artifacts where large binary files are included.
- The public tree is English-only. Historical Chinese exploratory notebooks, local environment folders, installers, and personal files are intentionally excluded.
- The repository does not claim a license yet. Add one before encouraging reuse.
