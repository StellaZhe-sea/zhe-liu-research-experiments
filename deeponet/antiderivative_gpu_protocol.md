# Antiderivative: paired GPU experiments

These are PyTorch CUDA migrations of the unstacked bias/no-bias architectures
in Appendix Table S6, not unchanged official TensorFlow source.

## Shared protocol

- Equation: s'(x)=u(x), s(0)=0, x in [0,1].
- Branch [100,40,40], hidden ReLU and linear output.
- Trunk [1,40,40], ReLU at both affine layers.
- Original GRF generator: RBF length scale 0.2, 1000-point grid,
  cubic interpolation, 100 fixed sensors.
- Original RK45 label generator with unchanged default solver tolerances.
- 10000 training functions, 100000 test functions, one query per function.
- A newly cached seed42 dataset is shared by both variants. Its first 256
  training rows must match the archived local data within stated tolerances.
- Seed42 is a local choice, not a claim about the paper's seed.
- Shared parameters start at the same exported official TensorFlow Glorot-normal
  initialization. Disabled bias parameters are zero/frozen in the no-bias case.
- Ordinary MSE, full training batch of 10000, 50000 Adam updates, lr=0.001.
- TF1-style Adam: beta1=0.9, beta2=0.999, epsilon=1e-8, original epsilon placement.
- Float32, AMP off, TF32 off. CPU/GPU arithmetic is not bit-identical.
- Minimum training MSE checkpoint among evaluations at 0,1000,...,50000.
- Monitored test loss uses the first 65535 points, as in the original local
  runner; final full-test MSE uses all 100000 points. They are labeled separately.
- No PDE residual, boundary penalty, normalization, regularizer, or clipping.
- `bias`: 7401 trainable parameters. `no_bias`: 7360; removes the 40 final
  branch biases and one scalar output bias, not every layer's biases.

## Verified diagnostics

`tools/prepare_antiderivative_gpu.py` exports the actual official DeepXDE 0.11.2
OpNN predictions, MSE and parameters at updates 0,1,10. It performs no formal
training. The GPU runner checks both variants against these fixtures before
each run. It also checks disabled-bias semantics and exact optimizer-state
recovery on a 10000-pair diagnostic batch. Diagnostic data are not reported as
scientific results. Reports are under `results/gpu_environment/`.

## VSCode terminal commands

Open `D:\Workspace_codex_DeepONet` and select
`D:\Workspace_codex_DeepONet\DeepONet_GPU\python.exe`.

```powershell
Set-Location "D:\Workspace_codex_DeepONet"
& ".\DeepONet_GPU\python.exe" -u ".\tools\run_antiderivative_gpu.py" --variant bias --seed 42
```

The first run automatically invokes the original CPU interpreter to generate
the shared dataset, which can take longer than CUDA training. Do not interrupt
`Generating operator data...` solely because training has not started.
The second variant reuses the cache. Run it only after the first finishes:

```powershell
& ".\DeepONet_GPU\python.exe" -u ".\tools\run_antiderivative_gpu.py" --variant no_bias --seed 42
```

Completion marker: `ANTIDERIVATIVE_GPU_RUN_COMPLETE` followed by the variant.
Each variant writes a separate timestamped directory under `results/`, containing
configuration and hashes, run.log, loss.dat, best/latest.pt, metrics.json, full
test predictions and four example prediction archives. Existing CPU runs remain
untouched. Duplicate GPU runs are refused. Resume interrupted training using
the SAME variant and `--resume-run` with its exact results directory; recovery
starts at the latest 1000-update checkpoint, not necessarily the interrupted step.

Timing reports CUDA-synchronized active Windows time excluding sleep/hibernation,
including periodic evaluation and checkpointing, but excluding data generation,
parity validation, and final best-model prediction export. It is not CPU time.
