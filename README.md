# Zhe Liu — selected research experiments

This English-language repository contains selected, completed experiments that
support the computational research described in my CV. Each project folder
contains a compact source snapshot. `evidence/` contains configuration and
metric records for the displayed results. Failed and unfinished runs are not
included. These are distinct projects, not one shared architecture.

| CV research line | Source | Selected evidence |
| --- | --- | --- |
| Multi-grade PDE learning and residual analysis | `spectral_mgdl/helmholtz_single_pde/` | `evidence/helmholtz_single_pde.json` |
| Further 1D NLS and 2D Helmholtz examples | `spectral_mgdl/nls_1d/`, `spectral_mgdl/helmholtz_2d/` | English source snapshots; historical NLS figure in `figures/` |
| [Multi-Grade DeepONet](MULTIGRADE_DEEPONET.md) | [`deeponet/tools/run_fair_pendulum.py`](deeponet/tools/run_fair_pendulum.py) | [`evidence/pendulum_summary.json`](evidence/pendulum_summary.json) and [`evidence/pendulum_protocol.json`](evidence/pendulum_protocol.json) |
| Multi-grade FNO | `multigrade_fno/` | `evidence/fno_burgers_comparison.json` |
| DeepONet benchmark experiments | `deeponet/tools/` | `evidence/antiderivative_*`, Caputo and stochastic metrics |
| FNO benchmark experiments | `multigrade_fno/` | `evidence/fno_burgers_six_resolutions.json`, Darcy and Navier–Stokes metrics |
| Additional Helmholtz operator experiment | `mgno_helmholtz/` | `evidence/mgno_helmholtz_resonance_v3.json` |

The FNO benchmark scripts and their support modules derive from the upstream
`ixScience/fourier_neural_operator` MIT-licensed implementation by Zongyi Li.
Its license is reproduced in `multigrade_fno/UPSTREAM_FNO_LICENSE`. The Darcy
metrics come from an attributed local adaptation of that source; the Darcy
script is not duplicated here. My multi-grade Burgers comparison is separate
from these paper benchmark adaptations.

See `RESULTS_AND_LIMITS.md` for the metric definitions and limits. Datasets,
model weights, private files, and large generated outputs are excluded.

