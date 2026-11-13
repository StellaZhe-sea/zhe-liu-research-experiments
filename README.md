# Zhe Liu — selected research experiments

This English-language repository contains selected, completed experiments that
support the computational research described in my CV. Each project folder
contains a compact source snapshot. `evidence/` contains configuration and
metric records for the displayed results. Failed and unfinished runs are not
included. These are distinct projects, not one shared architecture.

| CV research line | Source | Selected evidence |
| --- | --- | --- |
| [Spectral-residual multi-grade PDE learning](spectral_mgdl/README.md) | [`spectral_mgdl/src/`](spectral_mgdl/src/) | [19 figures](spectral_mgdl/results/figures/), [machine-readable metrics](spectral_mgdl/results/metrics.json), and [result boundaries](spectral_mgdl/docs/RESULT_BOUNDARIES.md) |
| [Multi-Grade DeepONet](MULTIGRADE_DEEPONET.md) | [`multigrade_deeponet/tools/run_fair_pendulum.py`](multigrade_deeponet/tools/run_fair_pendulum.py) | [`evidence/pendulum_summary.json`](evidence/pendulum_summary.json) and [`evidence/pendulum_protocol.json`](evidence/pendulum_protocol.json) |
| Multi-grade FNO | `multigrade_fno/` | `evidence/fno_burgers_comparison.json` |
| DeepONet benchmark experiments | `multigrade_deeponet/tools/` | `evidence/antiderivative_*`, Caputo and stochastic metrics |
| FNO benchmark experiments | `multigrade_fno/` | `evidence/fno_burgers_six_resolutions.json`, Darcy and Navier–Stokes metrics |
| Additional Helmholtz operator experiment | `mgno_helmholtz/` | `evidence/mgno_helmholtz_resonance_v3.json` |

The FNO benchmark scripts and their support modules derive from the upstream
`ixScience/fourier_neural_operator` MIT-licensed implementation by Zongyi Li.
Its license is reproduced in `multigrade_fno/UPSTREAM_FNO_LICENSE`. The Darcy
metrics come from an attributed local adaptation of that source; the Darcy
script is not duplicated here. My multi-grade Burgers comparison is separate
from these paper benchmark adaptations.

The `spectral_mgdl/` folder is a curated source-and-results archive for the
submitted research report. Its 19 figures and metrics are tied to distinct
experiments; values from different tasks are not directly comparable. The
DeepONet and FNO folders are separate operator-learning projects and do not
share the non-operator architecture in `spectral_mgdl/`.

See `RESULTS_AND_LIMITS.md` for the metric definitions and limits. Datasets,
model weights, private files, and large generated outputs are excluded.

