# Results and interpretation limits

## Multi-grade PDE learning

The synthetic fixed-problem 1D Helmholtz run reported a relative L2 error of
1.1612746300746522e-10 after three frozen grades. This is one PDE instance;
it is not evidence of unseen-operator generalization.

The expanded [`spectral_mgdl/`](spectral_mgdl/) archive includes 19 figures,
machine-readable metrics, and per-experiment documentation. The reported
single-PDE, multi-PDE, NLS, 2D Helmholtz, and ablation results use different
protocols and cannot be ranked as one benchmark. See its
[result boundaries](spectral_mgdl/docs/RESULT_BOUNDARIES.md) for the scope of
historical runs, synthetic targets, and negative results.

## Multi-grade FNO

On the recorded 200-sample, 1,024-point held-out Burgers test, the full
dynamic model reached relative L2 error 9.663196324254387e-06. The
parameter-matched FNO baseline reached 2.363947294652462e-03. This run
does not establish reduced total training time or the independent effect of
every component.

## Multi-Grade DeepONet

The recorded fair pendulum comparison used equal parameter counts and update
budgets. The final test MSE was 1.232621234e-03 for Multi-Grade DeepONet
versus 1.379008963e-03 for the traditional comparison. The runtime record
includes metric evaluation. This is a single selected comparison, not a
multi-seed performance distribution.

## DeepONet and FNO benchmarks

DeepONet antiderivative, Caputo, and stochastic benchmark records are stored
with their own configuration and metrics. FNO Burgers, Darcy, and 2D
Navier–Stokes records come from separate runs. Compare only matching tasks,
data splits, model settings, and metric definitions. The six-resolution FNO
Burgers record uses 100 test functions per resolution; the multi-grade FNO
comparison above uses a separate 200-sample test protocol.

## Additional MGNO Helmholtz experiment

The constant-coefficient resonance-v3 run reported unseen-test mean relative
solution error 3.186557e-04. Its current source snapshot is not an archived
exact replay of that historical run.
