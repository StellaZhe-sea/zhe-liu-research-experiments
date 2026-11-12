# Multi-Grade DeepONet

This project studies residual learning across four DeepONet grades. Each grade
has branch and trunk networks; later grades receive the preceding grade's
hidden features and add a correction to the accumulated prediction. Completed
grades are frozen during subsequent training.

## Source and protocol

- [Multi-grade architecture](deeponet/src/models.py)
- [Parameter-matched model construction](deeponet/src/fair_pendulum_models.py)
- [Pendulum training and comparison](deeponet/tools/run_fair_pendulum.py)
- [Detailed experimental protocol](deeponet/FAIR_PENDULUM_PROTOCOL.md)

## Selected result

The recorded pendulum comparison uses 64,836 parameters and 4,000 optimizer
updates for each model. The multi-grade model has four grades with 1,000 updates
each. Both methods use the same data and sample sequence; their learning-rate
schedules differ. Standard normalization is fitted using training data only.

The final test MSE was 1.232621234e-3 for Multi-Grade DeepONet and
1.379008963e-3 for the conventional DeepONet comparison. The recorded elapsed
times were 75.220 seconds and 138.307 seconds, respectively, including metric
evaluation. These numbers describe one selected comparison, not a multi-seed
performance distribution.

- [Results by grade and final metrics](evidence/pendulum_summary.json)
- [Selected-run configuration](evidence/pendulum_protocol.json)
- [Integrity audit](evidence/pendulum_final_integrity_audit.json)
- [Interpretation limits](RESULTS_AND_LIMITS.md)

The public archive contains selected source and metric records. Original
training data and large model checkpoints are not included.

