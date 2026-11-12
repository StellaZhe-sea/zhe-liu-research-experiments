from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 20260729
    device: str = "cpu"
    n: int = 256
    kappa_over_pi: float = 200.0
    # Three frozen discovery Grades: low-, middle-, then high-frequency
    # residual correction.  No fourth all-frequency refinement Grade.
    n_grades: int = 3

    # Each Grade now chooses its own training length from physics-only
    # validation of the absolute PINN residual.
    # 401 is the empirically validated stabilization window for warm-started
    # latent features.  Adaptive training may extend a difficult Grade, but
    # must not freeze an apparently converged output before its reusable
    # internal representation has stabilized.
    min_epochs_per_grade: int = 401
    max_epochs_per_grade: int = 800
    early_stop_patience: int = 100
    early_stop_relative_tolerance: float = 1.0e-4
    width: int = 20
    depth: int = 3
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-8
    warmup_epochs: int = 20
    warmup_start_factor: float = 1.0e-1
    lr_reduce_factor: float = 5.0e-1
    lr_plateau_patience: int = 60
    min_learning_rate: float = 2.0e-5
    shift_ratio: float = 2.0e-2
    gate_floor_ratio: float = 1.0e-2

    h1_step_weight: float = 1.0e-7
    # Physics-only Sobolev-style residual weighting.  This emphasizes high
    # DST modes without using the exact solution in backpropagation.
    spectral_loss_weight: float = 2.5e-1
    spectral_high_frequency_boost: float = 8.0
    null_loss_weight: float = 10.0
    monotone_loss_weight: float = 10.0

    @property
    def output_dir(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent
            / "helmholtz_out_multigrade_single_pde_warmup_dual_lr_3grades"
        )


CONFIG = Config()
