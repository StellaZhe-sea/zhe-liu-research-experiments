from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 20260729
    device: str = "cpu"
    dtype: str = "float64"

    n: int = 256
    n_grades: int = 3
    train_size: int = 160
    val_size: int = 40
    test_size: int = 40
    batch_size: int = 16
    epochs_per_grade: int = 60

    width: int = 20
    depth: int = 3
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-8
    min_learning_rate: float = 5.0e-5
    scheduler_patience: int = 12
    shift_ratio: float = 1.0e-3
    gate_floor_ratio: float = 1.0e-2

    h1_loss_weight: float = 0.65
    pde_loss_weight: float = 0.35
    null_loss_weight: float = 10.0

    @property
    def output_dir(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent
            / "helmholtz_out_multigrade_recurrent"
        )


CONFIG = Config()
