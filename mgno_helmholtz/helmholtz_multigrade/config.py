from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 20260729
    # Keep CPU as the safe default.  The launch scripts set this explicitly so
    # the same source tree can be validated in isolated CPU and CUDA venvs.
    device: str = os.environ.get("HELMHOLTZ_DEVICE", "cpu").lower()
    dtype: str = "float64"

    n: int = 256
    n_grades: int = 3
    train_size: int = 160
    val_size: int = 40
    test_size: int = 40
    batch_size: int = 16
    epochs_per_grade: int = 60
    early_stopping_patience: int = 10
    early_stopping_min_epochs: int = 12
    early_stopping_relative_delta: float = 1.0e-3

    width: int = 20
    depth: int = 3
    learning_rate: float = 2.0e-3
    grade_learning_rate_decay: float = 0.35
    weight_decay: float = 1.0e-8
    min_learning_rate: float = 5.0e-5
    scheduler_patience: int = 12
    # The old 1e-3 shift suppresses the nearest resonant inverse gain by up
    # to two orders of magnitude for the controlled offsets below.  The data
    # are noiseless, so a much smaller stabilising shift is appropriate.
    shift_ratio: float = 1.0e-5
    gate_floor_ratio: float = 1.0e-2
    # Retain a small, continuous contribution from every DST mode.  A hard
    # spectral cut can discard a weak mode that becomes important close to a
    # resonance, and prevents a later Grade from recovering it.
    gate_soft_floor: float = 5.0e-2
    gate_temperature_ratio: float = 1.5e-1

    h1_loss_weight: float = 0.65
    pde_loss_weight: float = 0.35
    null_loss_weight: float = 10.0
    # Emphasise the difficult part of the operator family, rather than letting
    # samples far from an eigenfrequency dominate the batch mean.
    resonance_loss_weight: float = 1.5
    resonance_width: float = 8.0e-2
    correction_floor_ratio: float = 1.0e-6
    residual_floor_ratio: float = 1.0e-8

    near_resonance_fraction: float = 0.4
    resonance_offsets: tuple = (0.01, 0.03, 0.08, 0.20)
    resonant_amplitude_floor: float = 0.1
    step_calibration_scales: tuple = (1.0, 0.5, 0.25, 0.125, 0.0)

    @property
    def output_dir(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent
            / f"helmholtz_out_multigrade_resonance_v4_{self.device.replace(':', '_')}"
        )


CONFIG = Config()
