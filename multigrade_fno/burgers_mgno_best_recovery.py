"""Recovered Burgers MGNO with standardized direct residual supervision."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Config:
    data: Path
    output_dir: Path
    device: str = "cuda"
    seed: int = 0
    ntrain: int = 1000
    ntest: int = 100
    resolution: int = 1024
    batch_size: int = 20
    modes: int = 16
    width: int = 64
    grade1_depth: int = 2
    grade2_depth: int = 2
    grade1_epochs: int = 200
    grade2_epochs: int = 300
    baseline_epochs: int = 500
    learning_rate: float = 1.0e-3
    grade2_learning_rate: float | None = None
    weight_decay: float = 1.0e-4
    scheduler_step: int = 100
    scheduler_gamma: float = 0.7
    run_fno4_baseline: bool = False
    verbose_every: int = 1
    post_training_test_audit: bool = True
    keep_epoch_checkpoints: bool = False
    overwrite: bool = False

    def validate(self) -> None:
        if not self.data.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {self.data}")
        if self.device != "cuda":
            raise ValueError("This experiment is intentionally CUDA-only")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but is not available")
        if self.ntrain < 1 or self.ntest < 1 or self.batch_size < 1:
            raise ValueError("Dataset and batch sizes must be positive")
        if 8192 % self.resolution != 0:
            raise ValueError("Resolution must divide the native 8192-point grid")
        if self.modes < 1 or self.modes > self.resolution // 2 + 1:
            raise ValueError("The Fourier mode count is incompatible with the grid")
        if self.width < 1 or min(
            self.grade1_epochs, self.grade2_epochs, self.baseline_epochs
        ) < 1:
            raise ValueError("Widths and epoch counts must be positive")
        if self.grade1_depth < 1 or self.grade2_depth < 1:
            raise ValueError("Each grade must contain at least one Fourier block")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Optimizer settings are invalid")
        if self.grade2_learning_rate is not None and self.grade2_learning_rate <= 0.0:
            raise ValueError("Grade 2 learning rate must be positive")
        if self.scheduler_step < 1 or not 0.0 < self.scheduler_gamma <= 1.0:
            raise ValueError("Scheduler settings are invalid")
        if not self.run_fno4_baseline:
            raise ValueError("The strict ablation requires the original FNO-4 baseline")
        if self.grade1_epochs + self.grade2_epochs != self.baseline_epochs:
            raise ValueError(
                "Multigrade total epochs must equal the original FNO epoch count"
            )
        if self.grade1_epochs % self.scheduler_step != 0:
            raise ValueError("Grade 1 must end on a global StepLR boundary")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_mat_field(path: Path, field: str) -> np.ndarray:
    try:
        data = scipy.io.loadmat(path)
        if field not in data:
            raise KeyError(f"Field '{field}' is absent from {path}")
        return np.asarray(data[field])
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            if field not in handle:
                raise KeyError(f"Field '{field}' is absent from {path}")
            return np.asarray(handle[field]).T


def make_coordinate_grid(resolution: int) -> torch.Tensor:
    """Return the coordinate convention used by this strict reproduction."""
    grid = np.linspace(0.0, 2.0 * np.pi, resolution).reshape(1, resolution, 1)
    return torch.tensor(grid, dtype=torch.float32)


def _load_burgers_fields(config: Config) -> tuple[np.ndarray, np.ndarray]:
    native_points = 8192
    stride = native_points // config.resolution
    inputs = _read_mat_field(config.data, "a")[:, ::stride].astype(np.float32)
    targets = _read_mat_field(config.data, "u")[:, ::stride].astype(np.float32)
    required = config.ntrain + config.ntest
    if inputs.shape[0] < required or targets.shape[0] < required:
        raise ValueError(
            f"Dataset has {inputs.shape[0]} samples but {required} are required"
        )
    return inputs, targets


def load_burgers_train_data(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Load only the first 1000 paper training samples."""
    inputs, targets = _load_burgers_fields(config)
    x_train = torch.from_numpy(inputs[: config.ntrain])
    y_train = torch.from_numpy(targets[: config.ntrain])
    grid = make_coordinate_grid(config.resolution)
    x_train = torch.cat(
        (x_train.unsqueeze(-1), grid.repeat(config.ntrain, 1, 1)), dim=-1
    )
    return x_train, y_train


def load_burgers_test_data(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the final 100 held-out samples only after all training is complete."""
    inputs, targets = _load_burgers_fields(config)
    x_test = torch.from_numpy(inputs[-config.ntest :])
    y_test = torch.from_numpy(targets[-config.ntest :])
    grid = make_coordinate_grid(config.resolution)
    x_test = torch.cat(
        (x_test.unsqueeze(-1), grid.repeat(config.ntest, 1, 1)), dim=-1
    )
    return x_test, y_test


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def complex_multiply_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Use the explicit real-imaginary multiplication from fourier_1d.py."""
    real = torch.einsum("bix,iox->box", a[..., 0], b[..., 0])
    real = real - torch.einsum("bix,iox->box", a[..., 1], b[..., 1])
    imag = torch.einsum("bix,iox->box", a[..., 1], b[..., 0])
    imag = imag + torch.einsum("bix,iox->box", a[..., 0], b[..., 1])
    return torch.stack((real, imag), dim=-1)


class SpectralConv1d(nn.Module):
    """Paper-compatible one-dimensional Fourier convolution."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / float(in_channels * out_channels)
        # Store real and imaginary parts explicitly, as in the paper code.
        # Parameter counts therefore represent real trainable scalars.
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_grid = x.shape[-1]
        spectrum = torch.view_as_real(torch.fft.rfft(x, norm="ortho"))
        output = torch.zeros(
            x.shape[0],
            self.out_channels,
            n_grid // 2 + 1,
            2,
            dtype=x.dtype,
            device=x.device,
        )
        active = min(self.modes, spectrum.shape[-2])
        output[:, :, :active] = complex_multiply_1d(
            spectrum[:, :, :active], self.weights[:, :, :active]
        )
        return torch.fft.irfft(
            torch.view_as_complex(output.contiguous()), n=n_grid, norm="ortho"
        )


class Projection1d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        physical = feature.permute(0, 2, 1)
        return self.fc2(F.relu(self.fc1(physical))).squeeze(-1)


class Grade1FNO(nn.Module):
    """Lift the input, apply two Fourier blocks, and project a solution."""

    def __init__(self, modes: int, width: int, depth: int) -> None:
        super().__init__()
        self.lifting = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.projection = Projection1d(width)
        self.activations = tuple(
             index < depth - 1
             for index in range(depth)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.lifting(x).permute(0, 2, 1)
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers, strict=True)
        ):
            feature = spectral(feature) + pointwise(feature)
            if self.activations[index]:
                feature = F.relu(feature)
        return self.projection(feature), feature


class Grade2ResidualFNO(nn.Module):
    """Lift a frozen Grade 1 Fourier feature and predict its solution residual."""

    def __init__(self, modes: int, width: int, depth: int) -> None:
        super().__init__()
        self.lifting = nn.Linear(width, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.projection = Projection1d(width)
        self.register_buffer("residual_mean", torch.zeros(()))
        self.register_buffer("residual_std", torch.ones(()))
        self.activations = tuple(
             index < depth - 1
             for index in range(depth)
        )

    def set_residual_statistics(self, residual: torch.Tensor) -> None:
        self.residual_mean.copy_(residual.mean().to(self.residual_mean))
        self.residual_std.copy_(
            residual.std(unbiased=False).clamp_min(1.0e-6).to(self.residual_std)
        )
        with torch.no_grad():
            self.projection.fc2.weight.zero_()
            self.projection.fc2.bias.fill_(
                float(-self.residual_mean / self.residual_std)
            )

    def forward(self, frozen_feature: torch.Tensor) -> torch.Tensor:
        feature = self.lifting(frozen_feature.permute(0, 2, 1)).permute(0, 2, 1)
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers, strict=True)
        ):
            feature = spectral(feature) + pointwise(feature)
            if self.activations[index]:
                feature = F.relu(feature)
        normalized = self.projection(feature)
        return self.residual_mean + self.residual_std * normalized


class SplitResidualFNO1d(nn.Module):
    """Two independently projected 2-layer grades with an additive correction."""

    def __init__(
        self,
        modes: int,
        width: int,
        grade1_depth: int,
        grade2_depth: int,
    ) -> None:
        super().__init__()
        self.grade1 = Grade1FNO(modes, width, grade1_depth)
        self.grade2 = Grade2ResidualFNO(modes, width, grade2_depth)

    def freeze_grade1(self) -> None:
        self.grade1.eval()
        self.grade1.requires_grad_(False)

    def grade1_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.grade1(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grade1, feature = self.grade1(x)
        correction = self.grade2(feature)
        return grade1 + correction, grade1, correction


class FNO4Baseline1d(nn.Module):
    """The original four-layer FNO architecture used by the paper."""

    def __init__(self, modes: int, width: int) -> None:
        super().__init__()
        self.lifting = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(4)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(4)
        )
        self.projection = Projection1d(width)
        self.activations = (True, True, True, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.lifting(x).permute(0, 2, 1)
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers, strict=True)
        ):
            feature = spectral(feature) + pointwise(feature)
            if self.activations[index]:
                feature = F.relu(feature)
        return self.projection(feature)

def save_epoch_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    model_name: str,
    epoch: int,
    active_grades: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir
        / f"{model_name}_epoch_{epoch:04d}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "epoch": epoch,
            "active_grades": active_grades,
        },
        checkpoint_path,
    )

    return checkpoint_path


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def relative_l2_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    error = (prediction - target).reshape(prediction.shape[0], -1).norm(dim=1)
    scale = target.reshape(target.shape[0], -1).norm(dim=1).clamp_min(epsilon)
    return (error / scale).sum()


def pointwise_mse_sum(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the summed squared error used to form the paper training MSE."""
    return (prediction - target).square().sum()


@torch.no_grad()
def collect_training_residual(
    model: SplitResidualFNO1d,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect only the frozen Grade 1 residual on training samples."""
    loader = make_loader(x_train, y_train, batch_size, shuffle=False)
    residuals: list[torch.Tensor] = []
    model.grade1.eval()
    for x, y in loader:
        prediction, _ = model.grade1_forward(x.to(device))
        residuals.append(y - prediction.cpu())
    return torch.cat(residuals)


@torch.no_grad()
def evaluate_grade1(
    model: SplitResidualFNO1d,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.grade1.eval()
    relative_l2 = 0.0
    squared_error = 0.0
    elements = 0
    samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction, _ = model.grade1_forward(x)
        relative_l2 += float(relative_l2_sum(prediction, y))
        squared_error += float(pointwise_mse_sum(prediction, y))
        elements += y.numel()
        samples += x.shape[0]
    return {
        "mse": squared_error / elements,
        "relative_l2": relative_l2 / samples,
    }


@torch.no_grad()
def evaluate_split(
    model: SplitResidualFNO1d,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | torch.Tensor]:
    model.eval()
    grade1_l2 = 0.0
    final_l2 = 0.0
    squared_error = 0.0
    elements = 0
    samples = 0
    outputs: list[torch.Tensor] = []
    grade1_outputs: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        final, grade1, correction = model(x)
        grade1_l2 += float(relative_l2_sum(grade1, y))
        final_l2 += float(relative_l2_sum(final, y))
        squared_error += float(pointwise_mse_sum(final, y))
        elements += y.numel()
        samples += x.shape[0]
        outputs.append(final.cpu())
        grade1_outputs.append(grade1.cpu())
        corrections.append(correction.cpu())
        targets.append(y.cpu())
    return {
        "grade1_relative_l2": grade1_l2 / samples,
        "final_relative_l2": final_l2 / samples,
        "mse": squared_error / elements,
        "prediction": torch.cat(outputs),
        "grade1_prediction": torch.cat(grade1_outputs),
        "correction": torch.cat(corrections),
        "target": torch.cat(targets),
    }


@torch.no_grad()
def evaluate_baseline(
    model: FNO4Baseline1d,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total = 0.0
    squared_error = 0.0
    elements = 0
    samples = 0
    outputs: list[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        total += float(relative_l2_sum(prediction, y))
        squared_error += float(pointwise_mse_sum(prediction, y))
        elements += y.numel()
        samples += x.shape[0]
        outputs.append(prediction.cpu())
    return squared_error / elements, total / samples, torch.cat(outputs)


def train_grade1(
    model: SplitResidualFNO1d,
    train_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.grade1.train()
    optimizer = torch.optim.Adam(
        model.grade1.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.grade1_epochs + 1):
        start = time.perf_counter()
        train_loss = 0.0
        train_mse = 0.0
        train_samples = 0
        model.grade1.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model.grade1_forward(x)
            loss = relative_l2_sum(prediction, y)
            mse = F.mse_loss(prediction, y, reduction="mean")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            train_mse += float(mse.detach())
            train_samples += x.shape[0]
        scheduler.step()
        row = {
            "stage": "grade1",
            "epoch": epoch,
            "global_epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "objective": train_loss / train_samples,
            "train_mse": train_mse / len(train_loader),
            "train_grade1_relative_l2": train_loss / train_samples,
            "train_final_relative_l2": train_loss / train_samples,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        if (
            config.post_training_test_audit
            or config.keep_epoch_checkpoints
        ):
            save_epoch_checkpoint(
                model=model.grade1,
                checkpoint_dir=(
                config.output_dir
                / "_epoch_checkpoints"
                / "multigrade"
                ),
                model_name="multigrade",
                epoch=epoch,
                active_grades=1,
            )

        if config.verbose_every and (
            epoch == 1 or epoch % config.verbose_every == 0
        ):
            print(
                f"Grade1 epoch={epoch:04d} "
                f"train_mse={row['train_mse']:.6e} "
                f"train_relative_l2={row['train_final_relative_l2']:.6e} "
                f"epoch_time={row['elapsed_seconds']:.3f}s"
            )
    model.freeze_grade1()
    return history


def train_grade2(
    model: SplitResidualFNO1d,
    train_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.freeze_grade1()
    snapshot = {
        name: value.detach().clone() for name, value in model.grade1.state_dict().items()
    }
    model.grade2.train()
    inherited_learning_rate = (
        config.learning_rate
        * config.scheduler_gamma
        ** (config.grade1_epochs // config.scheduler_step)
    )
    grade2_learning_rate = (
        config.grade2_learning_rate
        if config.grade2_learning_rate is not None
        else inherited_learning_rate
    )
    optimizer = torch.optim.Adam(
        model.grade2.parameters(),
        lr=grade2_learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.grade2_epochs + 1):
        start = time.perf_counter()
        train_loss = 0.0
        train_relative = 0.0
        train_mse = 0.0
        train_samples = 0
        model.grade2.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                grade1, frozen_feature = model.grade1_forward(x)
            optimizer.zero_grad(set_to_none=True)
            correction = model.grade2(frozen_feature)
            final_prediction = grade1 + correction
            residual_target = y - grade1
            normalized_target = (
                residual_target - model.grade2.residual_mean
            ) / model.grade2.residual_std
            normalized_correction = (
                correction - model.grade2.residual_mean
            ) / model.grade2.residual_std
            residual_objective = F.mse_loss(
                normalized_correction, normalized_target, reduction="mean"
            )
            final_relative = relative_l2_sum(final_prediction, y) / x.shape[0]
            loss = residual_objective + 0.1 * final_relative
            mse = F.mse_loss(final_prediction, y, reduction="mean")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * x.shape[0]
            train_relative += float(final_relative.detach()) * x.shape[0]
            train_mse += float(mse.detach())
            train_samples += x.shape[0]
        scheduler.step()
        row = {
            "stage": "grade2",
            "epoch": epoch,
            "global_epoch": config.grade1_epochs + epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "objective": train_loss / train_samples,
            "train_mse": train_mse / len(train_loader),
            "train_grade1_relative_l2": "",
            "train_final_relative_l2": train_relative / train_samples,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        if config.post_training_test_audit or config.keep_epoch_checkpoints:
            save_epoch_checkpoint(
                model=model,
                checkpoint_dir=(
                    config.output_dir / "_epoch_checkpoints" / "multigrade"
                ),
                model_name="multigrade",
                epoch=config.grade1_epochs + epoch,
                active_grades=2,
            )
        if config.verbose_every and (
            epoch == 1 or epoch % config.verbose_every == 0
        ):
            print(
                f"Grade2 epoch={epoch:04d} "
                f"train_mse={row['train_mse']:.6e} "
                f"train_relative_l2={row['train_final_relative_l2']:.6e} "
                f"epoch_time={row['elapsed_seconds']:.3f}s"
            )
    for name, value in model.grade1.state_dict().items():
        if not torch.equal(value, snapshot[name]):
            raise RuntimeError(f"Frozen Grade 1 changed during Grade 2 training: {name}")
    model.grade2.eval()
    model.grade2.requires_grad_(False)
    return history


def train_baseline(
    train_data: tuple[torch.Tensor, torch.Tensor],
    config: Config,
    device: torch.device,
) -> tuple[FNO4Baseline1d, list[dict[str, float | int | str]]]:
    set_seed(config.seed)
    model = FNO4Baseline1d(config.modes, config.width).to(device)
    train_loader = make_loader(
        *train_data,
        config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.baseline_epochs + 1):
        start = time.perf_counter()
        train_loss = 0.0
        train_mse = 0.0
        samples = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = relative_l2_sum(prediction, y)
            mse = F.mse_loss(prediction, y, reduction="mean")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            train_mse += float(mse.detach())
            samples += x.shape[0]
        scheduler.step()
        train_l2 = train_loss / samples
        row = {
            "stage": "fno4",
            "epoch": epoch,
            "global_epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "objective": train_loss / samples,
            "train_mse": train_mse / len(train_loader),
            "train_grade1_relative_l2": "",
            "train_final_relative_l2": train_l2,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        if config.post_training_test_audit or config.keep_epoch_checkpoints:
            save_epoch_checkpoint(
                model=model,
                checkpoint_dir=(
                    config.output_dir / "_epoch_checkpoints" / "fno4"
                ),
                model_name="fno4",
                epoch=epoch,
                active_grades=0,
            )
        if config.verbose_every and (
            epoch == 1 or epoch % config.verbose_every == 0
        ):
            print(
                f"FNO4 epoch={epoch:04d} "
                f"train_mse={row['train_mse']:.6e} "
                f"train_relative_l2={train_l2:.6e} "
                f"epoch_time={row['elapsed_seconds']:.3f}s"
            )
    model.eval()
    model.requires_grad_(False)
    return model, history


@torch.no_grad()
def evaluate_split_relative_l2_only(
    model: SplitResidualFNO1d,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the summed two-grade prediction without storing fields."""
    model.eval()
    total = 0.0
    samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction, _, _ = model(x)
        total += float(relative_l2_sum(prediction, y))
        samples += x.shape[0]
    return total / samples


@torch.no_grad()
def evaluate_baseline_relative_l2_only(
    model: FNO4Baseline1d,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate an FNO checkpoint without storing prediction fields."""
    model.eval()
    total = 0.0
    samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        total += float(relative_l2_sum(prediction, y))
        samples += x.shape[0]
    return total / samples


def add_cumulative_training_time(
    history: list[dict[str, float | int | str]],
) -> None:
    """Add independent cumulative training clocks for both compared models."""
    multigrade_seconds = 0.0
    fno_seconds = 0.0
    for row in history:
        if row["stage"] in ("grade1", "grade2"):
            multigrade_seconds += float(row["elapsed_seconds"])
            row["cumulative_training_seconds"] = multigrade_seconds
        else:
            fno_seconds += float(row["elapsed_seconds"])
            row["cumulative_training_seconds"] = fno_seconds
        row.setdefault("test_relative_l2", "")


def run_post_training_test_audit(
    history: list[dict[str, float | int | str]],
    test_loader: DataLoader,
    config: Config,
    device: torch.device,
) -> None:
    """Evaluate every saved epoch only after all optimization has finished."""
    if not config.post_training_test_audit:
        return

    print(
        "[Post-training test audit] Replaying frozen epoch checkpoints; "
        "test data were not accessed during optimization"
    )
    audit_split = SplitResidualFNO1d(
        config.modes,
        config.width,
        config.grade1_depth,
        config.grade2_depth,
    ).to(device)
    audit_fno = FNO4Baseline1d(config.modes, config.width).to(device)
    multigrade_dir = config.output_dir / "_epoch_checkpoints" / "multigrade"
    fno_dir = config.output_dir / "_epoch_checkpoints" / "fno4"

    for index, row in enumerate(history, start=1):
        global_epoch = int(row["global_epoch"])
        if row["stage"] in ("grade1", "grade2"):
            path = multigrade_dir / f"multigrade_epoch_{global_epoch:04d}.pt"
            payload = torch.load(path, map_location=device, weights_only=True)
            if int(payload["active_grades"]) == 1:
                audit_split.grade1.load_state_dict(payload["model_state_dict"])
                metric = evaluate_grade1(audit_split, test_loader, device)[
                    "relative_l2"
                ]
            else:
                audit_split.load_state_dict(payload["model_state_dict"])
                metric = evaluate_split_relative_l2_only(
                    audit_split, test_loader, device
                )
        else:
            path = fno_dir / f"fno4_epoch_{global_epoch:04d}.pt"
            payload = torch.load(path, map_location=device, weights_only=True)
            audit_fno.load_state_dict(payload["model_state_dict"])
            metric = evaluate_baseline_relative_l2_only(
                audit_fno, test_loader, device
            )
        row["test_relative_l2"] = metric
        if index == 1 or index % 25 == 0 or index == len(history):
            print(
                f"  audited {index:04d}/{len(history):04d} checkpoints | "
                f"{row['stage']} epoch={global_epoch:04d} "
                f"test_relative_l2={metric:.6e}"
            )


def save_metrics(rows: list[dict[str, object]], output_dir: Path) -> Path:
    path = output_dir / "metrics.csv"
    fields = [
        "stage",
        "epoch",
        "global_epoch",
        "learning_rate",
        "objective",
        "train_mse",
        "train_grade1_relative_l2",
        "train_final_relative_l2",
        "test_relative_l2",
        "elapsed_seconds",
        "cumulative_training_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_epoch_metrics_matrix(
    history: list[dict[str, object]], output_dir: Path
) -> tuple[Path, Path, tuple[int, int]]:
    """Save one aligned numerical row per global epoch for both models."""
    multigrade = {
        int(row["global_epoch"]): row
        for row in history
        if row["stage"] in ("grade1", "grade2")
    }
    fno4 = {
        int(row["global_epoch"]): row
        for row in history
        if row["stage"] == "fno4"
    }
    epochs = sorted(set(multigrade).intersection(fno4))
    columns = [
        "epoch",
        "multigrade_training_mse",
        "multigrade_training_relative_l2",
        "multigrade_test_relative_l2",
        "multigrade_epoch_training_seconds",
        "multigrade_cumulative_training_seconds",
        "fno4_training_mse",
        "fno4_training_relative_l2",
        "fno4_test_relative_l2",
        "fno4_epoch_training_seconds",
        "fno4_cumulative_training_seconds",
    ]
    matrix = np.asarray(
        [
            [
                epoch,
                multigrade[epoch]["train_mse"],
                multigrade[epoch]["train_final_relative_l2"],
                multigrade[epoch]["test_relative_l2"],
                multigrade[epoch]["elapsed_seconds"],
                multigrade[epoch]["cumulative_training_seconds"],
                fno4[epoch]["train_mse"],
                fno4[epoch]["train_final_relative_l2"],
                fno4[epoch]["test_relative_l2"],
                fno4[epoch]["elapsed_seconds"],
                fno4[epoch]["cumulative_training_seconds"],
            ]
            for epoch in epochs
        ],
        dtype=np.float64,
    )
    csv_path = output_dir / "epoch_metrics_matrix.csv"
    npy_path = output_dir / "epoch_metrics_matrix.npy"
    np.savetxt(
        csv_path,
        matrix,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    np.save(npy_path, matrix)
    return csv_path, npy_path, matrix.shape


def remove_temporary_epoch_checkpoints(config: Config) -> None:
    """Delete audit checkpoints after all numerical and visual reports exist."""
    checkpoint_dir = (config.output_dir / "_epoch_checkpoints").resolve()
    output_dir = config.output_dir.resolve()
    if output_dir not in checkpoint_dir.parents:
        raise RuntimeError("Temporary checkpoint directory escaped the output directory")
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
        print(f"[Cleanup] Deleted temporary epoch checkpoints: {checkpoint_dir}")


def save_figures(
    history: list[dict[str, object]],
    split_metrics: dict[str, float | torch.Tensor],
    baseline_prediction: torch.Tensor | None,
    output_dir: Path,
    config: Config,
) -> list[Path]:
    plt.style.use("default")
    plt.rcParams["svg.fonttype"] = "none"
    paths: list[Path] = []
    multigrade_label = (
        f"Multigrade FNO {config.grade1_depth}+{config.grade2_depth}"
    )
    multigrade_rows = [
        row for row in history if row["stage"] in ("grade1", "grade2")
    ]
    fno_rows = [row for row in history if row["stage"] == "fno4"]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.semilogy(
        [row["global_epoch"] for row in multigrade_rows],
        [row["train_mse"] for row in multigrade_rows],
        color="tab:blue",
        marker="o",
        markersize=1.8,
        linewidth=1.2,
        label=multigrade_label,
    )
    if fno_rows:
        ax.semilogy(
            [row["global_epoch"] for row in fno_rows],
            [row["train_mse"] for row in fno_rows],
            color="tab:orange",
            marker="s",
            markersize=1.8,
            linewidth=1.2,
            label="Original paper FNO-4",
        )
    ax.axvline(
        config.grade1_epochs,
        color="tab:gray",
        linestyle="--",
        alpha=0.5,
        label="Grade 1 frozen",
    )
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Burgers training MSE: two-model ablation")
    ax.grid(alpha=0.2)
    ax.legend()
    path = output_dir / "01_training_mse.svg"
    fig.savefig(path, format="svg", dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.semilogy(
        [row["global_epoch"] for row in multigrade_rows],
        [row["train_final_relative_l2"] for row in multigrade_rows],
        color="tab:blue",
        marker="o",
        markersize=1.8,
        linewidth=1.2,
        label=multigrade_label,
    )
    if fno_rows:
        ax.semilogy(
            [row["global_epoch"] for row in fno_rows],
            [row["train_final_relative_l2"] for row in fno_rows],
            color="tab:orange",
            marker="s",
            markersize=1.8,
            linewidth=1.2,
            label="Original paper FNO-4",
        )
    ax.axvline(
        config.grade1_epochs,
        color="tab:gray",
        linestyle="--",
        alpha=0.5,
        label="Grade 1 frozen",
    )
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Training relative L2 error")
    ax.set_title("Burgers training relative L2: two-model ablation")
    ax.grid(alpha=0.2)
    ax.legend()
    path = output_dir / "02_training_relative_l2.svg"
    fig.savefig(path, format="svg", dpi=300)
    plt.close(fig)
    paths.append(path)

    if all(row["test_relative_l2"] != "" for row in history):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.plot(
            [row["global_epoch"] for row in multigrade_rows],
            [row["test_relative_l2"] for row in multigrade_rows],
            color="tab:blue",
            label=multigrade_label,
        )
        ax.plot(
            [row["global_epoch"] for row in fno_rows],
            [row["test_relative_l2"] for row in fno_rows],
            color="tab:orange",
            label="Original paper FNO-4",
        )
        ax.axvline(
            config.grade1_epochs,
            color="tab:gray",
            linestyle="--",
            alpha=0.5,
            label="Grade 1 frozen",
        )
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("test relative L2 error")
        ax.set_yscale("log")
        ax.set_title(f"Burgers test relative L2 error, s={config.resolution}")
        ax.grid(alpha=0.25)
        ax.legend()
        path = output_dir / "07_test_relative_l2.svg"
        fig.savefig(path, format="svg", dpi=300)
        plt.close(fig)
        paths.append(path)

    target = split_metrics["target"]
    grade1 = split_metrics["grade1_prediction"]
    prediction = split_metrics["prediction"]
    assert isinstance(target, torch.Tensor)
    assert isinstance(grade1, torch.Tensor)
    assert isinstance(prediction, torch.Tensor)
    x = np.arange(target.shape[-1])
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(x, target[0], color="black", lw=2, label="target")
    axes[0].plot(x, prediction[0], color="#ff4ecd", label=multigrade_label)
    if baseline_prediction is not None:
        axes[0].plot(x, baseline_prediction[0], color="#ffb000", label="FNO-4")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title("First held-out Burgers sample")
    axes[0].legend()
    axes[1].semilogy(
        x,
        (target[0] - prediction[0]).abs() + 1.0e-12,
        color="#ff4ecd",
        label=f"{multigrade_label} error",
    )
    if baseline_prediction is not None:
        axes[1].semilogy(x, (target[0] - baseline_prediction[0]).abs() + 1.0e-12, color="#ffb000", label="FNO-4 error")
    axes[1].set_xlabel("Grid index")
    axes[1].set_ylabel("Absolute error")
    axes[1].legend()
    path = output_dir / "08_prediction_and_absolute_error.svg"
    fig.tight_layout()
    fig.savefig(path, format="svg", dpi=300)
    plt.close(fig)
    paths.append(path)

    grade1_residual = target - grade1
    final_residual = target - prediction
    grade1_spectrum = torch.fft.rfft(grade1_residual, norm="ortho").abs().square().mean(dim=0).sqrt()
    final_spectrum = torch.fft.rfft(final_residual, norm="ortho").abs().square().mean(dim=0).sqrt()
    fig, ax = plt.subplots(figsize=(11, 6))
    frequency = np.arange(grade1_spectrum.numel())
    ax.semilogy(frequency, grade1_spectrum + 1.0e-12, color="#00d4ff", label="Residual before Grade 2")
    ax.semilogy(frequency, final_spectrum + 1.0e-12, color="#ff4ecd", label="Residual after Grade 2")
    ax.set_xlabel("rFFT mode index")
    ax.set_ylabel("RMS residual amplitude")
    ax.set_title("What residual frequencies did Grade 2 remove?")
    ax.grid(alpha=0.2)
    ax.legend()
    path = output_dir / "09_residual_spectrum_diagnostic.svg"
    fig.tight_layout()
    fig.savefig(path, format="svg", dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def prepare_output_dir(config: Config) -> None:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {config.output_dir}"
            )
    config.output_dir.mkdir(parents=True, exist_ok=True)


def run(config: Config) -> dict[str, object]:
    config.validate()
    prepare_output_dir(config)
    set_seed(config.seed)
    device = torch.device(config.device)
    x_train, y_train = load_burgers_train_data(config)
    multigrade_shuffle_generator = torch.Generator().manual_seed(config.seed)
    train_loader_grade1 = make_loader(
        x_train,
        y_train,
        config.batch_size,
        shuffle=True,
        generator=multigrade_shuffle_generator,
    )
    train_loader_grade2 = make_loader(
        x_train,
        y_train,
        config.batch_size,
        shuffle=True,
        generator=multigrade_shuffle_generator,
    )
    split = SplitResidualFNO1d(config.modes, config.width, config.grade1_depth, config.grade2_depth).to(device)
    grade1_parameters = parameter_count(split.grade1)
    grade2_parameters = parameter_count(split.grade2)
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"Grade 1 parameters: {grade1_parameters}")
    print(f"Grade 2 parameters: {grade2_parameters}")
    print(f"Split-FNO parameters: {parameter_count(split)}")
    grade1_history = train_grade1(
        split, train_loader_grade1, config, device
    )
    training_residual = collect_training_residual(
        split, x_train, y_train, config.batch_size, device
    )
    split.grade2.set_residual_statistics(training_residual)
    print(
        "Grade 2 standardized residual target: "
        f"mean={float(split.grade2.residual_mean):.6e}, "
        f"std={float(split.grade2.residual_std):.6e}"
    )
    grade2_history = train_grade2(
        split, train_loader_grade2, config, device
    )
    split_training_seconds = sum(
        float(row["elapsed_seconds"])
        for row in grade1_history + grade2_history
    )
    baseline = None
    baseline_history: list[dict[str, float | int | str]] = []
    baseline_test_l2 = None
    baseline_test_mse = None
    baseline_prediction = None
    baseline_training_seconds = None
    baseline_parameters = parameter_count(FNO4Baseline1d(config.modes, config.width))
    if config.run_fno4_baseline:
        baseline, baseline_history = train_baseline(
            (x_train, y_train), config, device
        )
        baseline_training_seconds = sum(
            float(row["elapsed_seconds"]) for row in baseline_history
        )
    print("[Final test] All models are frozen; accessing held-out samples now")
    x_test, y_test = load_burgers_test_data(config)
    test_loader = make_loader(
        x_test,
        y_test,
        config.batch_size,
        shuffle=False,
    )
    history = grade1_history + grade2_history + baseline_history
    add_cumulative_training_time(history)
    run_post_training_test_audit(history, test_loader, config, device)
    split_metrics = evaluate_split(split, test_loader, device)
    if baseline is not None:
        baseline_test_mse, baseline_test_l2, baseline_prediction = evaluate_baseline(
            baseline, test_loader, device
        )
    save_metrics(history, config.output_dir)
    matrix_csv_path, matrix_npy_path, matrix_shape = save_epoch_metrics_matrix(
        history, config.output_dir
    )
    print(
        f"[Matrix] Saved epoch metrics matrix {matrix_shape}: "
        f"{matrix_csv_path}"
    )
    figures = save_figures(
        history,
        split_metrics,
        baseline_prediction,
        config.output_dir,
        config,
    )
    final_test = float(split_metrics["final_relative_l2"])
    split_final_training = grade2_history[-1]
    baseline_final_training = baseline_history[-1] if baseline_history else None
    total_multigrade_epochs = config.grade1_epochs + config.grade2_epochs
    multigrade_label = (
        f"Multigrade FNO {config.grade1_depth}+{config.grade2_depth}"
    )
    comparison_path = config.output_dir / f"final_epoch_{total_multigrade_epochs}_comparison.csv"
    comparison_rows = [
        {
            "model": multigrade_label,
            "epoch": config.grade1_epochs + config.grade2_epochs,
            "training_mse": split_final_training["train_mse"],
            "training_relative_l2": split_final_training["train_final_relative_l2"],
            "test_relative_l2": final_test,
            "training_seconds": split_training_seconds,
            "parameters": parameter_count(split),
        }
    ]
    if baseline_final_training is not None:
        comparison_rows.append(
            {
                "model": "Original paper FNO-4",
                "epoch": config.baseline_epochs,
                "training_mse": baseline_final_training["train_mse"],
                "training_relative_l2": baseline_final_training["train_final_relative_l2"],
                "test_relative_l2": baseline_test_l2,
                "training_seconds": baseline_training_seconds,
                "parameters": baseline_parameters,
            }
        )
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    result = {
        "config": {
            **asdict(config),
            "data": str(config.data.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "architecture": {
            "grade1": f"lifting(2,width) + {config.grade1_depth} Fourier blocks + projection",
            "grade2": f"lifting(frozen Grade 1 final Fourier feature) + {config.grade2_depth} Fourier blocks + projection",
            "training": "Grade 1 uses the paper relative L2; Grade 1 is then frozen; Grade 2 uses the same paper relative L2 on the summed prediction, which optimizes the frozen Grade 1 residual",
            "final_prediction": "grade1_prediction + grade2_residual_correction",
            "evaluation_protocol": "The held-out test set is evaluated once after training; it is never observed per epoch",
        },
        "fairness_protocol": {
            "shared_data": True,
            "shared_train_test_split": True,
            "shared_resolution": config.resolution,
            "shared_batch_size": config.batch_size,
            "shared_modes": config.modes,
            "shared_width": config.width,
            "shared_loss": "sample-wise target-relative L2",
            "shared_optimizer": "Adam",
            "shared_base_learning_rate": config.learning_rate,
            "grade2_initial_learning_rate": (
                config.grade2_learning_rate
                if config.grade2_learning_rate is not None
                else config.learning_rate
                * config.scheduler_gamma
                ** (config.grade1_epochs // config.scheduler_step)
            ),
            "grade2_learning_rate_restart": config.grade2_learning_rate is not None,
            "shared_weight_decay": config.weight_decay,
            "shared_scheduler": f"StepLR({config.scheduler_step}, {config.scheduler_gamma})",
            "shared_global_epoch_count": config.baseline_epochs,
            "shared_epoch_shuffle_sequence": True,
            "training_time_definition": "sum of training-only epoch times; excludes model construction, final testing, and reporting",
        },
        "parameters": {
            "grade1": grade1_parameters,
            "grade2": grade2_parameters,
            "split_total": parameter_count(split),
            "fno4": baseline_parameters,
            "split_to_fno4_ratio": parameter_count(split) / baseline_parameters,
        },
        "metrics": {
            "multigrade": {
                "epoch": config.grade1_epochs + config.grade2_epochs,
                "training_mse": split_final_training["train_mse"],
                "training_relative_l2": split_final_training["train_final_relative_l2"],
                "test_relative_l2": final_test,
                "training_seconds": split_training_seconds,
            },
            "original_fno4": {
                "epoch": config.baseline_epochs,
                "training_mse": (
                    baseline_final_training["train_mse"]
                    if baseline_final_training is not None
                    else None
                ),
                "training_relative_l2": (
                    baseline_final_training["train_final_relative_l2"]
                    if baseline_final_training is not None
                    else None
                ),
                "test_relative_l2": baseline_test_l2,
                "training_seconds": baseline_training_seconds,
            },
            "fno4_to_split_ratio": (
                baseline_test_l2 / final_test
                if baseline_test_l2 is not None
                else None
            ),
        },
        "figures": [str(path.resolve()) for path in figures],
        "final_comparison_csv": str(comparison_path.resolve()),
        "epoch_metrics_matrix_csv": str(matrix_csv_path.resolve()),
        "epoch_metrics_matrix_npy": str(matrix_npy_path.resolve()),
        "epoch_metrics_matrix_shape": list(matrix_shape),
    }
    with (config.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
    torch.save(
        {
            "config": result["config"],
            "architecture": result["architecture"],
            "model_state": split.state_dict(),
        },
        config.output_dir / "split_fno_checkpoint.pt",
    )
    torch.save(
        {
            "target": split_metrics["target"],
            "grade1_prediction": split_metrics["grade1_prediction"],
            "grade2_correction": split_metrics["correction"],
            "final_prediction": split_metrics["prediction"],
            "fno4_prediction": baseline_prediction,
        },
        config.output_dir / "test_predictions.pt",
    )
    if baseline is not None:
        torch.save(
            baseline.state_dict(), config.output_dir / "fno4_checkpoint.pt"
        )
    if not config.keep_epoch_checkpoints:
        remove_temporary_epoch_checkpoints(config)
    print(
        f"\nFINAL TWO-MODEL ABLATION RESULT: "
        f"MULTIGRADE EPOCH {total_multigrade_epochs}, "
        f"FNO-4 EPOCH {config.baseline_epochs}"
    )
    print(f"  {multigrade_label}")
    print(f"    training MSE:         {split_final_training['train_mse']:.6e}")
    print(f"    training relative L2: {split_final_training['train_final_relative_l2']:.6e}")
    print(f"    test relative L2:     {final_test:.6e}")
    print(f"    training time:        {split_training_seconds:.3f} s")
    print(f"    parameters:           {parameter_count(split)}")
    if baseline_test_l2 is not None:
        assert baseline_final_training is not None
        assert baseline_training_seconds is not None
        print("  Original paper FNO-4")
        print(f"    training MSE:         {baseline_final_training['train_mse']:.6e}")
        print(f"    training relative L2: {baseline_final_training['train_final_relative_l2']:.6e}")
        print(f"    test relative L2:     {baseline_test_l2:.6e}")
        print(f"    training time:        {baseline_training_seconds:.3f} s")
        print(f"    parameters:           {baseline_parameters}")
        print(f"  FNO-4/Multigrade test-error ratio: {baseline_test_l2 / final_test:.3f}x")
    print(f"  Output: {config.output_dir.resolve()}")
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Standalone frozen two-grade 2+2 FNO ablation for Burgers"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            r"D:\Workspace_codex_Fno\fourier-neural-operator-main\data\burgers_data_R10.mat"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "burgers_split_2plus2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--grade-depths", nargs=2, type=int, default=(2, 2))
    parser.add_argument("--grade1-epochs", type=int, default=200)
    parser.add_argument("--grade2-epochs", type=int, default=300)
    parser.add_argument("--baseline-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--grade2-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--run-fno4-baseline", action="store_true")
    parser.add_argument("--verbose-every", type=int, default=1)
    parser.add_argument("--disable-post-training-test-audit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return Config(
        data=args.data,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        ntrain=args.ntrain,
        ntest=args.ntest,
        resolution=args.resolution,
        batch_size=args.batch_size,
        modes=args.modes,
        width=args.width,
        grade1_depth=args.grade_depths[0],
        grade2_depth=args.grade_depths[1],
        grade1_epochs=args.grade1_epochs,
        grade2_epochs=args.grade2_epochs,
        baseline_epochs=args.baseline_epochs,
        learning_rate=args.learning_rate,
        grade2_learning_rate=args.grade2_learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        run_fno4_baseline=args.run_fno4_baseline,
        verbose_every=args.verbose_every,
        post_training_test_audit=not args.disable_post_training_test_audit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
