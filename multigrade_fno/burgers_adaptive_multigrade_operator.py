"""Adaptive multi-grade Fourier neural operator for the paper Burgers data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from burgers_mgno_best_recovery import (
    FNO4Baseline1d,
    Projection1d,
    SpectralConv1d,
    load_burgers_test_data,
    load_burgers_train_data,
    make_loader,
    pointwise_mse_sum,
    relative_l2_sum,
    set_seed,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Config:
    data: Path
    output_dir: Path
    device: str = "cuda"
    seed: int = 0
    ntrain: int = 1000
    ntest: int = 200
    resolution: int = 1024
    batch_size: int = 20
    modes: int = 16
    widths: tuple[int, ...] = (64, 64)
    depths: tuple[int, ...] = (2, 2)
    epochs: tuple[int, ...] = (200, 300)
    learning_rates: tuple[float, ...] = (1.0e-3, 1.0e-3)
    grade_scheduler_steps: tuple[int, ...] = (100, 100)
    grade_scheduler_gammas: tuple[float, ...] = (0.5, 0.7)
    baseline_epochs: int = 500
    baseline_depth: int = 4
    weight_decay: float = 1.0e-4
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    basis_min_modes: int = 4
    basis_max_modes: int = 16
    residual_basis_modes: tuple[int, ...] = (16,)
    basis_energy: float = 0.997
    relative_error_score_power: float = 0.25
    coherence_score_power: float = 0.5
    consensus_folds: int = 5
    consensus_variation_penalty: float = 1.0
    new_frequency_fraction: float = 0.625
    residual_energy_floor_fraction: float = 1.0e-3
    minimum_new_modes: int = 2
    soft_frequency_floor: float = 0.05
    residual_mse_weight: float = 0.25
    spectral_weights: tuple[float, ...] = (0.0, 0.05)
    h1_weights: tuple[float, ...] = (0.0, 0.02)
    h1_truth_floor_ratio: float = 1.0e-6
    predictor_weights: tuple[float, ...] = (0.0, 0.10)
    run_fno4_baseline: bool = True
    verbose_every: int = 1
    post_training_test_audit: bool = True
    keep_epoch_checkpoints: bool = False
    cache_frozen_features: bool = True
    overwrite: bool = False

    @property
    def grade_count(self) -> int:
        return len(self.depths)

    def validate(self) -> None:
        if not self.data.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {self.data}")
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This experiment requires CUDA")
        if 8192 % self.resolution:
            raise ValueError("Resolution must divide the native 8192-point grid")
        if not 1 <= self.modes <= self.resolution // 2 + 1:
            raise ValueError("Operator modes are incompatible with the grid")
        lengths = {
            len(self.widths),
            len(self.depths),
            len(self.epochs),
            len(self.learning_rates),
            len(self.grade_scheduler_steps),
            len(self.grade_scheduler_gammas),
            len(self.spectral_weights),
            len(self.h1_weights),
            len(self.predictor_weights),
        }
        if lengths != {self.grade_count} or self.grade_count < 2:
            raise ValueError("Every grade-specific list must have the same length")
        if any(value <= 0 for value in (*self.widths, *self.depths, *self.epochs)):
            raise ValueError("Grade widths, depths, and epochs must be positive")
        if len(set(self.widths)) != 1:
            raise ValueError("This version requires a shared width across grades")
        if any(value <= 0.0 for value in self.learning_rates):
            raise ValueError("Learning rates must be positive")
        if any(value <= 0 for value in self.grade_scheduler_steps):
            raise ValueError("Grade scheduler steps must be positive")
        if any(not 0.0 < value <= 1.0 for value in self.grade_scheduler_gammas):
            raise ValueError("Grade scheduler gammas must lie in (0, 1]")
        if len(self.residual_basis_modes) != self.grade_count - 1:
            raise ValueError("Each residual grade requires one basis capacity")
        if any(
            not self.basis_min_modes <= value <= self.resolution // 2 + 1
            for value in self.residual_basis_modes
        ):
            raise ValueError("Every residual basis capacity must fit the grid")
        if not 1 <= self.basis_min_modes <= self.basis_max_modes:
            raise ValueError("Residual basis limits must be positive and ordered")
        if self.basis_max_modes > self.resolution // 2 + 1:
            raise ValueError("The residual basis exceeds the available RFFT modes")
        if not 0.0 < self.basis_energy <= 1.0:
            raise ValueError("Basis energy must lie in (0, 1]")
        if not 0.0 <= self.soft_frequency_floor <= 1.0:
            raise ValueError("Soft frequency floor must lie in [0, 1]")
        if self.coherence_score_power < 0.0:
            raise ValueError("Coherence score power must be nonnegative")
        if not 2 <= self.consensus_folds <= self.ntrain:
            raise ValueError("Consensus folds must lie between 2 and ntrain")
        if self.consensus_variation_penalty < 0.0:
            raise ValueError("Consensus variation penalty must be nonnegative")
        if not 0.0 <= self.new_frequency_fraction <= 1.0:
            raise ValueError("New-frequency fraction must lie in [0, 1]")
        if not 0.0 <= self.residual_energy_floor_fraction < 1.0:
            raise ValueError("Residual energy floor fraction must lie in [0, 1)")
        if not 0 <= self.minimum_new_modes <= min(self.residual_basis_modes):
            raise ValueError("Minimum new modes must fit the residual basis")
        if not self.run_fno4_baseline:
            raise ValueError("The complete FNO baseline is required")
        if self.baseline_depth != 4:
            raise ValueError("The strict paper baseline must contain four Fourier layers")


def sample_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).flatten(1).norm(dim=1)
    scale = target.flatten(1).norm(dim=1).clamp_min(1.0e-12)
    return error / scale


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def relative_h1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    truth: torch.Tensor | None = None,
    truth_floor_ratio: float = 0.0,
) -> torch.Tensor:
    n_grid = prediction.shape[-1]
    wave = torch.fft.rfftfreq(n_grid, d=1.0 / n_grid).to(prediction.device)
    weight = 1.0 + wave.square()
    error_hat = torch.fft.rfft(prediction - target, norm="ortho")
    target_hat = torch.fft.rfft(target, norm="ortho")
    numerator = (error_hat.abs().square() * weight).sum(dim=-1)
    denominator = (target_hat.abs().square() * weight).sum(dim=-1)
    if truth is not None and truth_floor_ratio > 0.0:
        truth_hat = torch.fft.rfft(truth, norm="ortho")
        denominator = denominator + truth_floor_ratio * (
            truth_hat.abs().square() * weight
        ).sum(dim=-1)
    denominator = denominator.clamp_min(1.0e-12)
    return torch.sqrt(numerator / denominator).mean()


def selected_spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    prediction_hat = torch.fft.rfft(prediction, norm="ortho")[:, indices]
    target_hat = torch.fft.rfft(target, norm="ortho")[:, indices]
    power = target_hat.abs().square().mean(dim=0).clamp_min(1.0e-10)
    error = (prediction_hat - target_hat).abs().square().mean(dim=0)
    return (error / power).mean()


def periodic_transport_descriptors(field: torch.Tensor) -> torch.Tensor:
    """Return normalized periodic gradient, curvature, and front strength."""
    n_grid = field.shape[-1]
    spectrum = torch.fft.rfft(field, norm="ortho")
    wave = 2.0 * torch.pi * torch.fft.rfftfreq(
        n_grid, d=1.0 / n_grid, device=field.device
    ).to(field.dtype)
    gradient = torch.fft.irfft(1j * wave * spectrum, n=n_grid, norm="ortho")
    curvature = torch.fft.irfft(-wave.square() * spectrum, n=n_grid, norm="ortho")

    def normalize(value: torch.Tensor) -> torch.Tensor:
        scale = value.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-6)
        return value / scale

    return torch.stack(
        (normalize(gradient), normalize(curvature), normalize(gradient.abs())),
        dim=1,
    )


class SelectedSpectralConv1d(nn.Module):
    """Apply a Fourier convolution on grade-specific noncontiguous modes."""

    def __init__(self, in_channels: int, out_channels: int, slots: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.slots = int(slots)
        scale = 1.0 / float(max(1, in_channels * out_channels))
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, slots, 2)
        )
        self.register_buffer("frequency_indices", torch.arange(slots, dtype=torch.long))
        self.register_buffer("frequency_weights", torch.ones(slots))
        self.register_buffer("active_count", torch.tensor(slots, dtype=torch.long))

    def configure(
        self, frequency_indices: torch.Tensor, frequency_weights: torch.Tensor
    ) -> None:
        count = int(frequency_indices.numel())
        if not 1 <= count <= self.slots:
            raise ValueError("Selected Fourier mode count exceeds the allocated slots")
        self.frequency_indices.zero_()
        self.frequency_weights.zero_()
        self.frequency_indices[:count].copy_(
            frequency_indices.to(self.frequency_indices)
        )
        self.frequency_weights[:count].copy_(
            frequency_weights.to(self.frequency_weights)
        )
        self.active_count.fill_(count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_grid = x.shape[-1]
        spectrum = torch.fft.rfft(x, norm="ortho")
        count = int(self.active_count)
        indices = self.frequency_indices[:count]
        if int(indices.max()) >= spectrum.shape[-1]:
            raise RuntimeError("A selected frequency is incompatible with the grid")
        selected = spectrum.index_select(-1, indices)
        weights = torch.view_as_complex(self.weights[:, :, :count].contiguous())
        transformed = torch.einsum("bim,iom->bom", selected, weights)
        transformed = transformed * self.frequency_weights[None, None, :count]
        output = torch.zeros(
            x.shape[0],
            self.out_channels,
            spectrum.shape[-1],
            dtype=spectrum.dtype,
            device=x.device,
        )
        output.index_copy_(-1, indices, transformed)
        return torch.fft.irfft(output, n=n_grid, norm="ortho")


class Grade1Operator(nn.Module):
    """A complete FNO grade for the dominant solution component."""

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
        self.depth = int(depth)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.lifting(x).permute(0, 2, 1)
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers, strict=True)
        ):
            feature = spectral(feature) + pointwise(feature)
            if index < self.depth - 1:
                feature = F.relu(feature)
        return self.projection(feature), feature


class AdaptiveResidualGrade(nn.Module):
    """Predict and correct the residual left by all frozen predecessors."""

    def __init__(
        self,
        modes: int,
        width: int,
        depth: int,
        extra_channels: int = 0,
    ) -> None:
        super().__init__()
        base_channels = width + 6 + int(extra_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.modes = int(modes)
        self.predictor_lift = nn.Conv1d(base_channels, width, 1)
        self.predictor_spectral = SelectedSpectralConv1d(width, width, modes)
        self.predictor_pointwise = nn.Conv1d(width, width, 1)
        scale = 1.0 / max(1, width)
        self.predictor_coefficients = nn.Parameter(
            scale * torch.randn(width, modes, 2)
        )
        self.lifting = nn.Linear(base_channels, width)
        self.spectral_lift_weights = nn.Parameter(
            scale * torch.randn(width, modes, 2)
        )
        self.spectral_lift_gain = nn.Parameter(torch.tensor(0.1))
        self.temporal_memory_gain = nn.Parameter(torch.tensor(0.0))
        self.defect_spectral_gain = nn.Parameter(torch.tensor(0.0))
        self.spectral_layers = nn.ModuleList(
            SelectedSpectralConv1d(width, width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.projection = Projection1d(width)
        self.register_buffer("frequency_weights", torch.ones(modes))
        self.register_buffer("frequency_indices", torch.arange(modes, dtype=torch.long))
        self.register_buffer("frequency_scale", torch.ones(modes))
        self.register_buffer("active_count", torch.tensor(modes, dtype=torch.long))
        self.register_buffer("residual_mean", torch.zeros(()))
        self.register_buffer("residual_std", torch.ones(()))
        self.register_buffer("deployment_gain", torch.ones(()))

    def configure(
        self,
        frequency_weights: torch.Tensor,
        selected_indices: torch.Tensor,
        frequency_scale: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        count = int(selected_indices.numel())
        if not 1 <= count <= self.modes:
            raise ValueError("The selected residual basis exceeds this grade's slots")
        self.frequency_weights.zero_()
        self.frequency_indices.zero_()
        self.frequency_scale.fill_(1.0)
        self.frequency_weights[:count].copy_(
            frequency_weights.to(self.frequency_weights)
        )
        self.frequency_indices[:count].copy_(
            selected_indices.to(self.frequency_indices)
        )
        self.frequency_scale[:count].copy_(
            frequency_scale.to(self.frequency_scale).clamp_min(1.0e-8)
        )
        self.active_count.fill_(count)
        self.predictor_spectral.configure(selected_indices, frequency_weights)
        for layer in self.spectral_layers:
            layer.configure(selected_indices, frequency_weights)
        self.residual_mean.copy_(residual.mean().to(self.residual_mean))
        self.residual_std.copy_(
            residual.std(unbiased=False).clamp_min(1.0e-6).to(self.residual_std)
        )
        self.deployment_gain.fill_(1.0)
        with torch.no_grad():
            # Keep the initial correction near zero without blocking gradients
            # to the residual-spectrum predictor and Fourier lifting path.
            nn.init.xavier_uniform_(self.projection.fc2.weight, gain=1.0e-2)
            self.projection.fc2.bias.fill_(
                float(-self.residual_mean / self.residual_std)
            )

    def _predict_residual(
        self, base_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.predictor_lift(base_input))
        hidden = F.gelu(
            self.predictor_spectral(hidden) + self.predictor_pointwise(hidden)
        )
        hidden_spectrum = torch.fft.rfft(hidden, norm="ortho")
        active = int(self.active_count)
        indices = self.frequency_indices[:active]
        coefficient_weights = torch.view_as_complex(
            self.predictor_coefficients[:, :active].contiguous()
        )
        predicted = torch.zeros(
            hidden.shape[0],
            hidden_spectrum.shape[-1],
            dtype=hidden_spectrum.dtype,
            device=hidden.device,
        )
        normalized_coefficients = torch.einsum(
            "bcm,cm->bm",
            hidden_spectrum.index_select(-1, indices),
            coefficient_weights,
        )
        predicted_coefficients = (
            normalized_coefficients * self.frequency_scale[None, :active]
        )
        predicted.index_copy_(-1, indices, predicted_coefficients)
        physical = torch.fft.irfft(predicted, n=hidden.shape[-1], norm="ortho")
        return physical, predicted

    def _spectral_lifting(
        self, predicted_spectrum: torch.Tensor, n_grid: int
    ) -> torch.Tensor:
        active = int(self.active_count)
        indices = self.frequency_indices[:active]
        basis_weights = torch.view_as_complex(
            self.spectral_lift_weights[:, :active].contiguous()
        )
        lifted_spectrum = torch.zeros(
            predicted_spectrum.shape[0],
            self.width,
            predicted_spectrum.shape[-1],
            dtype=predicted_spectrum.dtype,
            device=predicted_spectrum.device,
        )
        whitened = predicted_spectrum.index_select(-1, indices) / self.frequency_scale[
            None, :active
        ]
        lifted_coefficients = (
            whitened[:, None, :]
            * self.frequency_weights[None, None, :active]
            * basis_weights[None, :, :]
        )
        lifted_spectrum.index_copy_(-1, indices, lifted_coefficients)
        return torch.fft.irfft(lifted_spectrum, n=n_grid, norm="ortho")

    def forward(
        self,
        x: torch.Tensor,
        cumulative: torch.Tensor,
        previous_feature: torch.Tensor,
        spectral_memory: torch.Tensor | None = None,
        dynamic_features: torch.Tensor | None = None,
        dynamic_spectrum: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transport = periodic_transport_descriptors(cumulative)
        channels = [
            x.permute(0, 2, 1),
            cumulative.unsqueeze(1),
            transport,
            previous_feature,
        ]
        if dynamic_features is not None:
            channels.append(dynamic_features)
        base = torch.cat(channels, dim=1)
        _, predicted_spectrum = self._predict_residual(base)
        spectral_context = predicted_spectrum
        if spectral_memory is not None:
            spectral_context = spectral_context + torch.tanh(
                self.temporal_memory_gain
            ) * spectral_memory
        if dynamic_spectrum is not None:
            spectral_context = spectral_context + torch.tanh(
                self.defect_spectral_gain
            ) * dynamic_spectrum
        physical_lift = self.lifting(base.permute(0, 2, 1)).permute(0, 2, 1)
        residual_fourier_lift = self._spectral_lifting(
            spectral_context, physical_lift.shape[-1]
        )
        feature = physical_lift + torch.tanh(
            self.spectral_lift_gain
        ) * residual_fourier_lift
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers, strict=True)
        ):
            feature = spectral(feature) + pointwise(feature)
            if index < self.depth - 1:
                feature = F.relu(feature)
        normalized = self.projection(feature)
        correction = self.residual_mean + self.residual_std * normalized
        return self.deployment_gain * correction, feature, predicted_spectrum


class AdaptiveMultigradeOperator(nn.Module):
    """An arbitrary-length sequence of frozen residual Fourier operators."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        width = config.widths[0]
        self.grade1 = Grade1Operator(config.modes, width, config.depths[0])
        self.residual_grades = nn.ModuleList(
            AdaptiveResidualGrade(capacity, width, depth)
            for capacity, depth in zip(
                config.residual_basis_modes, config.depths[1:], strict=True
            )
        )

    def freeze_before(self, grade_index: int) -> None:
        self.grade1.eval()
        self.grade1.requires_grad_(False)
        for index, grade in enumerate(self.residual_grades):
            if index < grade_index - 1:
                grade.eval()
                grade.requires_grad_(False)

    def forward_up_to(
        self, x: torch.Tensor, active_grades: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        cumulative, feature = self.grade1(x)
        corrections: list[torch.Tensor] = []
        for grade_index in range(1, active_grades):
            correction, feature, _ = self.apply_residual_grade(
                x, cumulative, feature, grade_index
            )
            cumulative = cumulative + correction
            corrections.append(correction)
        return cumulative, feature, corrections

    def apply_residual_grade(
        self,
        x: torch.Tensor,
        cumulative: torch.Tensor,
        previous_feature: torch.Tensor,
        grade_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grade = self.residual_grades[grade_index - 1]
        return grade(x, cumulative, previous_feature)

    def predecessor_state(
        self, x: torch.Tensor, grade_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cumulative, feature, _ = self.forward_up_to(x, grade_index)
        return cumulative, feature


@torch.no_grad()
def collect_predecessor_residual(
    model: AdaptiveMultigradeOperator,
    loader: DataLoader,
    grade_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residuals: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    proxies: list[torch.Tensor] = []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        prediction, _ = model.predecessor_state(x, grade_index)
        residuals.append(y - prediction.cpu())
        targets.append(y)
        proxies.append(
            torch.cat(
                (
                    x[:, :, 0].unsqueeze(1),
                    prediction.unsqueeze(1),
                    periodic_transport_descriptors(prediction),
                ),
                dim=1,
            ).cpu()
        )
    return torch.cat(residuals), torch.cat(targets), torch.cat(proxies)


@torch.no_grad()
def build_frozen_grade_cache(
    model: AdaptiveMultigradeOperator,
    loader: DataLoader,
    grade_index: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    predecessors: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    model.eval()
    for x, y in loader:
        predecessor, feature = model.predecessor_state(x.to(device), grade_index)
        inputs.append(x)
        targets.append(y)
        predecessors.append(predecessor.cpu())
        features.append(feature.cpu())
    dataset = TensorDataset(
        torch.cat(inputs),
        torch.cat(targets),
        torch.cat(predecessors),
        torch.cat(features),
    )
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def analyze_residual(
    residual: torch.Tensor,
    solution: torch.Tensor,
    proxy: torch.Tensor,
    used_frequencies: set[int],
    config: Config,
    basis_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    residual_spectrum = torch.fft.rfft(residual, norm="ortho")
    raw_residual_power = residual_spectrum.abs().square().mean(0)
    raw_solution_power = torch.fft.rfft(solution, norm="ortho").abs().square().mean(0)
    multiplicity = torch.full_like(raw_residual_power, 2.0)
    multiplicity[0] = 1.0
    if residual.shape[-1] % 2 == 0:
        multiplicity[-1] = 1.0
    residual_power = raw_residual_power * multiplicity
    solution_power = raw_solution_power * multiplicity
    proxy_spectrum = torch.fft.rfft(proxy, norm="ortho")
    epsilon = max(float(solution_power.max()) * 1.0e-12, 1.0e-20)
    solution_floor = max(float(solution_power.mean()) * 1.0e-3, epsilon)
    relative = residual_power / (solution_power + solution_floor)
    proxy_power = proxy_spectrum.abs().square().mean(0)
    cross_power = (
        (residual_spectrum[:, None, :] * proxy_spectrum.conj()).mean(0).abs().square()
    )
    coherence = (
        cross_power
        / (raw_residual_power[None, :] * proxy_power + epsilon)
    ).clamp(0.0, 1.0).amax(0)
    beta = config.relative_error_score_power
    coherence_factor = 0.05 + 0.95 * coherence
    base_score = (
        residual_power.clamp_min(epsilon).pow(1.0 - beta)
        * relative.clamp_min(epsilon).pow(beta)
        * coherence_factor.pow(config.coherence_score_power)
    )
    fold_scores: list[torch.Tensor] = []
    for fold in range(config.consensus_folds):
        fold_indices = torch.arange(fold, residual.shape[0], config.consensus_folds)
        fold_residual_spectrum = residual_spectrum.index_select(0, fold_indices)
        fold_solution = solution.index_select(0, fold_indices)
        fold_proxy_spectrum = proxy_spectrum.index_select(0, fold_indices)
        fold_raw_residual_power = (
            fold_residual_spectrum.abs().square().mean(0)
        )
        fold_residual_power = fold_raw_residual_power * multiplicity
        fold_solution_power = (
            torch.fft.rfft(fold_solution, norm="ortho")
            .abs()
            .square()
            .mean(0)
            * multiplicity
        )
        fold_solution_floor = max(
            float(fold_solution_power.mean()) * 1.0e-3,
            epsilon,
        )
        fold_relative = fold_residual_power / (
            fold_solution_power + fold_solution_floor
        )
        fold_proxy_power = fold_proxy_spectrum.abs().square().mean(0)
        fold_cross_power = (
            (
                fold_residual_spectrum[:, None, :]
                * fold_proxy_spectrum.conj()
            )
            .mean(0)
            .abs()
            .square()
        )
        fold_coherence = (
            fold_cross_power
            / (
                fold_raw_residual_power[None, :]
                * fold_proxy_power
                + epsilon
            )
        ).clamp(0.0, 1.0).amax(0)
        fold_score = (
            fold_residual_power.clamp_min(epsilon).pow(1.0 - beta)
            * fold_relative.clamp_min(epsilon).pow(beta)
            * (0.05 + 0.95 * fold_coherence).pow(
                config.coherence_score_power
            )
        )
        fold_scores.append(fold_score / fold_score.max().clamp_min(epsilon))
    consensus_scores = torch.stack(fold_scores)
    consensus_mean = consensus_scores.mean(0)
    consensus_cv = consensus_scores.std(0, unbiased=False) / consensus_mean.clamp_min(
        epsilon
    )
    consensus_stability = 1.0 / (
        1.0 + config.consensus_variation_penalty * consensus_cv
    )
    score = base_score * consensus_stability
    energy_floor = residual_power.max() * config.residual_energy_floor_fraction
    eligible_mask = residual_power >= energy_floor
    eligible_order = torch.argsort(score, descending=True)
    eligible_order = eligible_order[eligible_mask[eligible_order]]
    if eligible_order.numel() < config.basis_min_modes:
        eligible_order = torch.argsort(residual_power, descending=True)[
            : config.basis_min_modes
        ]
    cumulative = torch.cumsum(residual_power[eligible_order], 0) / residual_power.sum().clamp_min(epsilon)
    required = int(
        min(
            torch.searchsorted(cumulative, torch.tensor(config.basis_energy)).item()
            + 1,
            eligible_order.numel(),
        )
    )
    count = min(
        max(required, config.basis_min_modes),
        basis_capacity,
        int(eligible_order.numel()),
    )
    used_mask = torch.zeros_like(score, dtype=torch.bool)
    valid_used = [index for index in used_frequencies if 0 <= index < score.numel()]
    if valid_used:
        used_mask[torch.tensor(valid_used, dtype=torch.long)] = True
    new_candidates = eligible_order[~used_mask[eligible_order]]
    minimum_new = min(config.minimum_new_modes, count, int(new_candidates.numel()))
    maximum_new = max(
        minimum_new,
        min(count, int(round(count * config.new_frequency_fraction))),
    )
    chosen_list = [int(value) for value in new_candidates[:minimum_new]]
    new_count = len(chosen_list)
    for value in eligible_order.tolist():
        if len(chosen_list) >= count:
            break
        if value in chosen_list:
            continue
        is_new = not bool(used_mask[value])
        if is_new and new_count >= maximum_new:
            continue
        chosen_list.append(int(value))
        new_count += int(is_new)
    selected = torch.sort(torch.tensor(chosen_list, dtype=torch.long)).values
    soft = torch.full_like(score, config.soft_frequency_floor)
    selected_by_score = selected[torch.argsort(score[selected], descending=True)]
    ranked_weights = torch.linspace(
        1.0,
        max(config.soft_frequency_floor, 0.25),
        selected_by_score.numel(),
    )
    soft[selected_by_score] = ranked_weights
    captured = float(residual_power[selected].sum() / residual_power.sum().clamp_min(epsilon))
    oracle_remaining_fraction = max(0.0, 1.0 - captured)
    oracle_improvement = 1.0 / max(oracle_remaining_fraction, 1.0e-20) ** 0.5
    roles = ["repair" if int(index) in used_frequencies else "new" for index in selected]
    metadata = {
        "selected_indices": selected.tolist(),
        "selected_roles": roles,
        "new_count": roles.count("new"),
        "repair_count": roles.count("repair"),
        "captured_energy": captured,
        "oracle_remaining_energy_fraction": oracle_remaining_fraction,
        "oracle_residual_improvement": oracle_improvement,
        "basis_capacity": basis_capacity,
        "solution_power_floor": solution_floor,
        "residual_power_floor": float(energy_floor),
        "residual_power": residual_power.tolist(),
        "raw_residual_power": raw_residual_power.tolist(),
        "solution_power": solution_power.tolist(),
        "relative_error_power": relative.tolist(),
        "predictive_coherence": coherence.tolist(),
        "base_selection_score": base_score.tolist(),
        "consensus_mean_score": consensus_mean.tolist(),
        "consensus_variation": consensus_cv.tolist(),
        "consensus_stability": consensus_stability.tolist(),
        "selection_score": score.tolist(),
        "soft_frequency_weights": soft.tolist(),
    }
    return selected, soft[selected], metadata


def save_profile(path: Path, grade: int, metadata: dict[str, object]) -> None:
    selected = set(int(value) for value in metadata["selected_indices"])
    role_by_frequency = dict(
        zip(
            (int(value) for value in metadata["selected_indices"]),
            metadata["selected_roles"],
            strict=True,
        )
    )
    rows = zip(
        metadata["residual_power"],
        metadata["solution_power"],
        metadata["relative_error_power"],
        metadata["predictive_coherence"],
        metadata["base_selection_score"],
        metadata["consensus_mean_score"],
        metadata["consensus_variation"],
        metadata["consensus_stability"],
        metadata["selection_score"],
        metadata["soft_frequency_weights"],
        strict=True,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "grade",
                "frequency_index",
                "selected",
                "new_or_repair",
                "residual_power",
                "solution_power",
                "relative_error_power",
                "predictive_coherence",
                "base_selection_score",
                "consensus_mean_score",
                "consensus_variation",
                "consensus_stability",
                "selection_score",
                "soft_frequency_weight",
            )
        )
        for index, values in enumerate(rows):
            writer.writerow(
                (
                    grade,
                    index,
                    int(index in selected),
                    role_by_frequency.get(index, "unselected"),
                    *values,
                )
            )


@torch.no_grad()
def save_grade_spectral_diagnostic(
    model: AdaptiveMultigradeOperator,
    loader: DataLoader,
    grade_index: int,
    metadata: dict[str, object],
    device: torch.device,
    path: Path,
) -> None:
    """Visualize what one residual grade learned using training data only."""
    grade = model.residual_grades[grade_index - 1]
    model.eval()
    targets: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    predictors: list[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predecessor, previous_feature = model.predecessor_state(x, grade_index)
        correction, _, predicted_spectrum = model.apply_residual_grade(
            x, predecessor, previous_feature, grade_index
        )
        targets.append((y - predecessor).cpu())
        outputs.append(correction.cpu())
        predictors.append(predicted_spectrum.cpu())
    target = torch.cat(targets)
    output = torch.cat(outputs)
    predicted_spectrum = torch.cat(predictors)
    target_spectrum = torch.fft.rfft(target, norm="ortho")
    output_spectrum = torch.fft.rfft(output, norm="ortho")
    remaining_spectrum = target_spectrum - output_spectrum
    target_amplitude = target_spectrum.abs().mean(0).numpy()
    predicted_amplitude = predicted_spectrum.abs().mean(0).numpy()
    output_amplitude = output_spectrum.abs().mean(0).numpy()
    remaining_amplitude = remaining_spectrum.abs().mean(0).numpy()
    frequency = np.arange(target_amplitude.size)
    selected = np.asarray(metadata["selected_indices"], dtype=np.int64)
    roles = np.asarray(metadata["selected_roles"], dtype=object)
    score = np.asarray(metadata["selection_score"], dtype=np.float64)
    coherence = np.asarray(metadata["predictive_coherence"], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].semilogy(frequency, target_amplitude + 1.0e-14, label="Target residual")
    axes[0, 0].semilogy(
        frequency, predicted_amplitude + 1.0e-14, label="Predicted residual"
    )
    axes[0, 0].semilogy(frequency, output_amplitude + 1.0e-14, label="Grade output")
    axes[0, 0].semilogy(
        frequency, remaining_amplitude + 1.0e-14, label="Remaining residual"
    )
    axes[0, 0].set_title("Residual-spectrum learning")
    axes[0, 0].set_xlabel("Wavenumber")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(frequency, score + 1.0e-30, label="Selection score")
    axes[0, 1].plot(frequency, coherence, label="Predictive coherence")
    for role, color, marker in (("new", "tab:red", "^"), ("repair", "tab:green", "o")):
        indices = selected[roles == role]
        if indices.size:
            axes[0, 1].scatter(
                indices,
                score[indices] + 1.0e-30,
                color=color,
                marker=marker,
                s=28,
                label=f"Selected {role}",
                zorder=3,
            )
    axes[0, 1].set_title("Adaptive basis selection")
    axes[0, 1].set_xlabel("Wavenumber")
    axes[0, 1].legend(fontsize=8)

    grid = np.arange(target.shape[-1])
    axes[1, 0].plot(grid, target[0], color="black", label="Target correction")
    axes[1, 0].plot(grid, output[0], color="tab:blue", label="Grade output")
    axes[1, 0].plot(
        grid, target[0] - output[0], color="tab:red", label="Remaining"
    )
    axes[1, 0].set_title("Physical residual correction")
    axes[1, 0].set_xlabel("Grid index")
    axes[1, 0].legend(fontsize=8)

    before = target.square().sum(-1).sqrt()
    after = (target - output).square().sum(-1).sqrt()
    improvement = before / after.clamp_min(1.0e-12)
    axes[1, 1].hist(improvement.numpy(), bins=min(30, improvement.numel()))
    axes[1, 1].axvline(1.0, color="black", linestyle="--")
    axes[1, 1].set_title("Per-sample residual improvement")
    axes[1, 1].set_xlabel("Before/after residual L2")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Grade {grade_index + 1}: adaptive residual-spectrum evidence")
    fig.savefig(path, format="svg", dpi=300)
    plt.close(fig)


def save_checkpoint(
    model: nn.Module,
    directory: Path,
    name: str,
    epoch: int,
    active_grades: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "active_grades": active_grades,
        },
        directory / f"{name}_epoch_{epoch:04d}.pt",
    )


def train_grade1(
    model: AdaptiveMultigradeOperator,
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> list[dict[str, object]]:
    optimizer = torch.optim.Adam(
        model.grade1.parameters(),
        lr=config.learning_rates[0],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.grade_scheduler_steps[0],
        gamma=config.grade_scheduler_gammas[0],
    )
    history: list[dict[str, object]] = []
    global_epoch = 0
    for epoch in range(1, config.epochs[0] + 1):
        start = time.perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        objective_sum = relative_sum = mse_sum = 0.0
        samples = 0
        model.grade1.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model.grade1(x)
            relative = sample_relative_l2(prediction, y).mean()
            relative.backward()
            optimizer.step()
            batch = x.shape[0]
            objective_sum += float(relative.detach()) * batch
            relative_sum += float(relative.detach()) * batch
            mse_sum += float(F.mse_loss(prediction, y).detach()) * batch
            samples += batch
        scheduler.step()
        global_epoch += 1
        row = {
            "grade_name": "grade1",
            "grade": 1,
            "epoch": epoch,
            "global_epoch": global_epoch,
            "learning_rate": epoch_learning_rate,
            "objective": objective_sum / samples,
            "train_mse": mse_sum / samples,
            "train_relative_l2": relative_sum / samples,
            "elapsed_seconds": time.perf_counter() - start,
            "test_relative_l2": "",
        }
        history.append(row)
        save_checkpoint(
            model,
            config.output_dir / "_epoch_checkpoints" / "multigrade",
            "multigrade",
            global_epoch,
            1,
        )
        print(
            f"Grade1 epoch={epoch:04d} "
            f"epoch_time={row['elapsed_seconds']:.6f}s "
            f"lr={row['learning_rate']:.6e} "
            f"objective={row['objective']:.6e} "
            f"train_mse={row['train_mse']:.6e} "
            f"train_l2={row['train_relative_l2']:.6e}"
        )
    model.grade1.eval()
    model.grade1.requires_grad_(False)
    return history


def train_residual_grade(
    model: AdaptiveMultigradeOperator,
    loader: DataLoader,
    config: Config,
    device: torch.device,
    grade_index: int,
    global_offset: int,
    used_frequencies: set[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.freeze_before(grade_index)
    snapshots = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("grade1")
        or any(name.startswith(f"residual_grades.{i}.") for i in range(grade_index - 1))
    }
    residual, solution, proxy = collect_predecessor_residual(
        model, loader, grade_index, device
    )
    selected, soft, metadata = analyze_residual(
        residual,
        solution,
        proxy,
        used_frequencies,
        config,
        config.residual_basis_modes[grade_index - 1],
    )
    grade = model.residual_grades[grade_index - 1]
    residual_power = torch.as_tensor(metadata["raw_residual_power"])
    frequency_scale = residual_power[selected].sqrt().clamp_min(1.0e-8)
    grade.configure(soft, selected, frequency_scale, residual)
    cached_loader = None
    if config.cache_frozen_features:
        cached_loader = build_frozen_grade_cache(
            model,
            loader,
            grade_index,
            device,
            config.seed + grade_index,
        )
        print(
            f"Grade {grade_index + 1} frozen predecessor cache: "
            f"{len(cached_loader.dataset)} samples"
        )
    save_profile(
        config.output_dir / f"grade_{grade_index + 1:02d}_residual_profile.csv",
        grade_index + 1,
        metadata,
    )
    print(
        f"Grade {grade_index + 1} adaptive residual profile: "
        f"selected={selected.tolist()} | "
        f"new={metadata['new_count']} repair={metadata['repair_count']} | "
        f"captured={100.0 * float(metadata['captured_energy']):.3f}% | "
        f"oracle={float(metadata['oracle_residual_improvement']):.3f}x"
    )
    optimizer = torch.optim.Adam(
        grade.parameters(),
        lr=config.learning_rates[grade_index],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.grade_scheduler_steps[grade_index],
        gamma=config.grade_scheduler_gammas[grade_index],
    )
    history: list[dict[str, object]] = []
    selected_device = selected.to(device)
    for epoch in range(1, config.epochs[grade_index] + 1):
        start = time.perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        objective_sum = relative_sum = mse_sum = 0.0
        samples = 0
        grade.train()
        grade_loader = cached_loader if cached_loader is not None else loader
        for batch_values in grade_loader:
            if cached_loader is not None:
                x, y, predecessor, previous_feature = (
                    value.to(device) for value in batch_values
                )
            else:
                x, y = (value.to(device) for value in batch_values)
                with torch.no_grad():
                    predecessor, previous_feature = model.predecessor_state(
                        x, grade_index
                    )
            residual_target = y - predecessor
            optimizer.zero_grad(set_to_none=True)
            correction, _, predicted_spectrum = model.apply_residual_grade(
                x, predecessor, previous_feature, grade_index
            )
            final = predecessor + correction
            final_relative = sample_relative_l2(final, y).mean()
            normalized_correction = (
                correction - grade.residual_mean
            ) / grade.residual_std
            normalized_target = (
                residual_target - grade.residual_mean
            ) / grade.residual_std
            residual_mse = F.mse_loss(normalized_correction, normalized_target)
            spectral = selected_spectral_loss(
                correction, residual_target, selected_device
            )
            h1 = relative_h1(
                correction,
                residual_target,
                truth=y,
                truth_floor_ratio=config.h1_truth_floor_ratio,
            )
            target_spectrum = torch.fft.rfft(residual_target, norm="ortho")
            predictor_power = target_spectrum[:, selected_device].abs().square().mean(0).clamp_min(1.0e-10)
            predictor = (
                (predicted_spectrum[:, selected_device] - target_spectrum[:, selected_device])
                .abs()
                .square()
                .mean(0)
                / predictor_power
            ).mean()
            objective = (
                final_relative
                + config.residual_mse_weight * residual_mse
                + config.spectral_weights[grade_index] * spectral
                + config.h1_weights[grade_index] * h1
                + config.predictor_weights[grade_index] * predictor
            )
            objective.backward()
            optimizer.step()
            batch = x.shape[0]
            objective_sum += float(objective.detach()) * batch
            relative_sum += float(final_relative.detach()) * batch
            mse_sum += float(F.mse_loss(final, y).detach()) * batch
            samples += batch
        scheduler.step()
        global_epoch = global_offset + epoch
        row = {
            "grade_name": f"grade{grade_index + 1}",
            "grade": grade_index + 1,
            "epoch": epoch,
            "global_epoch": global_epoch,
            "learning_rate": epoch_learning_rate,
            "objective": objective_sum / samples,
            "train_mse": mse_sum / samples,
            "train_relative_l2": relative_sum / samples,
            "elapsed_seconds": time.perf_counter() - start,
            "test_relative_l2": "",
        }
        history.append(row)
        save_checkpoint(
            model,
            config.output_dir / "_epoch_checkpoints" / "multigrade",
            "multigrade",
            global_epoch,
            grade_index + 1,
        )
        print(
            f"Grade{grade_index + 1} epoch={epoch:04d} "
            f"epoch_time={row['elapsed_seconds']:.6f}s "
            f"lr={row['learning_rate']:.6e} "
            f"objective={row['objective']:.6e} "
            f"train_mse={row['train_mse']:.6e} "
            f"train_l2={row['train_relative_l2']:.6e}"
        )
    numerator = 0.0
    denominator = 0.0
    grade.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predecessor, previous_feature = model.predecessor_state(
                x, grade_index
            )
            correction, _, _ = model.apply_residual_grade(
                x, predecessor, previous_feature, grade_index
            )
            residual_target = y - predecessor
            numerator += float((residual_target * correction).sum())
            denominator += float(correction.square().sum())
    calibrated_gain = float(
        np.clip(numerator / max(denominator, 1.0e-20), 0.0, 1.0)
    )
    grade.deployment_gain.fill_(calibrated_gain)
    metadata["training_calibrated_deployment_gain"] = calibrated_gain
    save_checkpoint(
        model,
        config.output_dir / "_epoch_checkpoints" / "multigrade",
        "multigrade",
        global_offset + config.epochs[grade_index],
        grade_index + 1,
    )
    print(
        f"Grade {grade_index + 1} training-calibrated correction gain: "
        f"{calibrated_gain:.6f}"
    )
    diagnostic_path = (
        config.output_dir
        / f"grade_{grade_index + 1:02d}_adaptive_residual_spectrum.svg"
    )
    save_grade_spectral_diagnostic(
        model,
        loader,
        grade_index,
        metadata,
        device,
        diagnostic_path,
    )
    metadata["diagnostic_svg"] = str(diagnostic_path.resolve())
    for name, value in model.state_dict().items():
        if name in snapshots and not torch.equal(value, snapshots[name]):
            raise RuntimeError(f"Frozen predecessor changed: {name}")
    grade.eval()
    grade.requires_grad_(False)
    return history, metadata


@torch.no_grad()
def evaluate_multigrade(
    model: AdaptiveMultigradeOperator,
    loader: DataLoader,
    device: torch.device,
    active_grades: int,
) -> dict[str, object]:
    model.eval()
    relative = squared = 0.0
    elements = samples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x, y in loader:
        x, y_device = x.to(device), y.to(device)
        prediction, _, _ = model.forward_up_to(x, active_grades)
        relative += float(relative_l2_sum(prediction, y_device))
        squared += float(pointwise_mse_sum(prediction, y_device))
        elements += y_device.numel()
        samples += x.shape[0]
        predictions.append(prediction.cpu())
        targets.append(y)
    return {
        "relative_l2": relative / samples,
        "mse": squared / elements,
        "prediction": torch.cat(predictions),
        "target": torch.cat(targets),
    }


def train_baseline(
    train_data: tuple[torch.Tensor, torch.Tensor],
    config: Config,
    device: torch.device,
) -> tuple[FNO4Baseline1d, list[dict[str, object]]]:
    set_seed(config.seed)
    model = FNO4Baseline1d(config.modes, config.widths[0]).to(device)
    loader = make_loader(
        *train_data,
        config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-3, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.scheduler_step, gamma=config.scheduler_gamma
    )
    history: list[dict[str, object]] = []
    for epoch in range(1, config.baseline_epochs + 1):
        start = time.perf_counter()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        relative_sum = mse_sum = 0.0
        samples = 0
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            relative = sample_relative_l2(prediction, y).mean()
            relative.backward()
            optimizer.step()
            batch = x.shape[0]
            relative_sum += float(relative.detach()) * batch
            mse_sum += float(F.mse_loss(prediction, y).detach()) * batch
            samples += batch
        scheduler.step()
        row = {
            "grade_name": "fno4",
            "grade": 0,
            "epoch": epoch,
            "global_epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "objective": relative_sum / samples,
            "train_mse": mse_sum / samples,
            "train_relative_l2": relative_sum / samples,
            "elapsed_seconds": time.perf_counter() - start,
            "test_relative_l2": "",
        }
        history.append(row)
        save_checkpoint(
            model,
            config.output_dir / "_epoch_checkpoints" / "fno4",
            "fno4",
            epoch,
            0,
        )
        print(
            f"FNO4 epoch={epoch:04d} "
            f"epoch_time={row['elapsed_seconds']:.6f}s "
            f"lr={row['learning_rate']:.6e} "
            f"objective={row['objective']:.6e} "
            f"train_mse={row['train_mse']:.6e} "
            f"train_l2={row['train_relative_l2']:.6e}"
        )
    model.eval()
    model.requires_grad_(False)
    return model, history


@torch.no_grad()
def evaluate_baseline(
    model: FNO4Baseline1d,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    relative = squared = 0.0
    elements = samples = 0
    predictions: list[torch.Tensor] = []
    for x, y in loader:
        x, y_device = x.to(device), y.to(device)
        prediction = model(x)
        relative += float(relative_l2_sum(prediction, y_device))
        squared += float(pointwise_mse_sum(prediction, y_device))
        elements += y_device.numel()
        samples += x.shape[0]
        predictions.append(prediction.cpu())
    return {
        "relative_l2": relative / samples,
        "mse": squared / elements,
        "prediction": torch.cat(predictions),
    }


def post_training_test_audit(
    model: AdaptiveMultigradeOperator,
    baseline: FNO4Baseline1d,
    history: list[dict[str, object]],
    loader: DataLoader,
    config: Config,
    device: torch.device,
) -> None:
    if not config.post_training_test_audit:
        return
    print("[Post-training audit] Test data are accessed only after all optimization")
    multigrade_dir = config.output_dir / "_epoch_checkpoints" / "multigrade"
    fno_dir = config.output_dir / "_epoch_checkpoints" / "fno4"
    for index, row in enumerate(history, 1):
        epoch = int(row["global_epoch"])
        if row["grade_name"] == "fno4":
            payload = torch.load(
                fno_dir / f"fno4_epoch_{epoch:04d}.pt",
                map_location=device,
                weights_only=True,
            )
            baseline.load_state_dict(payload["model_state_dict"])
            metric = evaluate_baseline(baseline, loader, device)["relative_l2"]
        else:
            payload = torch.load(
                multigrade_dir / f"multigrade_epoch_{epoch:04d}.pt",
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(payload["model_state_dict"])
            active = int(payload["active_grades"])
            metric = evaluate_multigrade(model, loader, device, active)["relative_l2"]
        row["test_relative_l2"] = metric
        print(
            f"  audit={index:04d}/{len(history):04d} "
            f"grade={row['grade_name']} epoch={int(row['epoch']):04d} "
            f"test_l2={float(metric):.6e}"
        )


def save_reports(
    history: list[dict[str, object]],
    final_metrics: dict[str, object],
    baseline_metrics: dict[str, object],
    config: Config,
) -> list[Path]:
    fields = list(history[0])
    with (config.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    multigrade = [row for row in history if row["grade_name"] != "fno4"]
    fno = [row for row in history if row["grade_name"] == "fno4"]
    paths: list[Path] = []
    for filename, key, ylabel, logarithmic in (
        ("01_training_mse.svg", "train_mse", "Training MSE", True),
        (
            "02_training_relative_l2.svg",
            "train_relative_l2",
            "Training relative L2",
            True,
        ),
        ("07_test_relative_l2.svg", "test_relative_l2", "Test relative L2", True),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plot = ax.semilogy if logarithmic else ax.plot
        plot(
            [row["global_epoch"] for row in multigrade],
            [row[key] for row in multigrade],
            color="tab:blue",
            marker="o",
            markersize=1.5,
            linewidth=1.1,
            label="Adaptive MGNO",
        )
        plot(
            [row["global_epoch"] for row in fno],
            [row[key] for row in fno],
            color="tab:orange",
            marker="s",
            markersize=1.5,
            linewidth=1.1,
            label="Original FNO-4",
        )
        offset = 0
        for grade_index, epochs in enumerate(config.epochs[:-1], 1):
            offset += epochs
            ax.axvline(
                offset,
                color="tab:gray",
                linestyle="--" if grade_index == 1 else ":",
                alpha=0.6,
                label=f"Grade {grade_index} frozen",
            )
        ax.set_xlabel("Global epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend()
        path = config.output_dir / filename
        fig.savefig(path, format="svg", dpi=300)
        plt.close(fig)
        paths.append(path)
    target = final_metrics["target"]
    prediction = final_metrics["prediction"]
    baseline_prediction = baseline_metrics["prediction"]
    assert isinstance(target, torch.Tensor)
    assert isinstance(prediction, torch.Tensor)
    assert isinstance(baseline_prediction, torch.Tensor)
    grid = np.arange(target.shape[-1])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(grid, target[0], color="black", label="Target")
    axes[0].plot(grid, prediction[0], color="tab:blue", label="Adaptive MGNO")
    axes[0].plot(grid, baseline_prediction[0], color="tab:orange", label="FNO-4")
    axes[0].legend()
    axes[1].semilogy(
        grid, (target[0] - prediction[0]).abs() + 1.0e-12, color="tab:blue", label="MGNO error"
    )
    axes[1].semilogy(
        grid,
        (target[0] - baseline_prediction[0]).abs() + 1.0e-12,
        color="tab:orange",
        label="FNO-4 error",
    )
    axes[1].legend()
    axes[1].set_xlabel("Grid index")
    path = config.output_dir / "08_prediction_and_absolute_error.svg"
    fig.tight_layout()
    fig.savefig(path, format="svg", dpi=300)
    plt.close(fig)
    paths.append(path)
    return paths


def run(config: Config) -> dict[str, object]:
    config.validate()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory is not empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device(config.device)
    x_train, y_train = load_burgers_train_data(config)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = make_loader(
        x_train,
        y_train,
        config.batch_size,
        shuffle=True,
        generator=generator,
    )
    analysis_loader = make_loader(
        x_train, y_train, config.batch_size, shuffle=False
    )
    model = AdaptiveMultigradeOperator(config).to(device)
    histories = train_grade1(model, train_loader, config, device)
    profiles: list[dict[str, object]] = []
    used_frequencies = set(range(config.modes))
    offset = config.epochs[0]
    for grade_index in range(1, config.grade_count):
        grade_history, profile = train_residual_grade(
            model,
            train_loader,
            config,
            device,
            grade_index,
            offset,
            used_frequencies,
        )
        histories.extend(grade_history)
        profiles.append(profile)
        used_frequencies.update(int(value) for value in profile["selected_indices"])
        offset += config.epochs[grade_index]
    baseline, baseline_history = train_baseline(
        (x_train, y_train), config, device
    )
    histories.extend(baseline_history)
    print("[Final test] All MGNO grades and FNO weights are frozen")
    x_test, y_test = load_burgers_test_data(config)
    test_loader = make_loader(x_test, y_test, config.batch_size, shuffle=False)
    post_training_test_audit(
        model, baseline, histories, test_loader, config, device
    )
    final_metrics = evaluate_multigrade(
        model, test_loader, device, config.grade_count
    )
    baseline_metrics = evaluate_baseline(baseline, test_loader, device)
    figures = save_reports(
        histories, final_metrics, baseline_metrics, config
    )
    result = {
        "config": {
            **asdict(config),
            "data": str(config.data.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "protocol": {
            "grade_rule": "Every later grade learns the cumulative residual left by all frozen predecessors",
            "frequency_rule": "Every later grade selects a frozen noncontiguous basis from the complete predecessor-residual spectrum",
            "frequency_roles": "New and repair frequencies receive an explicit grade-level allocation",
            "spectral_energy": "One-sided RFFT energy uses Parseval multiplicity for every non-DC and non-Nyquist mode",
            "oracle_ceiling": "Every residual grade reports the exact projection ceiling of its frozen selected basis before optimization",
            "grade_capacity": "Each residual grade owns an independent adaptive-frequency capacity while every grade remains shallow",
            "selection_rule": "Energy, relative error, train-only predictive coherence, and cross-fold stability jointly rank residual modes",
            "consensus_rule": "Residual modes must remain stable across disjoint training folds; validation and test samples are never used for basis selection",
            "frozen_cache": "Frozen predecessor outputs and final Fourier features are computed once and reused without changing the multigrade objective",
            "sample_adaptation": "An inference-available predictor maps transport-aware predecessor state to complex residual coefficients",
            "spectral_lifting": "Whitened complex residual coefficients generate a trainable Fourier feature map before the complete Fourier core",
            "test_rule": "The held-out test is first accessed after all MGNO and FNO optimization",
            "early_stopping": False,
            "grade_scheduler": "Every MGNO grade has an independent fixed StepLR schedule; the paper FNO schedule remains separate",
        },
        "profiles": profiles,
        "metrics": {
            "mgno_test_relative_l2": final_metrics["relative_l2"],
            "fno4_test_relative_l2": baseline_metrics["relative_l2"],
            "fno4_over_mgno": float(baseline_metrics["relative_l2"])
            / float(final_metrics["relative_l2"]),
            "mgno_parameters": parameter_count(model),
            "fno4_parameters": parameter_count(baseline),
        },
        "figures": [str(path.resolve()) for path in figures],
    }
    with (config.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    torch.save(
        {"config": result["config"], "model_state_dict": model.state_dict()},
        config.output_dir / "adaptive_multigrade_operator.pt",
    )
    torch.save(
        baseline.state_dict(), config.output_dir / "fno4_checkpoint.pt"
    )
    if not config.keep_epoch_checkpoints:
        checkpoint_dir = config.output_dir / "_epoch_checkpoints"
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)
    print("\nFINAL ADAPTIVE MULTIGRADE RESULT")
    print(f"  grades: {config.grade_count}")
    print(f"  MGNO test relative L2: {float(final_metrics['relative_l2']):.6e}")
    print(f"  MGNO parameters: {parameter_count(model)}")
    print(f"  FNO-4 test relative L2: {float(baseline_metrics['relative_l2']):.6e}")
    print(f"  FNO-4 parameters: {parameter_count(baseline)}")
    print(f"  FNO/MGNO gain: {result['metrics']['fno4_over_mgno']:.3f}x")
    print(f"  Output: {config.output_dir.resolve()}")
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Adaptive arbitrary-depth multigrade FNO for Burgers"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            r"D:\Workspace_codex_Fno\fourier-neural-operator-main\data\burgers_data_R10.mat"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--widths", nargs="+", type=int, default=(64, 64))
    parser.add_argument("--depths", nargs="+", type=int, default=(2, 2))
    parser.add_argument("--epochs", nargs="+", type=int, default=(200, 300))
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=(1.0e-3, 1.0e-3)
    )
    parser.add_argument("--grade-scheduler-steps", nargs="+", type=int)
    parser.add_argument("--grade-scheduler-gammas", nargs="+", type=float)
    parser.add_argument("--baseline-epochs", type=int, default=500)
    parser.add_argument("--baseline-depth", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--basis-min-modes", type=int, default=4)
    parser.add_argument("--basis-max-modes", type=int, default=16)
    parser.add_argument("--residual-basis-modes", nargs="+", type=int)
    parser.add_argument("--basis-energy", type=float, default=0.997)
    parser.add_argument("--relative-error-score-power", type=float, default=0.25)
    parser.add_argument("--coherence-score-power", type=float, default=0.5)
    parser.add_argument("--consensus-folds", type=int, default=5)
    parser.add_argument("--consensus-variation-penalty", type=float, default=1.0)
    parser.add_argument("--new-frequency-fraction", type=float, default=0.625)
    parser.add_argument("--residual-energy-floor-fraction", type=float, default=1.0e-3)
    parser.add_argument("--minimum-new-modes", type=int, default=2)
    parser.add_argument("--soft-frequency-floor", type=float, default=0.05)
    parser.add_argument("--residual-mse-weight", type=float, default=0.25)
    parser.add_argument("--spectral-weights", nargs="+", type=float, default=(0.0, 0.05))
    parser.add_argument("--h1-weights", nargs="+", type=float, default=(0.0, 0.02))
    parser.add_argument("--h1-truth-floor-ratio", type=float, default=1.0e-6)
    parser.add_argument("--predictor-weights", nargs="+", type=float, default=(0.0, 0.10))
    parser.add_argument("--run-fno4-baseline", action="store_true")
    parser.add_argument("--verbose-every", type=int, default=1)
    parser.add_argument("--disable-post-training-test-audit", action="store_true")
    parser.add_argument("--disable-frozen-feature-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    grade_count = len(args.depths)
    grade_scheduler_steps = tuple(
        args.grade_scheduler_steps
        if args.grade_scheduler_steps is not None
        else [args.scheduler_step] * grade_count
    )
    grade_scheduler_gammas = tuple(
        args.grade_scheduler_gammas
        if args.grade_scheduler_gammas is not None
        else [args.scheduler_gamma] * grade_count
    )
    residual_basis_modes = tuple(
        args.residual_basis_modes
        if args.residual_basis_modes is not None
        else [args.basis_max_modes] * (grade_count - 1)
    )
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
        widths=tuple(args.widths),
        depths=tuple(args.depths),
        epochs=tuple(args.epochs),
        learning_rates=tuple(args.learning_rates),
        grade_scheduler_steps=grade_scheduler_steps,
        grade_scheduler_gammas=grade_scheduler_gammas,
        baseline_epochs=args.baseline_epochs,
        baseline_depth=args.baseline_depth,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        basis_min_modes=args.basis_min_modes,
        basis_max_modes=args.basis_max_modes,
        residual_basis_modes=residual_basis_modes,
        basis_energy=args.basis_energy,
        relative_error_score_power=args.relative_error_score_power,
        coherence_score_power=args.coherence_score_power,
        consensus_folds=args.consensus_folds,
        consensus_variation_penalty=args.consensus_variation_penalty,
        new_frequency_fraction=args.new_frequency_fraction,
        residual_energy_floor_fraction=args.residual_energy_floor_fraction,
        minimum_new_modes=args.minimum_new_modes,
        soft_frequency_floor=args.soft_frequency_floor,
        residual_mse_weight=args.residual_mse_weight,
        spectral_weights=tuple(args.spectral_weights),
        h1_weights=tuple(args.h1_weights),
        h1_truth_floor_ratio=args.h1_truth_floor_ratio,
        predictor_weights=tuple(args.predictor_weights),
        run_fno4_baseline=args.run_fno4_baseline,
        verbose_every=args.verbose_every,
        post_training_test_audit=not args.disable_post_training_test_audit,
        cache_frozen_features=not args.disable_frozen_feature_cache,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
