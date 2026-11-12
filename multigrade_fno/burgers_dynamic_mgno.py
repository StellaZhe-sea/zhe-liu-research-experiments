"""Dynamic defect-preconditioned multigrade neural operator for Burgers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import burgers_adaptive_multigrade_operator as base
from burgers_mgno_best_recovery import _read_mat_field, make_coordinate_grid
from burgers_temporal_data import BurgersETDRK4, TrajectorySpec


@dataclass
class Config(base.Config):
    """Configuration for inference-available dynamic Burgers corrections."""

    viscosity: float = 0.1
    final_time: float = 1.0
    defect_shift: float = 1.0e-3
    gate_floor_ratio: float = 1.0e-2
    gate_noise_multiplier: float = 3.0
    gate_soft_floor: float = 5.0e-2
    gate_temperature_ratio: float = 1.5e-1
    coarse_resolution: int = 128
    coarse_solver_steps: int = 16
    validation_start: int = 1000
    validation_size: int = 200
    minimum_validation_gain: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.viscosity <= 0.0 or self.final_time <= 0.0:
            raise ValueError("Viscosity and final time must be positive")
        if self.defect_shift < 0.0:
            raise ValueError("The defect shift must be nonnegative")
        if self.gate_floor_ratio < 0.0 or self.gate_noise_multiplier < 0.0:
            raise ValueError("Dynamic gate thresholds must be nonnegative")
        if not 0.0 <= self.gate_soft_floor < 1.0:
            raise ValueError("The dynamic gate floor must lie in [0, 1)")
        if self.gate_temperature_ratio <= 0.0:
            raise ValueError("The dynamic gate temperature must be positive")
        if self.resolution % self.coarse_resolution:
            raise ValueError("The coarse resolution must divide the learning grid")
        if self.coarse_solver_steps <= 0:
            raise ValueError("Coarse solver steps must be positive")
        if self.validation_start < self.ntrain or self.validation_size <= 0:
            raise ValueError("Validation data must begin after training data")
        if not 0.0 <= self.minimum_validation_gain < 1.0:
            raise ValueError("Minimum validation gain must lie in [0, 1)")


def load_validation_data(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Load validation samples that are disjoint from training and final test data."""

    stride = 8192 // config.resolution
    inputs = _read_mat_field(config.data, "a")[:, ::stride]
    targets = _read_mat_field(config.data, "u")[:, ::stride]
    stop = config.validation_start + config.validation_size
    final_test_start = inputs.shape[0] - config.ntest
    if stop > final_test_start:
        raise ValueError("Validation samples overlap the final held-out test set")
    x = torch.from_numpy(inputs[config.validation_start : stop]).float()
    y = torch.from_numpy(targets[config.validation_start : stop]).float()
    grid = make_coordinate_grid(config.resolution)
    x = torch.cat((x.unsqueeze(-1), grid.repeat(x.shape[0], 1, 1)), dim=-1)
    return x, y


def sample_rms(field: torch.Tensor, floor: float = 1.0e-8) -> torch.Tensor:
    return field.square().mean(-1, keepdim=True).sqrt().clamp_min(floor)


def spectral_derivatives(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_grid = field.shape[-1]
    wave = 2.0 * math.pi * torch.fft.rfftfreq(
        n_grid, d=1.0 / n_grid, device=field.device
    )
    spectrum = torch.fft.rfft(field, norm="ortho")
    gradient = torch.fft.irfft(
        1j * wave[None, :] * spectrum, n=n_grid, norm="ortho"
    )
    laplacian = torch.fft.irfft(
        -wave[None, :].square() * spectrum, n=n_grid, norm="ortho"
    )
    return gradient, laplacian


def spectral_resize(field: torch.Tensor, target_points: int) -> torch.Tensor:
    source_points = field.shape[-1]
    source_spectrum = torch.fft.rfft(field, norm="ortho")
    target_modes = target_points // 2 + 1
    resized_spectrum = torch.zeros(
        field.shape[0],
        target_modes,
        dtype=source_spectrum.dtype,
        device=field.device,
    )
    common = min(source_spectrum.shape[-1], target_modes)
    resized_spectrum[:, :common] = source_spectrum[:, :common] * (
        target_points / source_points
    ) ** 0.5
    return torch.fft.irfft(resized_spectrum, n=target_points, norm="ortho")


def dynamic_burgers_context(
    source: torch.Tensor,
    prediction: torch.Tensor,
    config: Config,
    coarse_solver: BurgersETDRK4 | None = None,
    coarse_prediction_full: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build a label-free Crank-Nicolson Burgers defect and soft spectrum."""

    source_gradient, source_laplacian = spectral_derivatives(source)
    prediction_gradient, prediction_laplacian = spectral_derivatives(prediction)
    flux_gradient = 0.5 * (
        source * source_gradient + prediction * prediction_gradient
    )
    diffusion = 0.5 * config.viscosity * (
        source_laplacian + prediction_laplacian
    )
    defect = prediction - source + config.final_time * (flux_gradient - diffusion)
    defect = defect - defect.mean(-1, keepdim=True)

    n_grid = source.shape[-1]
    wave = 2.0 * math.pi * torch.fft.rfftfreq(
        n_grid, d=1.0 / n_grid, device=source.device
    )
    defect_spectrum = torch.fft.rfft(defect, norm="ortho")
    mean_velocity = 0.5 * (source.mean(-1) + prediction.mean(-1))
    jacobian = (
        1.0
        + 0.5 * config.viscosity * config.final_time * wave[None, :].square()
        + 0.5j * config.final_time * mean_velocity[:, None] * wave[None, :]
    )
    shift = config.defect_shift * (
        1.0 + config.viscosity * config.final_time * wave[None, :].square()
    )
    inverse = jacobian.conj() / (jacobian.abs().square() + shift.square())
    preconditioned_spectrum = -inverse * defect_spectrum
    preconditioned_spectrum[:, 0] = 0.0
    preconditioned = torch.fft.irfft(
        preconditioned_spectrum, n=n_grid, norm="ortho"
    )

    if coarse_prediction_full is not None:
        coarse_prediction = coarse_prediction_full
        coarse_proxy = coarse_prediction - prediction
        coarse_proxy = coarse_proxy - coarse_proxy.mean(-1, keepdim=True)
        coarse_proxy_spectrum = torch.fft.rfft(coarse_proxy, norm="ortho")
        coarse_proxy_spectrum[:, 0] = 0.0
    elif coarse_solver is None:
        coarse_prediction = prediction
        coarse_proxy = preconditioned
        coarse_proxy_spectrum = preconditioned_spectrum
    else:
        coarse_source = spectral_resize(source, config.coarse_resolution)
        coarse_prediction = coarse_solver.solve(coarse_source)[:, -1]
        coarse_prediction = spectral_resize(coarse_prediction, n_grid)
        coarse_proxy = coarse_prediction - prediction
        coarse_proxy = coarse_proxy - coarse_proxy.mean(-1, keepdim=True)
        coarse_proxy_spectrum = torch.fft.rfft(coarse_proxy, norm="ortho")
        coarse_proxy_spectrum[:, 0] = 0.0

    amplitude = coarse_proxy_spectrum.abs()
    threshold_floor = config.gate_floor_ratio * amplitude.amax(-1, keepdim=True)
    threshold_noise = config.gate_noise_multiplier * amplitude.median(
        -1, keepdim=True
    ).values
    threshold = torch.maximum(threshold_floor, threshold_noise)
    temperature = (
        config.gate_temperature_ratio * threshold
    ).clamp_min(1.0e-12)
    gate = torch.sigmoid((amplitude - threshold) / temperature)
    gate = config.gate_soft_floor + (1.0 - config.gate_soft_floor) * gate
    strongest = amplitude.argmax(-1, keepdim=True)
    gate.scatter_(1, strongest, 1.0)
    gate[:, 0] = 0.0
    gate = gate.detach()

    selected_defect_spectrum = defect_spectrum * gate
    selected_coarse_proxy_spectrum = coarse_proxy_spectrum * gate
    selected_defect = torch.fft.irfft(
        selected_defect_spectrum, n=n_grid, norm="ortho"
    )
    selected_coarse_proxy = torch.fft.irfft(
        selected_coarse_proxy_spectrum, n=n_grid, norm="ortho"
    )
    dynamic_features = torch.stack(
        (
            defect / sample_rms(defect),
            preconditioned / sample_rms(preconditioned),
            coarse_proxy / sample_rms(coarse_proxy),
            selected_coarse_proxy / sample_rms(coarse_proxy),
        ),
        dim=1,
    )
    return dynamic_features, selected_coarse_proxy_spectrum, {
        "defect": defect,
        "preconditioned": preconditioned,
        "selected_defect": selected_defect,
        "coarse_prediction": coarse_prediction,
        "coarse_proxy": coarse_proxy,
        "selected_coarse_proxy": selected_coarse_proxy,
        "gate": gate,
        "threshold": threshold,
    }


class PhysicsAnchoredGrade1(torch.nn.Module):
    """Predict only the residual left by the fixed coarse Burgers solver."""

    def __init__(self, modes: int, width: int, depth: int) -> None:
        super().__init__()
        self.lifting = torch.nn.Linear(4, width)
        self.spectral_layers = torch.nn.ModuleList(
            base.SpectralConv1d(width, width, modes) for _ in range(depth)
        )
        self.pointwise_layers = torch.nn.ModuleList(
            torch.nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.projection = base.Projection1d(width)
        self.depth = int(depth)
        self.register_buffer("deployment_gain", torch.ones(()))
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.projection.fc2.weight, gain=1.0e-2)
            self.projection.fc2.bias.zero_()

    def forward(
        self, x: torch.Tensor, coarse_prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = x[:, :, 0]
        inputs = torch.cat(
            (
                x,
                coarse_prediction.unsqueeze(-1),
                (coarse_prediction - source).unsqueeze(-1),
            ),
            dim=-1,
        )
        feature = self.lifting(inputs).permute(0, 2, 1)
        for index, (spectral, pointwise) in enumerate(
            zip(self.spectral_layers, self.pointwise_layers, strict=True)
        ):
            feature = spectral(feature) + pointwise(feature)
            if index < self.depth - 1:
                feature = F.gelu(feature)
        correction = self.projection(feature)
        correction = correction - correction.mean(-1, keepdim=True)
        correction = self.deployment_gain * correction
        return correction, feature


class PhysicsGrade0(torch.nn.Module):
    """Fixed zero-parameter low-fidelity operator used as Grade 0."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fine_resolution = int(config.resolution)
        self.coarse_resolution = int(config.coarse_resolution)
        self.solver = BurgersETDRK4(
            TrajectorySpec(
                resolution=config.coarse_resolution,
                viscosity=config.viscosity,
                domain_length=2.0 * math.pi,
                final_time=config.final_time,
                record_steps=1,
                solver_substeps_per_record=config.coarse_solver_steps,
            ),
            torch.device(config.device),
        )
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        coarse_source = spectral_resize(source, self.coarse_resolution)
        prediction = self.solver.solve(coarse_source)[:, -1]
        return spectral_resize(prediction, self.fine_resolution)


class DynamicMGNO(base.AdaptiveMultigradeOperator):
    """Frozen Grades conditioned on inference-available Burgers defects."""

    def __init__(self, config: Config) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        width = config.widths[0]
        self.grade1 = PhysicsAnchoredGrade1(
            config.modes, width, config.depths[0]
        )
        self.residual_grades = torch.nn.ModuleList(
            base.AdaptiveResidualGrade(capacity, width, depth, extra_channels=4)
            for capacity, depth in zip(
                config.residual_basis_modes, config.depths[1:], strict=True
            )
        )
        self.grade0 = PhysicsGrade0(config)

    @property
    def coarse_solver(self) -> BurgersETDRK4:
        """Preserve the solver interface used by earlier experiment code."""

        return self.grade0.solver

    def forward_grade0(self, source: torch.Tensor) -> torch.Tensor:
        """Return the permanently frozen Grade 0 approximation."""

        return self.grade0(source)

    def coarse_prediction(self, source: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for the explicit Grade 0 operator."""

        return self.forward_grade0(source)

    def forward_up_to(
        self, x: torch.Tensor, active_grades: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        source = x[:, :, 0]
        shared_coarse_prediction = self.forward_grade0(source)
        correction, feature = self.grade1(x, shared_coarse_prediction)
        cumulative = shared_coarse_prediction + correction
        if float(self.grade1.deployment_gain) == 0.0:
            feature = torch.zeros_like(feature)
        corrections: list[torch.Tensor] = []
        for grade_index in range(1, active_grades):
            correction, feature, _ = self.apply_residual_grade(
                x,
                cumulative,
                feature,
                grade_index,
                shared_coarse_prediction,
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
        coarse_prediction_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = x[:, :, 0]
        if coarse_prediction_full is None:
            coarse_prediction_full = self.forward_grade0(source)
        features, spectrum, _ = dynamic_burgers_context(
            source,
            cumulative,
            self.config,
            coarse_prediction_full=coarse_prediction_full,
        )
        grade = self.residual_grades[grade_index - 1]
        correction, feature, predicted_spectrum = grade(
            x,
            cumulative,
            previous_feature,
            dynamic_features=features,
            dynamic_spectrum=spectrum,
        )
        correction = correction - correction.mean(-1, keepdim=True)
        if float(grade.deployment_gain) == 0.0:
            return torch.zeros_like(correction), previous_feature, predicted_spectrum
        return correction, feature, predicted_spectrum


@torch.no_grad()
def evaluate_coarse_solver(
    model: DynamicMGNO,
    x: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    values: list[torch.Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        source = x[start : start + batch_size, :, 0].to(device)
        source = spectral_resize(source, model.config.coarse_resolution)
        prediction = model.coarse_solver.solve(source)[:, -1]
        prediction = spectral_resize(prediction, target.shape[-1]).cpu()
        values.append(
            base.sample_relative_l2(
                prediction, target[start : start + batch_size]
            )
        )
    return float(torch.cat(values).mean())


def train_physics_anchored_grade1(
    model: DynamicMGNO,
    loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
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
    for epoch in range(1, config.epochs[0] + 1):
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        objective_sum = relative_sum = mse_sum = 0.0
        samples = 0
        model.grade1.train()
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            with torch.no_grad():
                coarse_prediction = model.forward_grade0(x[:, :, 0])
            optimizer.zero_grad(set_to_none=True)
            correction, _ = model.grade1(x, coarse_prediction)
            prediction = coarse_prediction + correction
            relative = base.sample_relative_l2(prediction, target).mean()
            target_correction = target - coarse_prediction
            h1 = base.relative_h1(
                correction,
                target_correction,
                truth=target,
                truth_floor_ratio=config.h1_truth_floor_ratio,
            )
            objective = relative + config.h1_weights[0] * h1
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.grade1.parameters(), 1.0)
            optimizer.step()
            batch_size = x.shape[0]
            objective_sum += float(objective.detach()) * batch_size
            relative_sum += float(relative.detach()) * batch_size
            mse_sum += float(F.mse_loss(prediction, target).detach()) * batch_size
            samples += batch_size
        scheduler.step()
        row = {
            "grade_name": "grade1",
            "grade": 1,
            "epoch": epoch,
            "global_epoch": epoch,
            "learning_rate": learning_rate,
            "objective": objective_sum / samples,
            "train_mse": mse_sum / samples,
            "train_relative_l2": relative_sum / samples,
            "elapsed_seconds": time.perf_counter() - start,
            "test_relative_l2": "",
        }
        history.append(row)
        base.save_checkpoint(
            model,
            config.output_dir / "_epoch_checkpoints" / "multigrade",
            "multigrade",
            epoch,
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
    calibration = calibrate_grade_on_validation(
        model, validation_loader, 0, config, device
    )
    base.save_checkpoint(
        model,
        config.output_dir / "_epoch_checkpoints" / "multigrade",
        "multigrade",
        config.epochs[0],
        1,
    )
    model.grade1.eval()
    model.grade1.requires_grad_(False)
    return history, calibration


@torch.no_grad()
def calibrate_grade_on_validation(
    model: DynamicMGNO,
    loader: torch.utils.data.DataLoader,
    grade_index: int,
    config: Config,
    device: torch.device,
) -> dict[str, object]:
    """Choose a correction scale using validation data without reading test data."""

    if grade_index == 0:
        gain = model.grade1.deployment_gain
    else:
        gain = model.residual_grades[grade_index - 1].deployment_gain
    gain.fill_(1.0)
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    totals = [0.0 for _ in scales]
    samples = 0
    model.eval()
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        if grade_index == 0:
            coarse = model.forward_grade0(x[:, :, 0])
            correction, _ = model.grade1(x, coarse)
            predecessor = coarse
        else:
            predecessor, previous_feature = model.predecessor_state(x, grade_index)
            correction, _, _ = model.apply_residual_grade(
                x, predecessor, previous_feature, grade_index
            )
        for index, scale in enumerate(scales):
            totals[index] += float(
                base.sample_relative_l2(predecessor + scale * correction, target).sum()
            )
        samples += x.shape[0]
    scores = [value / samples for value in totals]
    best_index = min(range(len(scales)), key=scores.__getitem__)
    baseline = scores[0]
    relative_gain = (baseline - scores[best_index]) / max(baseline, 1.0e-12)
    accepted = best_index > 0 and relative_gain >= config.minimum_validation_gain
    selected_scale = scales[best_index] if accepted else 0.0
    gain.fill_(selected_scale)
    decision = "ACCEPT" if accepted else "REJECT"
    print(
        f"{decision} Grade {grade_index + 1}: validation_l2 "
        f"{baseline:.6e} -> {scores[best_index]:.6e}, "
        f"gain={100.0 * relative_gain:.3f}%, scale={selected_scale:.2f}"
    )
    return {
        "grade": grade_index + 1,
        "accepted": accepted,
        "validation_relative_l2_before": baseline,
        "validation_relative_l2_after": scores[best_index] if accepted else baseline,
        "validation_gain": relative_gain if accepted else 0.0,
        "deployment_gain": selected_scale,
        "candidate_scales": list(scales),
        "candidate_validation_relative_l2": scores,
    }


def select_parameter_matched_fno4_width(
    target_parameters: int,
    modes: int,
) -> tuple[int, int]:
    """Find the narrowest FNO4 whose parameter count reaches the MGNO budget."""

    for width in range(1, 1025):
        model = base.FNO4Baseline1d(modes, width)
        parameters = base.parameter_count(model)
        if parameters >= target_parameters:
            return width, parameters
    raise RuntimeError("Unable to match the MGNO parameter budget with FNO4")


def train_parameter_matched_fno4(
    train_data: tuple[torch.Tensor, torch.Tensor],
    config: Config,
    device: torch.device,
    target_parameters: int,
    checkpoint_dir: Path | None = None,
) -> tuple[torch.nn.Module, list[dict[str, object]], int]:
    """Train an exact four-layer FNO with a parameter budget matched to MGNO."""

    width, matched_parameters = select_parameter_matched_fno4_width(
        target_parameters, config.modes
    )
    base.set_seed(config.seed)
    model = base.FNO4Baseline1d(config.modes, width).to(device)
    loader = base.make_loader(
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
    print(
        f"Parameter-matched FNO4: width={width}, modes={config.modes}, "
        f"parameters={matched_parameters}, target={target_parameters}"
    )
    for epoch in range(1, config.baseline_epochs + 1):
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        relative_sum = mse_sum = 0.0
        samples = 0
        model.train()
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            relative = base.sample_relative_l2(prediction, target).mean()
            relative.backward()
            optimizer.step()
            batch = x.shape[0]
            relative_sum += float(relative.detach()) * batch
            mse_sum += float(F.mse_loss(prediction, target).detach()) * batch
            samples += batch
        scheduler.step()
        row = {
            "grade_name": "fno4_parameter_matched",
            "grade": 0,
            "epoch": epoch,
            "global_epoch": epoch,
            "learning_rate": learning_rate,
            "objective": relative_sum / samples,
            "train_mse": mse_sum / samples,
            "train_relative_l2": relative_sum / samples,
            "elapsed_seconds": time.perf_counter() - start,
            "test_relative_l2": "",
        }
        history.append(row)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )
        print(
            f"FNO4-PM epoch={epoch:04d} "
            f"epoch_time={row['elapsed_seconds']:.6f}s "
            f"lr={row['learning_rate']:.6e} "
            f"train_mse={row['train_mse']:.6e} "
            f"train_l2={row['train_relative_l2']:.6e}"
        )
    model.eval()
    model.requires_grad_(False)
    return model, history, width


def run(config: Config) -> dict[str, object]:
    config.validate()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory is not empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(config.seed)
    device = torch.device(config.device)
    x_train, y_train = base.load_burgers_train_data(config)
    x_validation, y_validation = load_validation_data(config)
    train_loader = base.make_loader(
        x_train,
        y_train,
        config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = base.make_loader(
        x_validation, y_validation, config.batch_size, shuffle=False
    )
    model = DynamicMGNO(config).to(device)
    histories, grade1_calibration = train_physics_anchored_grade1(
        model, train_loader, validation_loader, config, device
    )
    profiles: list[dict[str, object]] = [grade1_calibration]
    used_frequencies = set(range(config.modes))
    global_offset = config.epochs[0]
    for grade_index in range(1, config.grade_count):
        grade_history, profile = base.train_residual_grade(
            model,
            train_loader,
            config,
            device,
            grade_index,
            global_offset,
            used_frequencies,
        )
        histories.extend(grade_history)
        calibration = calibrate_grade_on_validation(
            model, validation_loader, grade_index, config, device
        )
        profile["validation_calibration"] = calibration
        profiles.append(profile)
        base.save_checkpoint(
            model,
            config.output_dir / "_epoch_checkpoints" / "multigrade",
            "multigrade",
            global_offset + config.epochs[grade_index],
            grade_index + 1,
        )
        if calibration["accepted"]:
            used_frequencies.update(
                int(value) for value in profile["selected_indices"]
            )
        global_offset += config.epochs[grade_index]

    baseline, baseline_history = base.train_baseline(
        (x_train, y_train), config, device
    )
    histories.extend(baseline_history)
    matched_fno4, matched_fno4_history, matched_fno4_width = (
        train_parameter_matched_fno4(
            (x_train, y_train),
            config,
            device,
            base.parameter_count(model),
        )
    )
    print(
        "[Final test] Dynamic MGNO, paper FNO4, and parameter-matched "
        "FNO4 weights are frozen"
    )
    x_test, y_test = base.load_burgers_test_data(config)
    test_loader = base.make_loader(
        x_test, y_test, config.batch_size, shuffle=False
    )
    base.post_training_test_audit(
        model, baseline, histories, test_loader, config, device
    )
    mgno_metrics = base.evaluate_multigrade(
        model, test_loader, device, config.grade_count
    )
    fno_metrics = base.evaluate_baseline(baseline, test_loader, device)
    matched_fno4_metrics = base.evaluate_baseline(
        matched_fno4, test_loader, device
    )
    coarse_solver_rse = evaluate_coarse_solver(
        model, x_test, y_test, config.batch_size, device
    )
    figures = base.save_reports(histories, mgno_metrics, fno_metrics, config)
    result = {
        "config": {
            **asdict(config),
            "data": str(config.data.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "protocol": {
            "model": "Dynamic defect-preconditioned MGNO",
            "grade_rule": "Every later Grade learns the residual left by all frozen predecessors",
            "dynamic_rule": "Every sample receives an inference-available Crank-Nicolson Burgers defect and a soft spectral gate",
            "preconditioner_rule": "A fixed-cost coarse ETDRK4 Burgers solve provides an inference-available dynamic correction proxy",
            "label_rule": "Truth residuals configure training targets and the global Grade basis only; they never enter inference features",
            "validation_rule": "A disjoint validation split calibrates each frozen Grade; rejected Grades contribute zero correction and no downstream feature",
            "parameter_matching_rule": "The matched FNO keeps the exact four-layer paper architecture and operator modes, while width is chosen so its trainable parameter count is minimally above MGNO",
            "test_rule": "The final 200 samples are first accessed after all optimization",
        },
        "profiles": profiles,
        "metrics": {
            "dynamic_mgno_test_relative_l2": mgno_metrics["relative_l2"],
            "fno4_test_relative_l2": fno_metrics["relative_l2"],
            "parameter_matched_fno4_test_relative_l2": matched_fno4_metrics[
                "relative_l2"
            ],
            "fno4_over_dynamic_mgno": float(fno_metrics["relative_l2"])
            / float(mgno_metrics["relative_l2"]),
            "coarse_solver_test_relative_l2": coarse_solver_rse,
            "parameter_matched_fno4_over_dynamic_mgno": float(
                matched_fno4_metrics["relative_l2"]
            )
            / float(mgno_metrics["relative_l2"]),
            "dynamic_mgno_parameters": base.parameter_count(model),
            "fno4_parameters": base.parameter_count(baseline),
            "parameter_matched_fno4_parameters": base.parameter_count(
                matched_fno4
            ),
            "parameter_matched_fno4_width": matched_fno4_width,
        },
        "figures": [str(path.resolve()) for path in figures],
    }
    with (config.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    torch.save(
        {"config": result["config"], "model_state_dict": model.state_dict()},
        config.output_dir / "dynamic_mgno.pt",
    )
    torch.save(baseline.state_dict(), config.output_dir / "fno4_checkpoint.pt")
    torch.save(
        matched_fno4.state_dict(),
        config.output_dir / "fno4_parameter_matched_checkpoint.pt",
    )
    with (config.output_dir / "fno4_parameter_matched_training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(matched_fno4_history[0])
        )
        writer.writeheader()
        writer.writerows(matched_fno4_history)
    if not config.keep_epoch_checkpoints:
        checkpoint_dir = config.output_dir / "_epoch_checkpoints"
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)
    print("\nFINAL DYNAMIC MGNO RESULT")
    print(f"  Grades: {config.grade_count}")
    print(f"  Dynamic MGNO test relative L2: {float(mgno_metrics['relative_l2']):.6e}")
    print(f"  FNO4 test relative L2: {float(fno_metrics['relative_l2']):.6e}")
    print(
        "  Parameter-matched FNO4 test relative L2: "
        f"{float(matched_fno4_metrics['relative_l2']):.6e}"
    )
    print(f"  Coarse solver test relative L2: {coarse_solver_rse:.6e}")
    print(f"  FNO4/Dynamic MGNO gain: {result['metrics']['fno4_over_dynamic_mgno']:.3f}x")
    print(
        "  Parameter-matched FNO4/Dynamic MGNO gain: "
        f"{result['metrics']['parameter_matched_fno4_over_dynamic_mgno']:.3f}x"
    )
    print(f"  Output: {config.output_dir.resolve()}")
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Dynamic defect-preconditioned MGNO for paper Burgers data"
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
    parser.add_argument("--learning-rates", nargs="+", type=float, default=(0.001, 0.001))
    parser.add_argument("--grade-scheduler-steps", nargs="+", type=int, default=(100, 100))
    parser.add_argument("--grade-scheduler-gammas", nargs="+", type=float, default=(0.5, 0.7))
    parser.add_argument("--baseline-epochs", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--basis-min-modes", type=int, default=4)
    parser.add_argument("--basis-max-modes", type=int, default=16)
    parser.add_argument("--residual-basis-modes", nargs="+", type=int, default=(16,))
    parser.add_argument("--basis-energy", type=float, default=0.997)
    parser.add_argument("--relative-error-score-power", type=float, default=0.25)
    parser.add_argument("--coherence-score-power", type=float, default=0.5)
    parser.add_argument("--consensus-folds", type=int, default=5)
    parser.add_argument("--consensus-variation-penalty", type=float, default=1.0)
    parser.add_argument("--new-frequency-fraction", type=float, default=0.625)
    parser.add_argument("--residual-energy-floor-fraction", type=float, default=0.001)
    parser.add_argument("--minimum-new-modes", type=int, default=2)
    parser.add_argument("--soft-frequency-floor", type=float, default=0.05)
    parser.add_argument("--residual-mse-weight", type=float, default=0.25)
    parser.add_argument("--spectral-weights", nargs="+", type=float, default=(0.0, 0.05))
    parser.add_argument("--h1-weights", nargs="+", type=float, default=(0.0, 0.02))
    parser.add_argument("--predictor-weights", nargs="+", type=float, default=(0.0, 0.1))
    parser.add_argument("--viscosity", type=float, default=0.1)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--defect-shift", type=float, default=0.001)
    parser.add_argument("--gate-floor-ratio", type=float, default=0.01)
    parser.add_argument("--gate-noise-multiplier", type=float, default=3.0)
    parser.add_argument("--gate-soft-floor", type=float, default=0.05)
    parser.add_argument("--gate-temperature-ratio", type=float, default=0.15)
    parser.add_argument("--coarse-resolution", type=int, default=128)
    parser.add_argument("--coarse-solver-steps", type=int, default=16)
    parser.add_argument("--validation-start", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0)
    parser.add_argument("--h1-truth-floor-ratio", type=float, default=1.0e-6)
    parser.add_argument("--disable-post-training-test-audit", action="store_true")
    parser.add_argument("--disable-frozen-feature-cache", action="store_true")
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
        widths=tuple(args.widths),
        depths=tuple(args.depths),
        epochs=tuple(args.epochs),
        learning_rates=tuple(args.learning_rates),
        grade_scheduler_steps=tuple(args.grade_scheduler_steps),
        grade_scheduler_gammas=tuple(args.grade_scheduler_gammas),
        baseline_epochs=args.baseline_epochs,
        baseline_depth=4,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        basis_min_modes=args.basis_min_modes,
        basis_max_modes=args.basis_max_modes,
        residual_basis_modes=tuple(args.residual_basis_modes),
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
        run_fno4_baseline=True,
        post_training_test_audit=not args.disable_post_training_test_audit,
        cache_frozen_features=not args.disable_frozen_feature_cache,
        overwrite=args.overwrite,
        viscosity=args.viscosity,
        final_time=args.final_time,
        defect_shift=args.defect_shift,
        gate_floor_ratio=args.gate_floor_ratio,
        gate_noise_multiplier=args.gate_noise_multiplier,
        gate_soft_floor=args.gate_soft_floor,
        gate_temperature_ratio=args.gate_temperature_ratio,
        coarse_resolution=args.coarse_resolution,
        coarse_solver_steps=args.coarse_solver_steps,
        validation_start=args.validation_start,
        validation_size=args.validation_size,
        minimum_validation_gain=args.minimum_validation_gain,
    )


if __name__ == "__main__":
    run(parse_args())
