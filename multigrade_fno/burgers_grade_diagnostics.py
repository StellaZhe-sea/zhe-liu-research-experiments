"""Per-Grade scientific diagnostics for Dynamic Burgers MGNO variants."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import torch

import burgers_dynamic_mgno as dynamic

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRADE_COLORS = ("#4C78A8", "#2A9D8F", "#E9A23B", "#D95D8A")
BACKGROUND = "#F4F1EA"
AXIS_BACKGROUND = "#FFFDF8"
TEXT = "#293241"
MUTED = "#7C8798"
GRID = "#D8D2C4"


def make_svg_responsive(path: Path) -> None:
    """Make a standalone SVG fill browser width without stretching."""

    source = path.read_text(encoding="utf-8")
    source = re.sub(
        r'(<svg\s+[^>]*?)width="[^"]+"\s+height="[^"]+"',
        r'\1width="100%" height="auto" style="display:block;width:100%;height:auto"',
        source,
        count=1,
    )
    path.write_text(source, encoding="utf-8")


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        (prediction - target).flatten().norm()
        / target.flatten().norm().clamp_min(1.0e-12)
    )


def _components_from_spectrum(
    spectrum: torch.Tensor,
    indices: torch.Tensor,
    n_grid: int,
    maximum: int = 8,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for index in indices[:maximum]:
        isolated = torch.zeros_like(spectrum)
        isolated[..., index] = spectrum[..., index]
        component = torch.fft.irfft(isolated, n=n_grid, norm="ortho")
        values.append(component[0].detach().cpu().numpy())
    if not values:
        return np.zeros((0, n_grid), dtype=np.float32)
    return np.stack(values)


def _curve_values(
    history: list[dict[str, object]], grade: int, key: str
) -> np.ndarray:
    aliases = {
        "objective": ("objective",),
        "relative": ("train_relative_l2", "training_relative_l2"),
        "mse": ("train_mse", "training_mse"),
    }
    rows = [row for row in history if int(row["grade"]) == grade]
    for candidate in aliases[key]:
        if rows and candidate in rows[0]:
            return np.asarray([float(row[candidate]) for row in rows])
    return np.full(len(rows), np.nan)


def _save_grade0_diagnostic(
    model: torch.nn.Module,
    source: torch.Tensor,
    truth: torch.Tensor,
    coordinate: np.ndarray,
    output_dir: Path,
    variant_name: str,
) -> Path:
    """Visualize the fixed physics operator without implying neural training."""

    coarse = model.forward_grade0(source)
    residual = truth - coarse
    source_spectrum = torch.fft.rfft(source, norm="ortho").abs()[0].cpu().numpy()
    coarse_spectrum = torch.fft.rfft(coarse, norm="ortho").abs()[0].cpu().numpy()
    truth_spectrum = torch.fft.rfft(truth, norm="ortho").abs()[0].cpu().numpy()
    residual_spectrum = torch.fft.rfft(residual, norm="ortho").abs()[0].cpu().numpy()
    rse = _relative_l2(coarse, truth)
    mae = float(residual.abs().mean())
    mass_drift = float((coarse.mean(-1) - source.mean(-1)).abs().mean())
    closure_parameters = sum(parameter.numel() for parameter in model.grade0.parameters())
    learned = closure_parameters > 0
    operator_label = (
        "trained physics closure (frozen)" if learned else "fixed physics operator"
    )
    style = {
        "font.family": "DejaVu Sans",
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": AXIS_BACKGROUND,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.32,
        "legend.facecolor": AXIS_BACKGROUND,
        "legend.edgecolor": GRID,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        color = GRADE_COLORS[0]
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(16, 13),
            constrained_layout=True,
            facecolor=BACKGROUND,
        )
        figure.suptitle(
            f"[{variant_name}] Grade 0 | {operator_label} | "
            f"closure parameters={closure_parameters} | RSE={rse:.3e}",
            fontsize=15,
            fontweight="bold",
        )

        axis = axes[0, 0]
        axis.plot(coordinate, source[0].cpu(), color="#7C8DA6", lw=1.2, label="Initial condition")
        axis.plot(coordinate, truth[0].cpu(), color="#293241", lw=1.8, label="Truth")
        axis.plot(coordinate, coarse[0].cpu(), color=color, lw=1.5, label="Grade 0 output")
        axis.set_title("Physical-space input and output")
        axis.set_xlabel("x")
        axis.legend()
        axis.grid(True)

        axis = axes[0, 1]
        frequency = np.arange(truth_spectrum.size)
        axis.semilogy(frequency, source_spectrum + 1.0e-30, color="#7C8DA6", lw=1.0, label="Initial spectrum")
        axis.semilogy(frequency, truth_spectrum + 1.0e-30, color="#293241", lw=1.4, label="Truth spectrum")
        axis.semilogy(frequency, coarse_spectrum + 1.0e-30, color=color, lw=1.2, label="Grade 0 spectrum")
        axis.set_title("Grade 0 spectral approximation")
        axis.set_xlabel("RFFT mode index")
        axis.legend()
        axis.grid(True)

        axis = axes[1, 0]
        axis.plot(coordinate, residual[0].cpu(), color="#D95D39", lw=1.4)
        axis.axhline(0.0, color=MUTED, lw=0.8)
        axis.set_title("Residual passed to Grade 1")
        axis.set_xlabel("x")
        axis.set_ylabel("Truth - Grade 0")
        axis.grid(True)

        axis = axes[1, 1]
        axis.semilogy(frequency, residual_spectrum + 1.0e-30, color="#D95D39", lw=1.3)
        axis.set_title("Grade 0 residual spectrum")
        axis.set_xlabel("RFFT mode index")
        axis.set_ylabel("Amplitude")
        axis.grid(True)

        axis = axes[2, 0]
        labels = ("Relative L2", "MAE", "Mass drift")
        values = (rse, mae, max(mass_drift, 1.0e-16))
        bars = axis.bar(labels, values, color=(color, "#2A9D8F", "#E9A23B"))
        axis.set_yscale("log")
        axis.set_title("Grade 0 accuracy and conservation")
        axis.grid(axis="y")
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.2e}", ha="center", fontsize=9)

        axis = axes[2, 1]
        axis.axis("off")
        config = model.config
        description = (
            "Input initial condition\n"
            f"    Fine grid: {config.resolution}\n"
            "              ↓ spectral restriction\n"
            f"    Coarse grid: {config.coarse_resolution}\n"
            f"              ↓ ETDRK4 ({config.coarse_solver_steps} steps)\n"
            "    Low-fidelity Burgers solution\n"
            "              ↓ spectral prolongation\n"
            f"    Grade 0 output: {config.resolution} points\n\n"
            + (
                "Learned on training data · permanently frozen\n"
                f"Spectral-closure parameters: {closure_parameters}"
                if learned
                else "Deterministic · label-free · permanently frozen\nZero trainable parameters"
            )
        )
        axis.text(
            0.5,
            0.5,
            description,
            ha="center",
            va="center",
            fontsize=13,
            linespacing=1.45,
            bbox={"boxstyle": "round,pad=1.0", "facecolor": "#F4F1EA", "edgecolor": color},
        )

        path = output_dir / "grade_00_diagnostic.svg"
        figure.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)
        make_svg_responsive(path)
        return path


def save_dynamic_grade_diagnostics(
    model: torch.nn.Module,
    config: dynamic.Config,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    history: list[dict[str, object]],
    output_dir: Path,
    variant_name: str,
    context_function,
    coarse_anchored: bool,
) -> list[Path]:
    """Save a Helmholtz-style 6x2 SVG for every frozen Dynamic Grade."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    x = x_validation[:1].to(device)
    truth = y_validation[:1].to(device)
    source = x[:, :, 0]
    coordinate = x[0, :, 1].detach().cpu().numpy()
    n_grid = source.shape[-1]
    paths: list[Path] = []
    model.eval()

    with torch.no_grad():
        if coarse_anchored:
            paths.append(
                _save_grade0_diagnostic(
                    model,
                    source,
                    truth,
                    coordinate,
                    output_dir,
                    variant_name,
                )
            )
        for grade in range(1, config.grade_count + 1):
            if grade == 1:
                if coarse_anchored:
                    before = model.forward_grade0(source)
                    correction, _ = model.grade1(x, before)
                    after = before + correction
                else:
                    before = torch.zeros_like(truth)
                    after, _, _ = model.forward_up_to(x, 1)
                    correction = after
                target_correction = truth - before
                target_spectrum = torch.fft.rfft(
                    target_correction, norm="ortho"
                )
                count = min(config.modes, target_spectrum.shape[-1])
                selected = target_spectrum.abs()[0].topk(count).indices.sort().values
                frequency_weights = torch.ones(
                    selected.numel(), device=device
                )
                predicted_spectrum = torch.fft.rfft(correction, norm="ortho")
            else:
                before, previous_feature = model.predecessor_state(x, grade - 1)
                correction, _, predicted_spectrum = model.apply_residual_grade(
                    x, before, previous_feature, grade - 1
                )
                after = before + correction
                target_correction = truth - before
                grade_module = model.residual_grades[grade - 2]
                count = int(grade_module.active_count)
                selected = grade_module.frequency_indices[:count]
                frequency_weights = grade_module.frequency_weights[:count]
                target_spectrum = torch.fft.rfft(
                    target_correction, norm="ortho"
                )

            dynamic_features, dynamic_spectrum = context_function(source, before)
            correction_spectrum = torch.fft.rfft(correction, norm="ortho")
            remaining_spectrum = torch.fft.rfft(
                truth - after, norm="ortho"
            )
            components = _components_from_spectrum(
                predicted_spectrum, selected, n_grid
            )
            before_rse = _relative_l2(before, truth)
            after_rse = _relative_l2(after, truth)
            captured = float(
                target_spectrum[:, selected].abs().square().sum()
                / target_spectrum.abs().square().sum().clamp_min(1.0e-20)
            )

            with plt.rc_context(
                {
                    "font.family": "DejaVu Sans",
                    "figure.facecolor": BACKGROUND,
                    "axes.facecolor": AXIS_BACKGROUND,
                    "axes.edgecolor": MUTED,
                    "axes.labelcolor": TEXT,
                    "axes.titlecolor": TEXT,
                    "xtick.color": MUTED,
                    "ytick.color": MUTED,
                    "text.color": TEXT,
                    "grid.color": GRID,
                    "grid.alpha": 0.32,
                    "legend.facecolor": AXIS_BACKGROUND,
                    "legend.edgecolor": GRID,
                    "legend.framealpha": 0.8,
                    "svg.fonttype": "none",
                }
            ):
                color = GRADE_COLORS[(grade - 1) % len(GRADE_COLORS)]
                figure = plt.figure(figsize=(16, 24), facecolor=BACKGROUND)
                figure.suptitle(
                    f"[{variant_name}]  Grade {grade}  |  "
                    f"selected={count}  |  captured={100.0 * captured:.2f}%  |  "
                    f"RSE {before_rse:.2e} -> {after_rse:.2e}",
                    fontsize=14,
                    fontweight="bold",
                    y=0.992,
                )
                grid = gridspec.GridSpec(
                    6,
                    2,
                    figure=figure,
                    top=0.965,
                    bottom=0.035,
                    left=0.065,
                    right=0.985,
                    hspace=0.55,
                    wspace=0.30,
                )
                frequency = np.arange(dynamic_spectrum.shape[-1])

                axis = figure.add_subplot(grid[0, :])
                amplitude = dynamic_spectrum.abs()[0].cpu().numpy()
                axis.semilogy(
                    frequency,
                    amplitude + 1.0e-30,
                    color=MUTED,
                    linewidth=1.0,
                    label="Inference-available dynamic spectrum",
                )
                valid_selected = selected[selected < dynamic_spectrum.shape[-1]]
                axis.scatter(
                    valid_selected.cpu(),
                    amplitude[valid_selected.cpu()] + 1.0e-30,
                    color=color,
                    s=28,
                    zorder=5,
                    label="Selected residual modes",
                )
                axis.set_title("Dynamic spectral input available at inference")
                axis.set_xlabel("RFFT mode index")
                axis.set_ylabel("Amplitude")
                axis.grid(True)
                axis.legend(fontsize=8)

                axis = figure.add_subplot(grid[1, 0])
                target_amplitude = target_spectrum.abs()[0].cpu().numpy()
                axis.semilogy(
                    np.arange(target_amplitude.size),
                    target_amplitude + 1.0e-30,
                    color=MUTED,
                    linewidth=1.0,
                )
                axis.scatter(
                    selected.cpu(),
                    target_amplitude[selected.cpu()] + 1.0e-30,
                    color=color,
                    s=24,
                )
                axis.set_title("Current Grade target residual spectrum")
                axis.set_xlabel("RFFT mode index")
                axis.grid(True)

                axis = figure.add_subplot(grid[1, 1])
                axis.bar(
                    selected.cpu(),
                    frequency_weights.cpu(),
                    color=color,
                    width=0.8,
                    alpha=0.88,
                )
                axis.set_title("Adaptive residual-basis weights")
                axis.set_xlabel("Selected RFFT mode index")
                axis.set_ylabel("Weight")
                axis.grid(True)

                axis = figure.add_subplot(grid[2, 0])
                component_colors = plt.cm.coolwarm(
                    np.linspace(0.08, 0.92, max(len(components), 1))
                )
                for index, component in enumerate(components):
                    axis.plot(
                        coordinate,
                        component,
                        color=component_colors[index],
                        linewidth=0.9,
                        label=f"k={int(selected[index])}",
                    )
                axis.set_title("Predicted residual-spectrum Fourier components")
                axis.set_xlabel("x")
                if len(components):
                    axis.legend(fontsize=6, ncol=2)
                axis.grid(True)

                axis = figure.add_subplot(grid[2, 1])
                axis.plot(
                    coordinate,
                    target_correction[0].cpu(),
                    color=MUTED,
                    linestyle="--",
                    linewidth=1.3,
                    label="Target residual",
                )
                axis.plot(
                    coordinate,
                    correction[0].cpu(),
                    color=color,
                    linewidth=1.5,
                    label="Grade correction",
                )
                axis.set_title(f"Grade {grade} correction output vs target")
                axis.set_xlabel("x")
                axis.legend(fontsize=8)
                axis.grid(True)

                axis = figure.add_subplot(grid[3, 0])
                axis.plot(coordinate, truth[0].cpu(), color="#293241", lw=2, label="Truth")
                axis.plot(coordinate, before[0].cpu(), color=MUTED, ls="--", lw=1.1, label="Before")
                axis.plot(coordinate, after[0].cpu(), color=color, lw=1.4, label="After")
                axis.set_title("Cumulative solution")
                axis.set_xlabel("x")
                axis.legend(fontsize=8)
                axis.grid(True)

                axis = figure.add_subplot(grid[3, 1])
                axis.semilogy(
                    coordinate,
                    (before - truth).abs()[0].cpu() + 1.0e-30,
                    color=MUTED,
                    linestyle="--",
                    linewidth=1.1,
                    label=f"Before RSE={before_rse:.2e}",
                )
                axis.semilogy(
                    coordinate,
                    (after - truth).abs()[0].cpu() + 1.0e-30,
                    color=color,
                    linewidth=1.4,
                    label=f"After RSE={after_rse:.2e}",
                )
                axis.set_title("Pointwise absolute error")
                axis.set_xlabel("x")
                axis.legend(fontsize=8)
                axis.grid(True)

                epoch = np.arange(1, len(_curve_values(history, grade, "relative")) + 1)
                axis = figure.add_subplot(grid[4, 0])
                for values, label, line_color, style in (
                    (_curve_values(history, grade, "objective"), "Objective", color, "-"),
                    (_curve_values(history, grade, "relative"), "Cumulative relative L2", "#56B4E9", "--"),
                    (_curve_values(history, grade, "mse"), "MSE", "#E69F00", ":"),
                ):
                    if values.size and not np.isnan(values).all():
                        axis.semilogy(epoch, values, color=line_color, linestyle=style, linewidth=1.2, marker="o", markersize=2, label=label)
                axis.set_title("Training metrics: every epoch recorded")
                axis.set_xlabel("Local epoch")
                axis.legend(fontsize=7)
                axis.grid(True)

                axis = figure.add_subplot(grid[4, 1])
                feature_energy = dynamic_features[0].square().mean(-1).cpu().numpy()
                axis.bar(np.arange(feature_energy.size), feature_energy, color=color, alpha=0.88)
                axis.set_title("Inference-available dynamic feature energy")
                axis.set_xlabel("Dynamic feature channel")
                axis.set_ylabel("Mean square energy")
                axis.grid(True)

                axis = figure.add_subplot(grid[5, :])
                axis.semilogy(np.arange(target_amplitude.size), target_amplitude + 1.0e-30, color=MUTED, linestyle="--", linewidth=1.0, label="Target residual spectrum")
                axis.semilogy(np.arange(correction_spectrum.shape[-1]), correction_spectrum.abs()[0].cpu() + 1.0e-30, color=color, linewidth=1.1, label="Grade output spectrum")
                axis.semilogy(np.arange(predicted_spectrum.shape[-1]), predicted_spectrum.abs()[0].cpu() + 1.0e-30, color="#56B4E9", linewidth=0.9, alpha=0.9, label="Predicted target spectrum")
                axis.semilogy(np.arange(remaining_spectrum.shape[-1]), remaining_spectrum.abs()[0].cpu() + 1.0e-30, color="#E69F00", linewidth=0.9, alpha=0.9, label="Remaining spectrum")
                axis.scatter(selected.cpu(), correction_spectrum.abs()[0, selected].cpu() + 1.0e-30, color=color, s=20, zorder=5)
                axis.set_title("Target, predicted target, Grade output, and remaining spectra")
                axis.set_xlabel("RFFT mode index")
                axis.set_ylabel("Amplitude")
                axis.legend(fontsize=8, ncol=2)
                axis.grid(True)

                path = output_dir / f"grade_{grade:02d}_diagnostic.svg"
                figure.savefig(
                    path,
                    format="svg",
                    bbox_inches="tight",
                    pad_inches=0.02,
                    facecolor=figure.get_facecolor(),
                )
                plt.close(figure)
                make_svg_responsive(path)
                paths.append(path)
    return paths
