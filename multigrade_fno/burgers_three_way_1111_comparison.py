"""Parameter-controlled comparison of FNO4, plain MG-FNO, and Dynamic MGNO."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib
import torch

import burgers_adaptive_multigrade_operator as base
import burgers_dynamic_mgno as dynamic
from burgers_plain_multigrade_ablation import (
    Config as PlainConfig,
    PlainMultigradeFNO1d,
    evaluate_multigrade as evaluate_plain_multigrade,
    parameter_count,
    train_multigrade,
)
from burgers_grade_diagnostics import (
    make_svg_responsive,
    save_dynamic_grade_diagnostics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TUPLE_FIELDS = {
    "widths",
    "depths",
    "epochs",
    "learning_rates",
    "grade_scheduler_steps",
    "grade_scheduler_gammas",
    "residual_basis_modes",
    "spectral_weights",
    "h1_weights",
    "predictor_weights",
}

FNO_NAME = "FNO"
PLAIN_NAME = "Plain Multigrade"
DYNAMIC_NAME = "Full Dynamic Multigrade"


def load_dynamic_model(
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[dynamic.DynamicMGNO, dynamic.Config]:
    """Restore the frozen best Dynamic MGNO without retraining it."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = dict(checkpoint["config"])
    allowed = {field.name for field in fields(dynamic.Config)}
    values = {key: value for key, value in raw.items() if key in allowed}
    values["data"] = Path(values["data"])
    values["output_dir"] = output_dir
    values["device"] = str(device)
    for key in TUPLE_FIELDS:
        if key in values:
            values[key] = tuple(values[key])
    config = dynamic.Config(**values)
    model = dynamic.DynamicMGNO(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, config


def select_plain_width(target_parameters: int, modes: int) -> tuple[int, int]:
    """Choose the plain four-Grade width closest to the Dynamic MGNO budget."""

    candidates: list[tuple[int, int, int]] = []
    for width in range(1, 1025):
        model = PlainMultigradeFNO1d(modes, width, (1, 1, 1, 1))
        parameters = parameter_count(model)
        candidates.append((abs(parameters - target_parameters), width, parameters))
    _, width, parameters = min(candidates)
    return width, parameters


def normalize_history(
    plain_history: list[dict[str, object]],
    fno_history: list[dict[str, object]],
    dynamic_history: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in plain_history:
        rows.append(
            {
                "model": PLAIN_NAME,
                "grade": row["grade"],
                "epoch": row["epoch"],
                "global_epoch": row["global_epoch"],
                "learning_rate": row["learning_rate"],
                "training_relative_l2": row["training_relative_l2"],
                "training_mse": row["training_mse"],
                "epoch_seconds": row["epoch_seconds"],
            }
        )
    for row in fno_history:
        rows.append(
            {
                "model": FNO_NAME,
                "grade": 0,
                "epoch": row["epoch"],
                "global_epoch": row["global_epoch"],
                "learning_rate": row["learning_rate"],
                "training_relative_l2": row["train_relative_l2"],
                "training_mse": row["train_mse"],
                "epoch_seconds": row["elapsed_seconds"],
            }
        )
    for row in dynamic_history:
        rows.append(
            {
                "model": DYNAMIC_NAME,
                "grade": row["grade"],
                "epoch": row["epoch"],
                "global_epoch": row["global_epoch"],
                "learning_rate": row["learning_rate"],
                "training_relative_l2": row["train_relative_l2"],
                "training_mse": row["train_mse"],
                "epoch_seconds": row["elapsed_seconds"],
            }
        )
    return rows


def train_dynamic_mgno(
    config: dynamic.Config,
    train_data: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[dynamic.DynamicMGNO, list[dict[str, object]]]:
    """Retrain Dynamic MGNO so every epoch can be audited after optimization."""

    base.set_seed(config.seed)
    x_train, y_train = train_data
    train_loader = base.make_loader(
        x_train,
        y_train,
        config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    x_validation, y_validation = dynamic.load_validation_data(config)
    validation_loader = base.make_loader(
        x_validation, y_validation, config.batch_size, shuffle=False
    )
    model = dynamic.DynamicMGNO(config).to(device)
    history, _ = dynamic.train_physics_anchored_grade1(
        model, train_loader, validation_loader, config, device
    )
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
        history.extend(grade_history)
        calibration = dynamic.calibrate_grade_on_validation(
            model, validation_loader, grade_index, config, device
        )
        base.save_checkpoint(
            model,
            config.output_dir / "_epoch_checkpoints" / "dynamic_mgno",
            "dynamic_mgno",
            global_offset + config.epochs[grade_index],
            grade_index + 1,
        )
        source_checkpoint = (
            config.output_dir
            / "_epoch_checkpoints"
            / "multigrade"
            / f"multigrade_epoch_{global_offset + config.epochs[grade_index]:04d}.pt"
        )
        target_checkpoint = (
            config.output_dir
            / "_epoch_checkpoints"
            / "dynamic_mgno"
            / f"dynamic_mgno_epoch_{global_offset + config.epochs[grade_index]:04d}.pt"
        )
        if source_checkpoint.is_file():
            source_checkpoint.unlink()
        if calibration["accepted"]:
            used_frequencies.update(
                int(value) for value in profile["selected_indices"]
            )
        global_offset += config.epochs[grade_index]
    source_dir = config.output_dir / "_epoch_checkpoints" / "multigrade"
    target_dir = config.output_dir / "_epoch_checkpoints" / "dynamic_mgno"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("multigrade_epoch_*.pt"):
        epoch = source.stem.rsplit("_", 1)[-1]
        destination = target_dir / f"dynamic_mgno_epoch_{epoch}.pt"
        if destination.exists():
            destination.unlink()
        source.replace(destination)
    if source_dir.is_dir():
        source_dir.rmdir()
    model.eval()
    model.requires_grad_(False)
    return model, history


@torch.no_grad()
def mean_absolute_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return (prediction - target).abs().mean()


@torch.no_grad()
def audit_test_absolute_error(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    model_name: str,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    kind: str,
) -> list[dict[str, object]]:
    """Replay saved epochs only after training and compute full-test MAE."""

    rows: list[dict[str, object]] = []
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    for index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        absolute_sum = 0.0
        elements = 0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            if kind == "dynamic":
                prediction, _, _ = model.forward_up_to(
                    x, int(checkpoint["active_grades"])
                )
            elif kind == "plain":
                prediction, _, _ = model.forward_up_to(
                    x, int(checkpoint["active_grades"])
                )
            else:
                prediction = model(x)
            absolute_sum += float((prediction - target).abs().sum())
            elements += target.numel()
        epoch = int(checkpoint.get("global_epoch", checkpoint.get("epoch")))
        value = absolute_sum / elements
        rows.append(
            {
                "model": model_name,
                "global_epoch": epoch,
                "test_mean_absolute_error": value,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(checkpoints):
            print(
                f"Post-training MAE audit {model_name}: "
                f"epoch={epoch:04d} mae={value:.6e}"
            )
    return rows


def save_plots(
    history: list[dict[str, object]],
    test_audit: list[dict[str, object]],
    output_dir: Path,
) -> None:
    colors = {
        FNO_NAME: "#7C8DA6",
        PLAIN_NAME: "#2A9D8F",
        DYNAMIC_NAME: "#D95D39",
    }
    style = {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#F4F1EA",
        "axes.facecolor": "#FFFDF8",
        "axes.edgecolor": "#A8A29A",
        "axes.labelcolor": "#293241",
        "axes.titlecolor": "#293241",
        "xtick.color": "#566273",
        "ytick.color": "#566273",
        "text.color": "#293241",
        "grid.color": "#D8D2C4",
        "legend.facecolor": "#FFFDF8",
        "legend.edgecolor": "#D8D2C4",
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        for field, ylabel, filename in (
            ("training_relative_l2", "Training relative L2", "01_training_relative_l2.svg"),
            ("training_mse", "Training MSE", "02_training_mse.svg"),
        ):
            figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
            for name, color in colors.items():
                rows = [row for row in history if row["model"] == name]
                axis.plot(
                    [int(row["global_epoch"]) for row in rows],
                    [float(row[field]) for row in rows],
                    color=color,
                    linewidth=2.7 if name == DYNAMIC_NAME else 1.7,
                    alpha=1.0 if name == DYNAMIC_NAME else 0.86,
                    zorder=4 if name == DYNAMIC_NAME else 2,
                    label=name,
                )
            axis.set_yscale("log")
            axis.set_xlabel("Global epoch")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.35)
            axis.legend()
            path = output_dir / filename
            figure.savefig(
                path,
                format="svg",
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
            )
            plt.close(figure)
            make_svg_responsive(path)

        figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
        for name, color in colors.items():
            rows = [row for row in test_audit if row["model"] == name]
            axis.plot(
                [int(row["global_epoch"]) for row in rows],
                [float(row["test_mean_absolute_error"]) for row in rows],
                color=color,
                linewidth=2.7 if name == DYNAMIC_NAME else 1.7,
                alpha=1.0 if name == DYNAMIC_NAME else 0.86,
                zorder=4 if name == DYNAMIC_NAME else 2,
                label=name,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Global epoch")
        axis.set_ylabel("Held-out test mean absolute error")
        axis.grid(alpha=0.35)
        axis.legend()
        path = output_dir / "03_test_mean_absolute_error.svg"
        figure.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)
        make_svg_responsive(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    template_model, dynamic_config = load_dynamic_model(
        args.dynamic_checkpoint, args.output_dir, device
    )
    print("\n=== Full Dynamic Multigrade structure ===")
    print(
        "Grade 0: fixed ETDRK4 coarse operator | "
        f"grid={dynamic_config.coarse_resolution}->{dynamic_config.resolution} | "
        f"solver_steps={dynamic_config.coarse_solver_steps} | "
        "trainable_parameters=0 | frozen=True"
    )
    print("Grade 1 target: truth - Grade 0 output")
    print("Grades 2-4 targets: cumulative residuals left by all frozen predecessors")
    print("Trainable hierarchy: Grade 1 -> Grade 2 -> Grade 3 -> Grade 4\n")
    target_parameters = parameter_count(template_model)
    del template_model
    dynamic_config.epochs = tuple(args.grade_epochs)
    dynamic_config.baseline_epochs = sum(args.grade_epochs)

    x_train, y_train = base.load_burgers_train_data(dynamic_config)
    dynamic_model, dynamic_history = train_dynamic_mgno(
        dynamic_config, (x_train, y_train), device
    )
    plain_width, plain_parameters = select_plain_width(
        target_parameters, dynamic_config.modes
    )
    plain_config = PlainConfig(
        data=dynamic_config.data,
        output_dir=args.output_dir,
        device=args.device,
        seed=dynamic_config.seed,
        ntrain=dynamic_config.ntrain,
        ntest=dynamic_config.ntest,
        resolution=dynamic_config.resolution,
        batch_size=dynamic_config.batch_size,
        modes=dynamic_config.modes,
        width=plain_width,
        four_grade_epochs=tuple(args.grade_epochs),
        baseline_epochs=sum(args.grade_epochs),
        learning_rate=args.learning_rate,
        weight_decay=dynamic_config.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
    )
    plain_config.grade_scheduler_steps = dynamic_config.grade_scheduler_steps
    plain_config.grade_scheduler_gammas = dynamic_config.grade_scheduler_gammas
    base.set_seed(dynamic_config.seed)
    plain_model = PlainMultigradeFNO1d(
        dynamic_config.modes, plain_width, (1, 1, 1, 1)
    ).to(device)
    print(
        f"Plain MG-FNO parameter match: width={plain_width}, "
        f"parameters={plain_parameters}, target={target_parameters}"
    )
    plain_history, plain_time = train_multigrade(
        "Plain-MG-FNO-4",
        plain_model,
        (x_train, y_train),
        tuple(args.grade_epochs),
        plain_config,
        device,
        args.output_dir / "_epoch_checkpoints" / "plain_mg_fno",
    )

    dynamic_config.baseline_epochs = sum(args.grade_epochs)
    matched_fno, fno_history, fno_width = dynamic.train_parameter_matched_fno4(
        (x_train, y_train),
        dynamic_config,
        device,
        target_parameters,
        args.output_dir / "_epoch_checkpoints" / "fno4",
    )
    history = normalize_history(plain_history, fno_history, dynamic_history)

    x_validation, y_validation = dynamic.load_validation_data(dynamic_config)

    def full_dynamic_context(source: torch.Tensor, before: torch.Tensor):
        coarse = dynamic_model.forward_grade0(source)
        features, spectrum, _ = dynamic.dynamic_burgers_context(
            source,
            before,
            dynamic_config,
            coarse_prediction_full=coarse,
        )
        return features, spectrum

    save_dynamic_grade_diagnostics(
        dynamic_model,
        dynamic_config,
        x_validation,
        y_validation,
        dynamic_history,
        figure_dir,
        DYNAMIC_NAME,
        full_dynamic_context,
        True,
    )

    print(
        "[Final test] Training is complete; replaying all frozen epochs on "
        "the held-out test set"
    )
    x_test, y_test = base.load_burgers_test_data(dynamic_config)
    grade0_test_relative_l2 = dynamic.evaluate_coarse_solver(
        dynamic_model,
        x_test,
        y_test,
        dynamic_config.batch_size,
        device,
    )
    test_loader = base.make_loader(
        x_test, y_test, dynamic_config.batch_size, shuffle=False
    )
    test_audit = []
    test_audit.extend(
        audit_test_absolute_error(
            matched_fno,
            args.output_dir / "_epoch_checkpoints" / "fno4",
            FNO_NAME,
            test_loader,
            device,
            "fno4",
        )
    )
    test_audit.extend(
        audit_test_absolute_error(
            plain_model,
            args.output_dir / "_epoch_checkpoints" / "plain_mg_fno",
            PLAIN_NAME,
            test_loader,
            device,
            "plain",
        )
    )
    test_audit.extend(
        audit_test_absolute_error(
            dynamic_model,
            args.output_dir / "_epoch_checkpoints" / "dynamic_mgno",
            DYNAMIC_NAME,
            test_loader,
            device,
            "dynamic",
        )
    )
    dynamic_metrics = base.evaluate_multigrade(
        dynamic_model, test_loader, device, dynamic_config.grade_count
    )
    plain_progression, _ = evaluate_plain_multigrade(
        plain_model, test_loader, device
    )
    fno_metrics = base.evaluate_baseline(matched_fno, test_loader, device)
    test_values = {
        "FNO4": float(fno_metrics["relative_l2"]),
        "Plain MG-FNO": float(plain_progression[-1]),
        "Dynamic MGNO": float(dynamic_metrics["relative_l2"]),
    }
    result = {
        "configuration_checkpoint": str(args.dynamic_checkpoint.resolve()),
        "protocol": {
            "models": [
                FNO_NAME,
                PLAIN_NAME,
                DYNAMIC_NAME,
            ],
            "plain_rule": "Only cumulative residual targets, frozen predecessors, immediate predecessor features, and additive outputs",
            "excluded_from_plain": "No coarse solver, physical defect, adaptive residual spectrum, Fourier feature mapping, H1 loss, or validation selection",
            "full_dynamic_grade0": "A deterministic label-free ETDRK4 low-fidelity operator with zero trainable parameters; its output is the frozen cumulative base and its residual is the Grade 1 target",
            "test_rule": "Every epoch is saved during training; only after all optimization are checkpoints replayed on the final 200 samples",
        },
        "metrics": {
            "fno4_test_relative_l2": test_values["FNO4"],
            "plain_mg_fno_test_relative_l2": test_values["Plain MG-FNO"],
            "dynamic_mgno_test_relative_l2": test_values["Dynamic MGNO"],
            "grade0_test_relative_l2": grade0_test_relative_l2,
            "grade0_trainable_parameters": parameter_count(dynamic_model.grade0),
            "plain_grade_progression": plain_progression,
            "fno4_over_dynamic": test_values["FNO4"] / test_values["Dynamic MGNO"],
            "plain_over_dynamic": test_values["Plain MG-FNO"] / test_values["Dynamic MGNO"],
            "dynamic_parameters": target_parameters,
            "plain_parameters": plain_parameters,
            "plain_width": plain_width,
            "fno4_parameters": parameter_count(matched_fno),
            "fno4_width": fno_width,
            "plain_training_seconds": plain_time,
        },
        "dynamic_config": asdict(dynamic_config),
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    with (args.output_dir / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "test_absolute_error_by_epoch.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_audit[0]))
        writer.writeheader()
        writer.writerows(test_audit)
    torch.save(dynamic_model.state_dict(), args.output_dir / "dynamic_mgno_1111.pt")
    torch.save(plain_model.state_dict(), args.output_dir / "plain_mg_fno_1111.pt")
    torch.save(matched_fno.state_dict(), args.output_dir / "parameter_matched_fno4.pt")
    save_plots(history, test_audit, figure_dir)
    for source in args.output_dir.glob("grade_*_adaptive_residual_spectrum.svg"):
        destination = figure_dir / source.name
        if destination.exists():
            destination.unlink()
        source.replace(destination)
    checkpoint_root = args.output_dir / "_epoch_checkpoints"
    if checkpoint_root.is_dir():
        shutil.rmtree(checkpoint_root)

    print("\nFINAL THREE-MODEL COMPARISON")
    print("  Full Dynamic hierarchy: Grade 0-4 (5 total; 4 trainable)")
    print(f"  FNO test relative L2: {test_values['FNO4']:.6e}")
    print(f"  Plain Multigrade test relative L2: {test_values['Plain MG-FNO']:.6e}")
    print(f"  Full Dynamic Multigrade test relative L2: {test_values['Dynamic MGNO']:.6e}")
    print(f"  Internal Grade 0 test relative L2: {grade0_test_relative_l2:.6e}")
    print("  Internal Grade 0 trainable parameters: 0")
    print(f"  FNO/Dynamic gain: {result['metrics']['fno4_over_dynamic']:.3f}x")
    print(f"  Plain/Dynamic gain: {result['metrics']['plain_over_dynamic']:.3f}x")
    print(
        f"  Parameters: FNO={result['metrics']['fno4_parameters']}, "
        f"Plain={plain_parameters}, Dynamic={target_parameters}"
    )
    print(f"  Output: {args.output_dir.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter-controlled FNO, Plain Multigrade, and Full Dynamic Multigrade comparison"
    )
    parser.add_argument("--dynamic-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grade-epochs", nargs=4, type=int, default=(100, 100, 150, 150))
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
