"""Plain Burgers FNO versus frozen two-Grade and four-Grade decompositions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from burgers_mgno_best_recovery import (
    FNO4Baseline1d,
    Projection1d,
    SpectralConv1d,
    load_burgers_test_data,
    load_burgers_train_data,
    make_loader,
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
    width: int = 64
    two_grade_depths: tuple[int, ...] = (2, 2)
    two_grade_epochs: tuple[int, ...] = (200, 300)
    four_grade_depths: tuple[int, ...] = (1, 1, 1, 1)
    four_grade_epochs: tuple[int, ...] = (100, 100, 150, 150)
    baseline_epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    overwrite: bool = False

    def validate(self) -> None:
        if not self.data.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {self.data}")
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This experiment requires CUDA")
        if 8192 % self.resolution:
            raise ValueError("Resolution must divide the native 8192-point grid")
        if self.two_grade_depths != (2, 2):
            raise ValueError("The plain two-Grade ablation must be 2+2")
        if self.four_grade_depths != (1, 1, 1, 1):
            raise ValueError("The plain four-Grade ablation must be 1+1+1+1")
        if sum(self.two_grade_depths) != 4 or sum(self.four_grade_depths) != 4:
            raise ValueError("Each multigrade model must contain four Fourier cores")
        if sum(self.two_grade_epochs) != self.baseline_epochs:
            raise ValueError("Two-Grade and FNO4 total epochs must match")
        if sum(self.four_grade_epochs) != self.baseline_epochs:
            raise ValueError("Four-Grade and FNO4 total epochs must match")


def sample_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).flatten(1).norm(dim=1)
    scale = target.flatten(1).norm(dim=1).clamp_min(1.0e-12)
    return error / scale


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class FourierGrade1(nn.Module):
    """Lift the paper input and return a projected solution and final feature."""

    def __init__(
        self,
        modes: int,
        width: int,
        depth: int,
        activation_flags: tuple[bool, ...],
    ) -> None:
        super().__init__()
        self.lifting = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.projection = Projection1d(width)
        self.activation_flags = activation_flags

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.lifting(x).permute(0, 2, 1)
        for activate, spectral, pointwise in zip(
            self.activation_flags,
            self.spectral_layers,
            self.pointwise_layers,
            strict=True,
        ):
            feature = spectral(feature) + pointwise(feature)
            if activate:
                feature = F.relu(feature)
        return self.projection(feature), feature


class FourierResidualGrade(nn.Module):
    """Lift a frozen predecessor feature and predict an additive residual."""

    def __init__(
        self,
        modes: int,
        width: int,
        depth: int,
        activation_flags: tuple[bool, ...],
    ) -> None:
        super().__init__()
        self.lifting = nn.Linear(width, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(depth)
        )
        self.pointwise_layers = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(depth)
        )
        self.projection = Projection1d(width)
        self.activation_flags = activation_flags

    def forward(self, predecessor_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.lifting(
            predecessor_feature.permute(0, 2, 1)
        ).permute(0, 2, 1)
        for activate, spectral, pointwise in zip(
            self.activation_flags,
            self.spectral_layers,
            self.pointwise_layers,
            strict=True,
        ):
            feature = spectral(feature) + pointwise(feature)
            if activate:
                feature = F.relu(feature)
        return self.projection(feature), feature


class PlainMultigradeFNO1d(nn.Module):
    """A plain frozen multigrade decomposition of the four paper FNO cores."""

    def __init__(self, modes: int, width: int, depths: tuple[int, ...]) -> None:
        super().__init__()
        total_depth = sum(depths)
        global_flags = tuple(index < total_depth - 1 for index in range(total_depth))
        offset = 0
        first_flags = global_flags[: depths[0]]
        self.grade1 = FourierGrade1(modes, width, depths[0], first_flags)
        offset += depths[0]
        grades: list[FourierResidualGrade] = []
        for depth in depths[1:]:
            flags = global_flags[offset : offset + depth]
            grades.append(FourierResidualGrade(modes, width, depth, flags))
            offset += depth
        self.residual_grades = nn.ModuleList(grades)
        self.depths = depths

    @property
    def grade_count(self) -> int:
        return len(self.depths)

    def freeze_before(self, grade_index: int) -> None:
        self.grade1.eval()
        self.grade1.requires_grad_(grade_index == 0)
        for index, grade in enumerate(self.residual_grades, start=1):
            grade.eval()
            grade.requires_grad_(index == grade_index)

    def forward_up_to(
        self, x: torch.Tensor, active_grades: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        cumulative, feature = self.grade1(x)
        predictions = [cumulative]
        for grade in self.residual_grades[: active_grades - 1]:
            correction, feature = grade(feature)
            cumulative = cumulative + correction
            predictions.append(cumulative)
        return cumulative, feature, predictions


@torch.no_grad()
def evaluate_multigrade(
    model: PlainMultigradeFNO1d,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[float], torch.Tensor]:
    totals = [0.0] * model.grade_count
    samples = 0
    outputs: list[torch.Tensor] = []
    model.eval()
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        prediction, _, grade_predictions = model.forward_up_to(x, model.grade_count)
        for index, value in enumerate(grade_predictions):
            totals[index] += float(sample_relative_l2(value, target).sum())
        samples += x.shape[0]
        outputs.append(prediction.cpu())
    return [value / samples for value in totals], torch.cat(outputs)


@torch.no_grad()
def evaluate_fno4(
    model: FNO4Baseline1d,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    total = 0.0
    samples = 0
    outputs: list[torch.Tensor] = []
    model.eval()
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        prediction = model(x)
        total += float(sample_relative_l2(prediction, target).sum())
        samples += x.shape[0]
        outputs.append(prediction.cpu())
    return total / samples, torch.cat(outputs)


def make_train_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Config,
    seed_offset: int,
) -> torch.utils.data.DataLoader:
    return make_loader(
        x,
        y,
        config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + seed_offset),
    )


def train_multigrade(
    name: str,
    model: PlainMultigradeFNO1d,
    train_data: tuple[torch.Tensor, torch.Tensor],
    epochs: tuple[int, ...],
    config: Config,
    device: torch.device,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, object]], float]:
    x_train, y_train = train_data
    history: list[dict[str, object]] = []
    total_start = time.perf_counter()
    global_epoch = 0
    for grade_index, grade_epochs in enumerate(epochs):
        model.freeze_before(grade_index)
        active = model.grade1 if grade_index == 0 else model.residual_grades[grade_index - 1]
        frozen_snapshot = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith(
                "grade1" if grade_index == 0 else f"residual_grades.{grade_index - 1}"
            )
        }
        optimizer = torch.optim.Adam(
            active.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler_steps = getattr(config, "grade_scheduler_steps", None)
        scheduler_gammas = getattr(config, "grade_scheduler_gammas", None)
        scheduler_step = (
            int(scheduler_steps[grade_index])
            if scheduler_steps is not None
            else config.scheduler_step
        )
        scheduler_gamma = (
            float(scheduler_gammas[grade_index])
            if scheduler_gammas is not None
            else config.scheduler_gamma
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step,
            gamma=scheduler_gamma,
        )
        loader = make_train_loader(x_train, y_train, config, grade_index)
        for epoch in range(1, grade_epochs + 1):
            epoch_start = time.perf_counter()
            relative_sum = mse_sum = 0.0
            samples = 0
            active.train()
            learning_rate = optimizer.param_groups[0]["lr"]
            for x, target in loader:
                x, target = x.to(device), target.to(device)
                if grade_index == 0:
                    predecessor = torch.zeros_like(target)
                    previous_feature = torch.empty(0, device=device)
                else:
                    with torch.no_grad():
                        predecessor, previous_feature, _ = model.forward_up_to(
                            x, grade_index
                        )
                optimizer.zero_grad(set_to_none=True)
                if grade_index == 0:
                    prediction, _ = model.grade1(x)
                else:
                    correction, _ = active(previous_feature)
                    prediction = predecessor + correction
                relative = sample_relative_l2(prediction, target).mean()
                relative.backward()
                optimizer.step()
                batch = x.shape[0]
                relative_sum += float(relative.detach()) * batch
                mse_sum += float(F.mse_loss(prediction, target).detach()) * batch
                samples += batch
            scheduler.step()
            global_epoch += 1
            row = {
                "model": name,
                "grade": grade_index + 1,
                "epoch": epoch,
                "global_epoch": global_epoch,
                "learning_rate": learning_rate,
                "training_relative_l2": relative_sum / samples,
                "training_mse": mse_sum / samples,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "active_grades": grade_index + 1,
                        "global_epoch": global_epoch,
                    },
                    checkpoint_dir / f"epoch_{global_epoch:04d}.pt",
                )
            print(
                f"{name} Grade{grade_index + 1} epoch={epoch:04d} "
                f"epoch_time={row['epoch_seconds']:.6f}s "
                f"lr={learning_rate:.6e} "
                f"train_mse={row['training_mse']:.6e} "
                f"train_l2={row['training_relative_l2']:.6e}"
            )
        model.freeze_before(-1)
        for key, reference in frozen_snapshot.items():
            if not torch.equal(model.state_dict()[key], reference):
                raise RuntimeError(f"Frozen parameter changed: {name}.{key}")
    model.eval()
    model.requires_grad_(False)
    return history, time.perf_counter() - total_start


def train_fno4(
    train_data: tuple[torch.Tensor, torch.Tensor],
    config: Config,
    device: torch.device,
) -> tuple[FNO4Baseline1d, list[dict[str, object]], float]:
    set_seed(config.seed)
    model = FNO4Baseline1d(config.modes, config.width).to(device)
    loader = make_train_loader(*train_data, config, 0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.scheduler_step, gamma=config.scheduler_gamma
    )
    history: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for epoch in range(1, config.baseline_epochs + 1):
        epoch_start = time.perf_counter()
        relative_sum = mse_sum = 0.0
        samples = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            relative = sample_relative_l2(prediction, target).mean()
            relative.backward()
            optimizer.step()
            batch = x.shape[0]
            relative_sum += float(relative.detach()) * batch
            mse_sum += float(F.mse_loss(prediction, target).detach()) * batch
            samples += batch
        scheduler.step()
        row = {
            "model": "FNO4",
            "grade": 0,
            "epoch": epoch,
            "global_epoch": epoch,
            "learning_rate": learning_rate,
            "training_relative_l2": relative_sum / samples,
            "training_mse": mse_sum / samples,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"FNO4 epoch={epoch:04d} epoch_time={row['epoch_seconds']:.6f}s "
            f"lr={learning_rate:.6e} train_mse={row['training_mse']:.6e} "
            f"train_l2={row['training_relative_l2']:.6e}"
        )
    model.eval()
    model.requires_grad_(False)
    return model, history, time.perf_counter() - total_start


def save_training_plot(
    histories: list[dict[str, object]], output_dir: Path
) -> None:
    colors = {"FNO4": "#252525", "MG-FNO-2": "#2C7FB8", "MG-FNO-4": "#D95F0E"}
    for field, title, filename in (
        ("training_relative_l2", "Training relative L2", "01_training_relative_l2.svg"),
        ("training_mse", "Training MSE", "02_training_mse.svg"),
    ):
        figure, axis = plt.subplots(figsize=(8.4, 5.2))
        for name in colors:
            rows = [row for row in histories if row["model"] == name]
            axis.plot(
                [int(row["global_epoch"]) for row in rows],
                [float(row[field]) for row in rows],
                color=colors[name],
                linewidth=1.6,
                label=name,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Global epoch")
        axis.set_ylabel(title)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / filename, format="svg", bbox_inches="tight")
        plt.close(figure)


def run(config: Config) -> dict[str, object]:
    config.validate()
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory is not empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    x_train, y_train = load_burgers_train_data(config)

    set_seed(config.seed)
    two_grade = PlainMultigradeFNO1d(
        config.modes, config.width, config.two_grade_depths
    ).to(device)
    two_history, two_time = train_multigrade(
        "MG-FNO-2",
        two_grade,
        (x_train, y_train),
        config.two_grade_epochs,
        config,
        device,
    )

    set_seed(config.seed)
    four_grade = PlainMultigradeFNO1d(
        config.modes, config.width, config.four_grade_depths
    ).to(device)
    four_history, four_time = train_multigrade(
        "MG-FNO-4",
        four_grade,
        (x_train, y_train),
        config.four_grade_epochs,
        config,
        device,
    )

    fno4, fno_history, fno_time = train_fno4((x_train, y_train), config, device)
    histories = two_history + four_history + fno_history

    print("[Final test] All three models are frozen; loading held-out data now")
    x_test, y_test = load_burgers_test_data(config)
    test_loader = make_loader(
        x_test, y_test, config.batch_size, shuffle=False
    )
    two_grade_rse, two_prediction = evaluate_multigrade(
        two_grade, test_loader, device
    )
    four_grade_rse, four_prediction = evaluate_multigrade(
        four_grade, test_loader, device
    )
    fno_rse, fno_prediction = evaluate_fno4(fno4, test_loader, device)

    result = {
        "config": {**asdict(config), "data": str(config.data.resolve()), "output_dir": str(config.output_dir.resolve())},
        "protocol": {
            "data": "First 1000 samples train; final 200 samples test",
            "test": "Held-out test samples are loaded only after all optimization",
            "objective": "Original sample-wise relative L2 on the cumulative prediction",
            "multigrade": "Every completed Grade is frozen and every later Grade predicts its cumulative residual",
            "features": "Every later Grade receives only the final Fourier feature of its immediate frozen predecessor",
            "excluded": "No physics solver, defect, adaptive spectrum, frequency mapping, H1 loss, or validation selection",
        },
        "metrics": {
            "fno4_test_relative_l2": fno_rse,
            "two_grade_test_relative_l2": two_grade_rse[-1],
            "four_grade_test_relative_l2": four_grade_rse[-1],
            "two_grade_progression": two_grade_rse,
            "four_grade_progression": four_grade_rse,
            "fno4_over_two_grade": fno_rse / two_grade_rse[-1],
            "fno4_over_four_grade": fno_rse / four_grade_rse[-1],
            "two_grade_parameters": parameter_count(two_grade),
            "four_grade_parameters": parameter_count(four_grade),
            "fno4_parameters": parameter_count(fno4),
            "two_grade_training_seconds": two_time,
            "four_grade_training_seconds": four_time,
            "fno4_training_seconds": fno_time,
        },
    }
    with (config.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    with (config.output_dir / "training_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    torch.save(two_grade.state_dict(), config.output_dir / "plain_two_grade.pt")
    torch.save(four_grade.state_dict(), config.output_dir / "plain_four_grade.pt")
    torch.save(fno4.state_dict(), config.output_dir / "paper_fno4.pt")
    torch.save(
        {
            "target": y_test,
            "fno4": fno_prediction,
            "two_grade": two_prediction,
            "four_grade": four_prediction,
        },
        config.output_dir / "test_predictions.pt",
    )
    save_training_plot(histories, config.output_dir)

    print("\nFINAL PLAIN MULTIGRADE ABLATION")
    print(f"  FNO4 test relative L2: {fno_rse:.6e}")
    print(f"  Plain MG-FNO 2+2 test relative L2: {two_grade_rse[-1]:.6e}")
    print(f"  Plain MG-FNO 1+1+1+1 test relative L2: {four_grade_rse[-1]:.6e}")
    print(f"  FNO4/2-Grade ratio: {fno_rse / two_grade_rse[-1]:.3f}x")
    print(f"  FNO4/4-Grade ratio: {fno_rse / four_grade_rse[-1]:.3f}x")
    print(f"  FNO4 parameters: {parameter_count(fno4)}")
    print(f"  2-Grade parameters: {parameter_count(two_grade)}")
    print(f"  4-Grade parameters: {parameter_count(four_grade)}")
    print(f"  Output: {config.output_dir.resolve()}")
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Paper FNO4 versus plain two-Grade and four-Grade Burgers models"
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
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
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
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    run(parse_args())
