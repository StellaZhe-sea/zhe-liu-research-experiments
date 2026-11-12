"""CPU entry point for recurrent-derived frozen multigrade operator learning."""

import math
import os
import time
from dataclasses import asdict

# These must be set before importing NumPy/PyTorch on Windows.  Using MKL's
# sequential layer avoids loading a second Intel OpenMP runtime from Anaconda.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

from config import CONFIG
from models import BatchedHelmholtz1D, GradeDSTFNO
from plotting import (
    plot_grade_diagnostics,
    plot_summary,
    save_and_plot_all_epoch_metrics,
)
from training import (
    SCALAR_CURVE_KEYS,
    apply_grade,
    assert_previous_grades_frozen,
    make_dataset,
    make_plot_record,
    metrics,
    save_grade_data,
    train_one_grade,
)


def state_snapshot(grades):
    return [
        {
            name: tensor.detach().clone()
            for name, tensor in grade.state_dict().items()
        }
        for grade in grades
    ]


def assert_states_unchanged(grades, snapshots):
    for grade_id, (grade, snapshot) in enumerate(
        zip(grades, snapshots), start=1
    ):
        for name, current in grade.state_dict().items():
            if not torch.equal(current, snapshot[name]):
                raise RuntimeError(
                    f"Frozen Grade {grade_id} changed at parameter {name}."
                )


def format_metrics(result):
    return (
        f"mean_RSE={result['mean_rse']:.6e}, "
        f"max_RSE={result['max_rse']:.6e}, "
        f"mean_rel_res={result['mean_rel_res']:.6e}"
    )


def main():
    config = CONFIG
    torch.set_default_dtype(torch.float64)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "HELMHOLTZ_DEVICE requests CUDA, but this PyTorch environment "
            "cannot access a CUDA device. Use run_cpu.ps1 or repair .venv-gpu."
        )
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    op = BatchedHelmholtz1D(
        n=config.n,
        shift_ratio=config.shift_ratio,
        device=config.device,
    )
    train = make_dataset(config.train_size, op, seed=1001, config=config)
    validation = make_dataset(config.val_size, op, seed=5001, config=config)
    test = make_dataset(config.test_size, op, seed=9001, config=config)
    u_train = torch.zeros_like(train["u"])
    u_validation = torch.zeros_like(validation["u"])

    grades = nn.ModuleList()
    records = []
    accepted_curves = []
    print(
        f"[*] device={config.device} dtype={torch.get_default_dtype()} "
        f"train/validation/unseen-test="
        f"{config.train_size}/{config.val_size}/{config.test_size}"
    )
    if config.device.startswith("cuda"):
        print(f"[*] accelerator={torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        "[*] architecture=independent recurrent DST-FNO grades; "
        "physical state only; previous grades frozen"
    )
    print("[*] unseen test is sealed until all Grade selection is complete")
    wall_start = time.perf_counter()

    for grade_id in range(1, config.n_grades + 1):
        assert_previous_grades_frozen(grades)
        frozen_snapshots = state_snapshot(grades)
        grade = GradeDSTFNO(
            n=config.n, width=config.width, depth=config.depth
        ).to(config.device)
        if grades:
            grade.load_state_dict(grades[-1].state_dict())
        grade.train()
        grade.requires_grad_(True)

        print(
            f"\n=== TRAIN GRADE {grade_id}: "
            f"learn residual left by {grade_id - 1} frozen grades ==="
        )
        u_validation_before = u_validation.clone()
        validation_before = metrics(op, validation, u_validation_before)
        _, curve = train_one_grade(
            grade_id,
            grade,
            op,
            train,
            u_train,
            validation,
            u_validation,
            config,
        )
        assert_states_unchanged(grades, frozen_snapshots)

        candidate_validation = apply_grade(
            grade, op, validation, u_validation, config
        ).detach()
        validation_after = metrics(op, validation, candidate_validation)
        if validation_after["mean_rse"] >= validation_before["mean_rse"]:
            rejected_record = make_plot_record(
                op, validation, u_validation_before, candidate_validation, grade,
                grade_id, curve, config, split="validation", status="rejected",
            )
            rejected_plots = plot_grade_diagnostics(rejected_record, out_dir)
            print(
                f"  REJECT Grade {grade_id}: validation solution RSE "
                "did not improve."
            )
            print(f"  rejected plots={rejected_plots}")
            break

        grades.append(grade)
        assert_previous_grades_frozen(grades)
        u_train = apply_grade(
            grade, op, train, u_train, config
        ).detach()
        u_validation = candidate_validation

        train_metrics = metrics(op, train, u_train)
        validation_metrics = metrics(op, validation, u_validation)
        print(f"  frozen Grade {grade_id}")
        print(f"    train:      {format_metrics(train_metrics)}")
        print(f"    validation: {format_metrics(validation_metrics)}")

        record = make_plot_record(
            op,
            validation,
            u_validation_before,
            u_validation,
            grade,
            grade_id,
            curve,
            config,
            split="validation",
            status="accepted",
        )
        records.append(record)
        accepted_curves.append(curve)
        grade_plots = plot_grade_diagnostics(record, out_dir)
        data_paths = save_grade_data(
            out_dir,
            op,
            validation,
            u_validation_before,
            u_validation,
            grade,
            grade_id,
            curve,
            validation,
            config,
        )
        print(f"  plots={grade_plots}")
        print(f"  data={data_paths}")

    checkpoint = out_dir / "recurrent_frozen_multigrade_operator.pt"
    torch.save(
        {
            "grades": grades.state_dict(),
            "config": asdict(config),
            "architecture": (
                "independent GradeDSTFNO sequence derived from "
                "RecurrentPhysicalDSTFNO"
            ),
        },
        checkpoint,
    )
    summary = plot_summary(records, out_dir)
    all_epoch_csv, all_epoch_plot = save_and_plot_all_epoch_metrics(
        accepted_curves, out_dir, SCALAR_CURVE_KEYS
    )
    # Strict protocol: the unseen test is evaluated exactly once, after all
    # Grade selection and early stopping decisions are final.
    u_test = torch.zeros_like(test["u"])
    for grade in grades:
        u_test = apply_grade(grade, op, test, u_test, config).detach()
    final_test = metrics(op, test, u_test)
    zero_test = metrics(op, test, torch.zeros_like(test["u"]))
    preconditioned_u = op.shifted_precondition(-test["f"], test["kappa"])
    preconditioned_test = metrics(op, test, preconditioned_u)
    direct_u = op.direct_solve(test["f"], test["kappa"])
    direct_test = metrics(op, test, direct_u)
    final_residual = op.residual(u_test, test["f"], test["kappa"])
    np.savez_compressed(
        out_dir / "final_unseen_test_all_points.npz",
        x=op.x.cpu().numpy(),
        kappa=test["kappa"].cpu().numpy(),
        truth=test["u"].cpu().numpy(),
        forcing=test["f"].cpu().numpy(),
        prediction=u_test.cpu().numpy(),
        residual=final_residual.cpu().numpy(),
        rse=final_test["rse"],
        relative_residual=final_test["relative_residual"],
    )
    active_counts = [
        int(np.count_nonzero(record["gate"] > 0.5)) for record in records
    ]

    print(f"\nUNSEEN TEST zero:           {format_metrics(zero_test)}")
    print(f"UNSEEN TEST preconditioner: {format_metrics(preconditioned_test)}")
    print(f"UNSEEN TEST neural model:   {format_metrics(final_test)}")
    print(f"UNSEEN TEST exact DST:      {format_metrics(direct_test)}")
    print(f"focused frequencies (soft gate > 0.5) per grade={active_counts}")
    print(f"wall={time.perf_counter() - wall_start:.1f}s")
    print(f"checkpoint={checkpoint}")
    print(f"summary={summary}")
    print(f"all_epoch_csv={all_epoch_csv}")
    print(f"all_epoch_plot={all_epoch_plot}")


if __name__ == "__main__":
    main()
