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
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    op = BatchedHelmholtz1D(
        n=config.n,
        shift_ratio=config.shift_ratio,
        device=config.device,
    )
    train = make_dataset(config.train_size, op, seed=1001)
    validation = make_dataset(config.val_size, op, seed=5001)
    test = make_dataset(config.test_size, op, seed=9001)
    u_train = torch.zeros_like(train["u"])
    u_validation = torch.zeros_like(validation["u"])
    u_test = torch.zeros_like(test["u"])

    grades = nn.ModuleList()
    records = []
    accepted_curves = []
    initial_test = metrics(op, test, u_test)
    print(
        f"[*] device={config.device} dtype={torch.get_default_dtype()} "
        f"train/validation/unseen-test="
        f"{config.train_size}/{config.val_size}/{config.test_size}"
    )
    print(
        "[*] architecture=independent recurrent DST-FNO grades; "
        "physical state only; previous grades frozen"
    )
    print(f"[*] initial unseen test: {format_metrics(initial_test)}")
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
        validation_before = metrics(op, validation, u_validation)
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
        if (
            validation_after["mean_rel_res"]
            >= validation_before["mean_rel_res"]
        ):
            print(
                f"  REJECT Grade {grade_id}: validation residual "
                "did not improve."
            )
            break

        grades.append(grade)
        assert_previous_grades_frozen(grades)
        u_train = apply_grade(
            grade, op, train, u_train, config
        ).detach()
        u_validation = candidate_validation

        # Strict zero-shot test: first touched after this grade is frozen.
        u_test_before = u_test.clone()
        u_test = apply_grade(
            grade, op, test, u_test, config
        ).detach()
        train_metrics = metrics(op, train, u_train)
        validation_metrics = metrics(op, validation, u_validation)
        test_metrics = metrics(op, test, u_test)
        print(f"  frozen Grade {grade_id}")
        print(f"    train:      {format_metrics(train_metrics)}")
        print(f"    validation: {format_metrics(validation_metrics)}")
        print(f"    unseen test:{format_metrics(test_metrics)}")

        record = make_plot_record(
            op,
            test,
            u_test_before,
            u_test,
            grade,
            grade_id,
            curve,
            config,
        )
        records.append(record)
        accepted_curves.append(curve)
        grade_plots = plot_grade_diagnostics(record, out_dir)
        data_paths = save_grade_data(
            out_dir,
            op,
            test,
            u_test_before,
            u_test,
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
    final_test = metrics(op, test, u_test)
    active_counts = [
        int(np.count_nonzero(record["gate"])) for record in records
    ]

    print(f"\nFINAL unseen test: {format_metrics(final_test)}")
    print(f"adaptive active frequencies per grade={active_counts}")
    print(f"wall={time.perf_counter() - wall_start:.1f}s")
    print(f"checkpoint={checkpoint}")
    print(f"summary={summary}")
    print(f"all_epoch_csv={all_epoch_csv}")
    print(f"all_epoch_plot={all_epoch_plot}")


if __name__ == "__main__":
    main()
