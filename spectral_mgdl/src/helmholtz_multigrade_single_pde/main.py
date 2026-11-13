"""Fixed k=200pi single-PDE frozen multigrade recurrent-DST-FNO."""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from config import CONFIG
from models import GradeDSTFNO, Helmholtz1D
from plotting import plot_grade, plot_summary, save_all_epoch_metrics
from training import (
    SCALAR_CURVE_KEYS,
    apply_grade,
    assert_frozen,
    make_plot_record,
    make_problem,
    save_grade_data,
    solution_metrics,
    train_one_grade,
)


def snapshot(grades):
    return [
        {
            name: tensor.detach().clone()
            for name, tensor in grade.state_dict().items()
        }
        for grade in grades
    ]


def assert_unchanged(grades, states):
    for grade_id, (grade, state) in enumerate(
        zip(grades, states), start=1
    ):
        for name, current in grade.state_dict().items():
            if not torch.equal(current, state[name]):
                raise RuntimeError(
                    f"Frozen Grade {grade_id} changed: {name}"
                )


def main():
    config = CONFIG
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    op = Helmholtz1D(
        n=config.n,
        kappa=config.kappa_over_pi * math.pi,
        shift_ratio=config.shift_ratio,
    )
    problem = make_problem(op)
    u = torch.zeros_like(problem["u"])
    grades = nn.ModuleList()
    curves = []
    records = []
    start_time = time.perf_counter()

    print("[*] task=single fixed PDE; NOT operator learning")
    print(
        f"[*] kappa={op.kappa:.12f} = {config.kappa_over_pi}pi "
        f"N={config.n} grades={config.n_grades}"
    )
    print(f"[*] exact resonant modes={op.k[op.null_mask].tolist()}")
    print(
        f"[*] complex truth modes="
        f"{list(zip(problem['truth_k'].tolist(), problem['truth_a'].tolist()))}"
    )
    print(f"[*] initial={solution_metrics(op, u, problem)}")

    for grade_id in range(1, config.n_grades + 1):
        assert_frozen(grades)
        frozen_states = snapshot(grades)
        grade = GradeDSTFNO(
            n=config.n, width=config.width, depth=config.depth
        )
        if grades:
            grade.load_state_dict(grades[-1].state_dict())
        grade.train()
        grade.requires_grad_(True)
        before = u.clone()
        before_metrics = solution_metrics(op, before, problem)

        print(
            f"\n=== TRAIN GRADE {grade_id}: learn residual of frozen "
            f"Grades 1..{grade_id-1} ==="
        )
        _, curve = train_one_grade(
            grade_id, grade, op, problem, before, config
        )
        assert_unchanged(grades, frozen_states)
        candidate = apply_grade(
            grade, op, problem, before, config
        ).detach()
        after_metrics = solution_metrics(op, candidate, problem)
        if (
            after_metrics["relative_residual"]
            >= before_metrics["relative_residual"]
        ):
            print(
                f"  REJECT Grade {grade_id}: relative residual "
                "did not improve."
            )
            break

        grades.append(grade)
        assert_frozen(grades)
        u = candidate
        curves.append(curve)
        record = make_plot_record(
            op, problem, before, u, grade, grade_id, curve, config
        )
        records.append(record)
        plots = plot_grade(record, out_dir, op.kappa)
        data = save_grade_data(
            out_dir,
            op,
            problem,
            before,
            u,
            grade,
            grade_id,
            curve,
            config,
        )
        print(f"  frozen Grade {grade_id}: {after_metrics}")
        print(
            f"  active frequencies={record['selected_k'].tolist()}"
        )
        print(f"  plots={plots}")
        print(f"  data={data}")

    checkpoint = out_dir / "frozen_multigrade_single_pde.pt"
    torch.save(
        {
            "grades": grades.state_dict(),
            "config": asdict(config),
            "truth_k": problem["truth_k"],
            "truth_a": problem["truth_a"],
            "task": "single fixed PDE; not operator learning",
        },
        checkpoint,
    )
    summary = plot_summary(records, out_dir)
    epoch_csv, epoch_plot = save_all_epoch_metrics(
        curves, out_dir, SCALAR_CURVE_KEYS
    )
    print(f"\nFINAL={solution_metrics(op, u, problem)}")
    print(f"wall={time.perf_counter() - start_time:.1f}s")
    print(f"checkpoint={checkpoint}")
    print(f"summary={summary}")
    print(f"all_epoch_csv={epoch_csv}")
    print(f"all_epoch_plot={epoch_plot}")


if __name__ == "__main__":
    main()
