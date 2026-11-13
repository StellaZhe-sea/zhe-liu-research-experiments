import copy
import math
import time

import numpy as np
import torch

from models import dst1_ortho, grade_correction, idst1_ortho


SCALAR_CURVE_KEYS = (
    "loss",
    "pde",
    "spectral_pde",
    "h1_step",
    "null",
    "monotone",
    "rse",
    "relative_residual",
    "pinn",
    "lr",
    "next_lr",
    "lr_reductions",
    "elapsed",
    "best_loss",
    "best_pinn",
    "training_early_stop_wait",
    "pinn_early_stop_wait",
    "early_stop_wait",
)


def make_problem(op):
    """The exact complex-spectrum kappa=200pi problem from recurrent code."""
    truth_k = np.array(
        [3, 7, 11, 19, 31, 47, 73, 103, 137, 173, 199, 201, 229, 247]
    )
    truth_a = np.array(
        [
            1.0,
            0.55,
            -0.32,
            7.0e-3,
            -5.0e-3,
            3.5e-3,
            -2.5e-3,
            8.0e-5,
            -6.0e-5,
            4.5e-5,
            -3.5e-5,
            3.0e-5,
            -2.0e-5,
            1.5e-5,
        ]
    )
    u_true = torch.sum(
        torch.tensor(truth_a)[:, None]
        * torch.sin(
            math.pi
            * torch.tensor(truth_k)[:, None]
            * op.x[None, :]
        ),
        dim=0,
        keepdim=True,
    )
    f = op.apply(u_true).detach()
    return {
        "u": u_true,
        "f": f,
        "truth_k": truth_k,
        "truth_a": truth_a,
    }


def solution_metrics(op, u, problem):
    residual = op.residual(u, problem["f"])
    rse = torch.linalg.vector_norm(u - problem["u"]) / (
        torch.linalg.vector_norm(problem["u"]) + 1.0e-30
    )
    relative_residual = torch.sqrt(
        torch.mean(residual.square())
        / (torch.mean(problem["f"].square()) + 1.0e-30)
    )
    return {
        "rse": float(rse),
        "relative_residual": float(relative_residual),
        "pinn": float(torch.mean(residual.square())),
    }


def recurrent_pinn_loss(op, u_previous, correction, problem, config):
    """Grade-local residual contraction loss.

    A shared recurrent model can receive gradients through its early,
    large-residual steps.  An independently trained late grade cannot: using
    ||f|| as the denominator makes its objective and gradients vanish.  Each
    frozen grade therefore minimizes its own residual contraction factor,
    while the unnormalized PINN loss is still recorded separately.
    """
    u_next = u_previous + correction
    residual_before = op.residual(u_previous, problem["f"])
    residual_after = op.residual(u_next, problem["f"])
    residual_before_energy = (
        torch.mean(residual_before.square()).detach() + 1.0e-30
    )
    pde = torch.mean(residual_after.square()) / residual_before_energy

    residual_before_hat = dst1_ortho(residual_before)
    residual_after_hat = dst1_ortho(residual_after)
    normalized_mode = op.k / float(op.n)
    spectral_weight = (
        1.0
        + config.spectral_high_frequency_boost
        * normalized_mode.square()
    )
    spectral_before_energy = (
        torch.mean(
            residual_before_hat.square()
            * spectral_weight.unsqueeze(0)
        ).detach()
        + 1.0e-30
    )
    spectral_pde = torch.mean(
        residual_after_hat.square() * spectral_weight.unsqueeze(0)
    ) / spectral_before_energy

    correction_hat = dst1_ortho(correction)
    h1_weight = 1.0 + (math.pi * op.k) ** 2
    h1_weight = h1_weight / h1_weight.mean()
    h1_step = torch.mean(
        correction_hat.square() * h1_weight.unsqueeze(0)
    )
    null_penalty = op.null_coefficient(u_next).square().mean()
    monotone = torch.relu(pde - 1.0)
    total = (
        pde
        + config.spectral_loss_weight * spectral_pde
        + config.h1_step_weight * h1_step
        + config.null_loss_weight * null_penalty
        + config.monotone_loss_weight * monotone
    )
    return total, {
        "pde": pde.detach(),
        "spectral_pde": spectral_pde.detach(),
        "h1_step": h1_step.detach(),
        "null": null_penalty.detach(),
        "monotone": monotone.detach(),
        "u_next": u_next,
        "residual_after": residual_after,
    }


@torch.no_grad()
def apply_grade(grade, op, problem, u_previous, config, return_aux=False):
    grade.eval()
    residual = op.residual(u_previous, problem["f"])
    correction, auxiliary = grade_correction(
        grade,
        op,
        u_previous,
        residual,
        problem["f"],
        config.gate_floor_ratio,
    )
    u_next = u_previous + correction
    if return_aux:
        return u_next, auxiliary
    return u_next


def freeze_grade(grade):
    grade.eval()
    grade.requires_grad_(False)
    if any(parameter.requires_grad for parameter in grade.parameters()):
        raise RuntimeError("Grade freezing failed.")


def assert_frozen(grades):
    for grade_id, grade in enumerate(grades, start=1):
        if grade.training or any(
            parameter.requires_grad for parameter in grade.parameters()
        ):
            raise RuntimeError(f"Grade {grade_id} is not strictly frozen.")


def train_one_grade(
    grade_id, grade, op, problem, u_previous, config
):
    optimizer = torch.optim.AdamW(
        grade.parameters(),
        lr=config.learning_rate * config.warmup_start_factor,
        weight_decay=config.weight_decay,
    )
    curve = {key: [] for key in SCALAR_CURVE_KEYS}
    curve.update(
        {
            "u_after": [],
            "residual_after": [],
            "abs_error": [],
            "u_before_grade": u_previous.detach().cpu().numpy(),
        }
    )
    best_loss = float("inf")
    best_pinn = float("inf")
    significant_best_loss = float("inf")
    significant_best_pinn = float("inf")
    last_significant_loss_epoch = 0
    last_significant_pinn_epoch = 0
    lr_reductions = 0
    best_state = None
    start_time = time.perf_counter()

    residual_input = op.residual(
        u_previous, problem["f"]
    ).detach()
    for epoch in range(config.max_epochs_per_grade):
        if epoch < config.warmup_epochs:
            denominator = max(config.warmup_epochs - 1, 1)
            warmup_progress = epoch / denominator
            warmup_factor = (
                config.warmup_start_factor
                + (1.0 - config.warmup_start_factor)
                * warmup_progress
            )
            optimizer.param_groups[0]["lr"] = (
                config.learning_rate * warmup_factor
            )
        lr_used = optimizer.param_groups[0]["lr"]
        grade.train()
        optimizer.zero_grad()
        correction, _ = grade_correction(
            grade,
            op,
            u_previous.detach(),
            residual_input,
            problem["f"],
            config.gate_floor_ratio,
        )
        loss, components = recurrent_pinn_loss(
            op, u_previous, correction, problem, config
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(grade.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            correction_epoch, _ = grade_correction(
                grade,
                op,
                u_previous.detach(),
                residual_input,
                problem["f"],
                config.gate_floor_ratio,
            )
            evaluated_loss, evaluated_components = recurrent_pinn_loss(
                op,
                u_previous,
                correction_epoch,
                problem,
                config,
            )
            value = float(evaluated_loss)
            u_epoch = evaluated_components["u_next"].detach()
            residual_epoch = evaluated_components[
                "residual_after"
            ].detach()
            metrics = solution_metrics(op, u_epoch, problem)
            best_loss = min(best_loss, value)
            if metrics["pinn"] < best_pinn:
                best_pinn = metrics["pinn"]
                best_state = copy.deepcopy(grade.state_dict())
            if (
                not math.isfinite(significant_best_loss)
                or value
                < significant_best_loss
                * (1.0 - config.early_stop_relative_tolerance)
            ):
                significant_best_loss = value
                last_significant_loss_epoch = epoch
            if (
                not math.isfinite(significant_best_pinn)
                or metrics["pinn"]
                < significant_best_pinn
                * (1.0 - config.early_stop_relative_tolerance)
            ):
                significant_best_pinn = metrics["pinn"]
                last_significant_pinn_epoch = epoch
            training_early_stop_wait = (
                epoch - last_significant_loss_epoch
            )
            pinn_early_stop_wait = (
                epoch - last_significant_pinn_epoch
            )
            # Stop only when neither criterion has improved recently.
            early_stop_wait = min(
                training_early_stop_wait,
                pinn_early_stop_wait,
            )
            if (
                epoch + 1 >= config.warmup_epochs
                and training_early_stop_wait
                >= config.lr_plateau_patience
                and pinn_early_stop_wait
                >= config.lr_plateau_patience
                and optimizer.param_groups[0]["lr"]
                > config.min_learning_rate
            ):
                old_lr = optimizer.param_groups[0]["lr"]
                new_lr = max(
                    old_lr * config.lr_reduce_factor,
                    config.min_learning_rate,
                )
                optimizer.param_groups[0]["lr"] = new_lr
                lr_reductions += 1
                significant_best_loss = value
                significant_best_pinn = metrics["pinn"]
                last_significant_loss_epoch = epoch
                last_significant_pinn_epoch = epoch
                training_early_stop_wait = 0
                pinn_early_stop_wait = 0
                early_stop_wait = 0
                print(
                    f"  dual-plateau LR: {old_lr:.3e} -> "
                    f"{new_lr:.3e} at epoch={epoch}"
                )
            next_lr = optimizer.param_groups[0]["lr"]
            curve["loss"].append(value)
            curve["pde"].append(float(evaluated_components["pde"]))
            curve["spectral_pde"].append(
                float(evaluated_components["spectral_pde"])
            )
            curve["h1_step"].append(
                float(evaluated_components["h1_step"])
            )
            curve["null"].append(float(evaluated_components["null"]))
            curve["monotone"].append(
                float(evaluated_components["monotone"])
            )
            curve["rse"].append(metrics["rse"])
            curve["relative_residual"].append(
                metrics["relative_residual"]
            )
            curve["pinn"].append(metrics["pinn"])
            curve["lr"].append(lr_used)
            curve["next_lr"].append(next_lr)
            curve["lr_reductions"].append(lr_reductions)
            curve["elapsed"].append(time.perf_counter() - start_time)
            curve["best_loss"].append(best_loss)
            curve["best_pinn"].append(best_pinn)
            curve["training_early_stop_wait"].append(
                training_early_stop_wait
            )
            curve["pinn_early_stop_wait"].append(
                pinn_early_stop_wait
            )
            curve["early_stop_wait"].append(early_stop_wait)
            curve["u_after"].append(u_epoch.squeeze(0).cpu().numpy())
            curve["residual_after"].append(
                residual_epoch.squeeze(0).cpu().numpy()
            )
            curve["abs_error"].append(
                torch.abs(problem["u"] - u_epoch)
                .squeeze(0)
                .cpu()
                .numpy()
            )
        if (
            epoch % 50 == 0
            or epoch == config.max_epochs_per_grade - 1
        ):
            print(
                f"  grade={grade_id} epoch={epoch:4d} "
                f"loss={value:.3e} PINN={metrics['pinn']:.3e} "
                f"RSE(eval)={metrics['rse']:.3e} "
                f"lr={lr_used:.2e} "
                f"wait(train/PINN)="
                f"{training_early_stop_wait}/"
                f"{pinn_early_stop_wait}"
            )
        if (
            epoch + 1 >= config.min_epochs_per_grade
            and optimizer.param_groups[0]["lr"]
            <= config.min_learning_rate
            and early_stop_wait >= config.early_stop_patience
        ):
            print(
                f"  adaptive stop Grade {grade_id} at epoch={epoch}: "
                "neither Training nor PINN loss achieved >= "
                f"{config.early_stop_relative_tolerance:.1e} relative "
                f"improvement for {config.early_stop_patience} epochs"
            )
            break

    grade.load_state_dict(best_state)
    freeze_grade(grade)
    return best_loss, curve


@torch.no_grad()
def make_plot_record(
    op, problem, u_before, u_after, grade, grade_id, curve, config
):
    residual_before = op.residual(u_before, problem["f"])
    residual_after = op.residual(u_after, problem["f"])
    _, auxiliary = apply_grade(
        grade, op, problem, u_before, config, return_aux=True
    )
    preconditioned_hat = dst1_ortho(
        auxiliary["preconditioned"]
    ).squeeze(0)
    selected_index = torch.nonzero(
        auxiliary["gate"].squeeze(0) > 0.5, as_tuple=False
    ).squeeze(-1)
    selected_coefficients = torch.zeros(
        len(selected_index), op.n, dtype=preconditioned_hat.dtype
    )
    rows = torch.arange(len(selected_index))
    selected_coefficients[rows, selected_index] = preconditioned_hat[
        selected_index
    ]
    p_scale = torch.sqrt(
        torch.mean(auxiliary["preconditioned"].square()) + 1.0e-30
    )
    selected_components = (
        idst1_ortho(selected_coefficients) / p_scale
    )
    selected_amplitude = preconditioned_hat[selected_index].abs()
    alpha = selected_amplitude.square() / (
        selected_amplitude.square().sum() + 1.0e-30
    )
    log_amplitude = torch.log(selected_amplitude + 1.0e-30)
    log_norm = (log_amplitude - log_amplitude.min()) / (
        log_amplitude.max() - log_amplitude.min() + 1.0e-30
    )
    entropy = -torch.sum(alpha * torch.log(alpha + 1.0e-30)) / math.log(
        max(len(alpha), 2)
    )
    correction = u_after - u_before
    target = problem["u"] - u_before
    before_metrics = solution_metrics(op, u_before, problem)
    after_metrics = solution_metrics(op, u_after, problem)
    return {
        "grade": grade_id,
        "x": op.x.cpu().numpy(),
        "truth": problem["u"].squeeze(0).cpu().numpy(),
        "before": u_before.squeeze(0).cpu().numpy(),
        "after": u_after.squeeze(0).cpu().numpy(),
        "correction": correction.squeeze(0).cpu().numpy(),
        "target_correction": target.squeeze(0).cpu().numpy(),
        "correction_hat": dst1_ortho(correction).squeeze(0).cpu().numpy(),
        "target_correction_hat": dst1_ortho(target).squeeze(0).cpu().numpy(),
        "preconditioned_hat": preconditioned_hat.cpu().numpy(),
        "residual_before_hat": dst1_ortho(
            residual_before
        ).squeeze(0).cpu().numpy(),
        "residual_after_hat": dst1_ortho(
            residual_after
        ).squeeze(0).cpu().numpy(),
        "selected_k": (selected_index + 1).cpu().numpy(),
        "selected_components": selected_components.cpu().numpy(),
        "selected_alpha": alpha.cpu().numpy(),
        "selected_log_norm": log_norm.cpu().numpy(),
        "selection_entropy": float(entropy),
        "threshold": float(auxiliary["threshold"].squeeze()),
        "eta": float(auxiliary["eta"]),
        "rse_before": before_metrics["rse"],
        "rse_after": after_metrics["rse"],
        "relative_residual": after_metrics["relative_residual"],
        "pinn": after_metrics["pinn"],
        "curve": curve,
    }


@torch.no_grad()
def save_grade_data(
    out_dir,
    op,
    problem,
    u_before,
    u_after,
    grade,
    grade_id,
    curve,
    config,
):
    residual_before = op.residual(u_before, problem["f"])
    residual_after = op.residual(u_after, problem["f"])
    _, auxiliary = apply_grade(
        grade, op, problem, u_before, config, return_aux=True
    )
    arrays = {
        "x": op.x.cpu().numpy(),
        "truth": problem["u"].squeeze(0).cpu().numpy(),
        "forcing": problem["f"].squeeze(0).cpu().numpy(),
        "u_before": u_before.squeeze(0).cpu().numpy(),
        "u_after": u_after.squeeze(0).cpu().numpy(),
        "correction": (u_after - u_before).squeeze(0).cpu().numpy(),
        "residual_before": residual_before.squeeze(0).cpu().numpy(),
        "residual_after": residual_after.squeeze(0).cpu().numpy(),
        "selected_residual_input": auxiliary[
            "selected_residual"
        ].squeeze(0).cpu().numpy(),
        "selected_preconditioned_input": auxiliary[
            "selected_preconditioned"
        ].squeeze(0).cpu().numpy(),
        "adaptive_gate": auxiliary["gate"].squeeze(0).cpu().numpy(),
    }
    grade_npz = out_dir / f"grade_{grade_id:02d}_all_points.npz"
    np.savez_compressed(grade_npz, **arrays)

    columns = np.column_stack(
        [
            np.full(op.n, grade_id),
            arrays["x"],
            arrays["truth"],
            arrays["forcing"],
            arrays["u_before"],
            arrays["u_after"],
            arrays["correction"],
            arrays["residual_before"],
            arrays["residual_after"],
            arrays["selected_residual_input"],
            arrays["selected_preconditioned_input"],
        ]
    )
    grade_csv = out_dir / f"grade_{grade_id:02d}_all_points.csv"
    np.savetxt(
        grade_csv,
        columns,
        delimiter=",",
        header=(
            "grade,x,truth,forcing,u_before,u_after,correction,"
            "residual_before,residual_after,selected_residual_input,"
            "selected_preconditioned_input"
        ),
        comments="",
        fmt=["%d"] + ["%.17e"] * 10,
    )

    epoch = np.arange(1, len(curve["loss"]) + 1)
    metrics_csv = out_dir / f"grade_{grade_id:02d}_epoch_metrics.csv"
    np.savetxt(
        metrics_csv,
        np.column_stack(
            [epoch]
            + [np.asarray(curve[key]) for key in SCALAR_CURVE_KEYS]
        ),
        delimiter=",",
        header="epoch," + ",".join(SCALAR_CURVE_KEYS),
        comments="",
        fmt=["%d"] + ["%.17e"] * len(SCALAR_CURVE_KEYS),
    )

    epoch_npz = out_dir / f"grade_{grade_id:02d}_every_epoch_points.npz"
    np.savez_compressed(
        epoch_npz,
        epoch=epoch,
        x=arrays["x"],
        truth=arrays["truth"],
        forcing=arrays["forcing"],
        u_before_grade=arrays["u_before"],
        adaptive_gate=arrays["adaptive_gate"],
        selected_residual_input=arrays["selected_residual_input"],
        selected_preconditioned_input=arrays[
            "selected_preconditioned_input"
        ],
        u_after=np.stack(curve["u_after"]),
        residual_after=np.stack(curve["residual_after"]),
        abs_error=np.stack(curve["abs_error"]),
        **{key: np.asarray(curve[key]) for key in SCALAR_CURVE_KEYS},
    )
    return grade_npz, grade_csv, metrics_csv, epoch_npz
