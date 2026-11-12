import copy
import math
import time

import numpy as np
import torch

from models import (
    dst1_ortho,
    grade_correction,
    idst1_ortho,
)


SCALAR_CURVE_KEYS = (
    "train",
    "train_h1",
    "train_pde",
    "val",
    "val_h1",
    "val_pde",
    "val_mean_rse",
    "val_max_rse",
    "val_mean_rel_res",
    "lr",
    "elapsed",
    "best_val",
)


def make_dataset(size, op, seed, config):
    """Generate a stratified operator dataset with genuine resonant excitation."""
    rng = np.random.default_rng(seed)
    kappa_index = rng.uniform(170.0, 230.0, size=size)
    n_near = max(1, int(round(config.near_resonance_fraction * size)))
    near = np.arange(n_near)
    kappa_index[near] = (
        rng.integers(180, 221, size=len(near))
        + rng.choice([-1.0, 1.0], size=len(near))
        * rng.choice(config.resonance_offsets, size=len(near))
    )
    kappa = torch.tensor(
        kappa_index * math.pi, dtype=op.x.dtype, device=op.device
    )

    coefficients = np.zeros((size, op.n), dtype=np.float64)
    resonant_mode = np.rint(kappa_index).astype(np.int64)
    near_mask = np.zeros(size, dtype=bool)
    near_mask[near] = True
    for sample in range(size):
        n_active = int(rng.integers(8, 20))
        modes = rng.choice(
            np.arange(1, op.n + 1), size=n_active, replace=False
        )
        amplitudes = rng.normal(size=n_active)
        amplitudes *= 10.0 ** rng.uniform(-3.0, 0.0, size=n_active)
        coefficients[sample, modes - 1] = amplitudes
        if near_mask[sample]:
            # kappa near an integer is not enough: the matching eigenmode must
            # be present in the solution spectrum for this to be a resonant
            # training example.  Give it a controlled, non-negligible signal.
            mode = resonant_mode[sample]
            sign = rng.choice([-1.0, 1.0])
            magnitude = 10.0 ** rng.uniform(
                math.log10(config.resonant_amplitude_floor), 0.0
            )
            coefficients[sample, mode - 1] = sign * magnitude
    coefficients = torch.tensor(
        coefficients, dtype=op.x.dtype, device=op.device
    )
    u = idst1_ortho(coefficients)
    f = op.apply(u, kappa)
    return {
        "u": u,
        "f": f,
        "kappa": kappa,
        "coeff": coefficients,
        "near_resonance": torch.tensor(near_mask, device=op.device),
        "resonant_mode": torch.tensor(resonant_mode, device=op.device),
    }


def grade_loss(op, correction, u_previous, u_true, f, kappa, config):
    u_next = u_previous + correction
    target_correction = u_true - u_previous
    difference_hat = dst1_ortho(correction - target_correction)
    target_hat = dst1_ortho(target_correction)
    truth_hat = dst1_ortho(u_true)
    h1_weight = 1.0 + (math.pi * op.k[None, :]) ** 2
    target_energy = torch.sum(h1_weight * target_hat.square(), dim=1)
    truth_energy = torch.sum(h1_weight * truth_hat.square(), dim=1)
    h1_per_sample = torch.sum(h1_weight * difference_hat.square(), dim=1) / (
        target_energy
        + config.correction_floor_ratio * truth_energy
        + 1.0e-20
    )

    residual = op.residual(u_next, f, kappa)
    residual_previous = op.residual(u_previous, f, kappa).detach()
    forcing_rms_square = torch.mean(f.square(), dim=1) + 1.0e-20
    residual_previous_rms_square = torch.mean(
        residual_previous.square(), dim=1
    )
    pde_per_sample = torch.mean(residual.square(), dim=1) / (
        residual_previous_rms_square
        + config.residual_floor_ratio * forcing_rms_square
        + 1.0e-20
    )
    # kappa/pi is the continuous modal index.  This weight smoothly focuses
    # training on the poorly conditioned, near-integer part of the family.
    distance_to_resonance = (kappa / math.pi - torch.round(kappa / math.pi)).abs()
    resonance_weight = 1.0 + config.resonance_loss_weight * torch.exp(
        -distance_to_resonance / config.resonance_width
    )
    resonance_weight = resonance_weight / resonance_weight.mean()
    h1 = torch.mean(resonance_weight * h1_per_sample)
    pde = torch.mean(resonance_weight * pde_per_sample)

    null_mask = op.eigenvalues(kappa).abs() < 10.0
    u_hat = dst1_ortho(u_next)
    null_penalty = torch.mean(
        torch.sum(u_hat.square() * null_mask, dim=1)
    )
    total = (
        config.h1_loss_weight * h1
        + config.pde_loss_weight * pde
        + config.null_loss_weight * null_penalty
    )
    return total, h1.detach(), pde.detach(), null_penalty.detach()


@torch.no_grad()
def apply_grade(grade, op, data, u_previous, config, return_aux=False):
    grade.eval()
    residual = op.residual(
        u_previous, data["f"], data["kappa"]
    )
    correction, auxiliary = grade_correction(
        grade,
        op,
        u_previous,
        residual,
        data["f"],
        data["kappa"],
        config.gate_floor_ratio,
        config.gate_soft_floor,
        config.gate_temperature_ratio,
    )
    result = u_previous + correction
    if return_aux:
        return result, auxiliary
    return result


@torch.no_grad()
def metrics(op, data, u):
    residual = op.residual(u, data["f"], data["kappa"])
    rse = torch.linalg.vector_norm(u - data["u"], dim=1) / (
        torch.linalg.vector_norm(data["u"], dim=1) + 1.0e-20
    )
    relative_residual = torch.sqrt(
        torch.mean(residual.square(), dim=1)
        / (torch.mean(data["f"].square(), dim=1) + 1.0e-20)
    )
    return {
        "mean_rse": float(rse.mean()),
        "max_rse": float(rse.max()),
        "mean_rel_res": float(relative_residual.mean()),
        "rse": rse.cpu().numpy(),
        "relative_residual": relative_residual.cpu().numpy(),
    }


@torch.no_grad()
def validation_loss(grade, op, data, u_previous, config):
    grade.eval()
    residual = op.residual(u_previous, data["f"], data["kappa"])
    correction, _ = grade_correction(
        grade,
        op,
        u_previous,
        residual,
        data["f"],
        data["kappa"],
        config.gate_floor_ratio,
        config.gate_soft_floor,
        config.gate_temperature_ratio,
    )
    loss, h1, pde, null = grade_loss(
        op,
        correction,
        u_previous,
        data["u"],
        data["f"],
        data["kappa"],
        config,
    )
    return (
        float(loss),
        float(h1),
        float(pde),
        float(null),
        u_previous + correction,
    )


def train_one_grade(
    grade_id, grade, op, train, u_train, validation, u_validation, config
):
    """Train only the current grade on residuals left by frozen earlier grades."""
    optimizer = torch.optim.AdamW(
        grade.parameters(),
        lr=max(
            config.min_learning_rate,
            config.learning_rate
            * config.grade_learning_rate_decay ** (grade_id - 1),
        ),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    curve = {
        key: [] for key in SCALAR_CURVE_KEYS
    }
    curve.update(
        {
            "val_u_before": u_validation.detach().cpu().numpy(),
            "val_u_after": [],
            "val_residual_after": [],
            "val_abs_error": [],
        }
    )

    n_train = train["u"].shape[0]
    best_validation = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(config.epochs_per_grade):
        grade.train()
        order = torch.randperm(n_train, device=op.device)
        total_sum = 0.0
        h1_sum = 0.0
        pde_sum = 0.0

        for start in range(0, n_train, config.batch_size):
            indices = order[start : start + config.batch_size]
            u0 = u_train[indices].detach()
            f = train["f"][indices]
            kappa = train["kappa"][indices]
            residual = op.residual(u0, f, kappa).detach()

            optimizer.zero_grad()
            correction, _ = grade_correction(
                grade,
                op,
                u0,
                residual,
                f,
                kappa,
                config.gate_floor_ratio,
                config.gate_soft_floor,
                config.gate_temperature_ratio,
            )
            loss, h1, pde, _ = grade_loss(
                op,
                correction,
                u0,
                train["u"][indices],
                f,
                kappa,
                config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grade.parameters(), 1.0)
            optimizer.step()

            batch_size = len(indices)
            total_sum += float(loss.detach()) * batch_size
            h1_sum += float(h1) * batch_size
            pde_sum += float(pde) * batch_size

        val_total, val_h1, val_pde, _, val_candidate = validation_loss(
            grade, op, validation, u_validation, config
        )
        val_metrics = metrics(op, validation, val_candidate)
        # Close to resonance, a small PDE residual can coexist with a large
        # solution error because the operator has a small eigenvalue.  Select
        # checkpoints by supervised solution RSE, not residual alone.
        selection_metric = val_metrics["mean_rse"]
        current_best = min(best_validation, selection_metric)

        curve["train"].append(total_sum / n_train)
        curve["train_h1"].append(h1_sum / n_train)
        curve["train_pde"].append(pde_sum / n_train)
        curve["val"].append(val_total)
        curve["val_h1"].append(val_h1)
        curve["val_pde"].append(val_pde)
        curve["val_mean_rse"].append(val_metrics["mean_rse"])
        curve["val_max_rse"].append(val_metrics["max_rse"])
        curve["val_mean_rel_res"].append(val_metrics["mean_rel_res"])
        curve["lr"].append(optimizer.param_groups[0]["lr"])
        curve["elapsed"].append(time.perf_counter() - start_time)
        curve["best_val"].append(current_best)
        val_residual = op.residual(
            val_candidate, validation["f"], validation["kappa"]
        )
        curve["val_u_after"].append(val_candidate.cpu().numpy())
        curve["val_residual_after"].append(val_residual.cpu().numpy())
        curve["val_abs_error"].append(
            torch.abs(validation["u"] - val_candidate).cpu().numpy()
        )

        scheduler.step(selection_metric)
        relative_margin = config.early_stopping_relative_delta * max(
            best_validation, 1.0e-12
        )
        if best_state is None or selection_metric < best_validation - relative_margin:
            best_validation = selection_metric
            best_state = copy.deepcopy(grade.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == config.epochs_per_grade - 1:
            print(
                f"  grade={grade_id} epoch={epoch + 1:3d} "
                f"train={total_sum / n_train:.3e} "
                f"validation={val_total:.3e}"
            )

        if (
            epoch + 1 >= config.early_stopping_min_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"  grade={grade_id} early-stop epoch={epoch + 1}; "
                f"best RSE epoch={best_epoch}"
            )
            break

    grade.load_state_dict(best_state)
    calibration_scale, calibrated_residual = calibrate_grade_step(
        grade, op, validation, u_validation, config
    )
    print(
        f"  grade={grade_id} calibrated step scale={calibration_scale:.3f}, "
        f"validation RSE={calibrated_residual:.3e}"
    )
    freeze_grade(grade)
    return best_validation, curve


@torch.no_grad()
def calibrate_grade_step(grade, op, validation, u_validation, config):
    """Choose a validation-safe global scale for the learned Grade update."""
    original_eta = torch.sigmoid(grade.step_logit.detach())
    best_scale = 0.0
    best_score = metrics(op, validation, u_validation)["mean_rse"]
    best_logit = grade.step_logit.detach().clone()

    for scale in config.step_calibration_scales:
        if scale == 0.0:
            continue
        eta = (original_eta * scale).clamp(1.0e-6, 1.0 - 1.0e-6)
        grade.step_logit.copy_(torch.logit(eta))
        candidate = apply_grade(grade, op, validation, u_validation, config)
        score = metrics(op, validation, candidate)["mean_rse"]
        if score < best_score:
            best_scale = float(scale)
            best_score = score
            best_logit = grade.step_logit.detach().clone()

    grade.step_logit.copy_(best_logit)
    return best_scale, best_score


def freeze_grade(grade):
    grade.eval()
    grade.requires_grad_(False)
    if any(parameter.requires_grad for parameter in grade.parameters()):
        raise RuntimeError("A previous grade was not frozen.")


def assert_previous_grades_frozen(grades):
    for index, grade in enumerate(grades, start=1):
        if grade.training:
            raise RuntimeError(f"Grade {index} is not in eval mode.")
        if any(parameter.requires_grad for parameter in grade.parameters()):
            raise RuntimeError(f"Grade {index} still has trainable parameters.")


@torch.no_grad()
def make_plot_record(
    op, data, u_before, u_after, grade, grade_id, curve, config,
    split="test", status="accepted",
):
    residual_before = op.residual(
        u_before, data["f"], data["kappa"]
    )
    residual_after = op.residual(
        u_after, data["f"], data["kappa"]
    )
    _, auxiliary = apply_grade(
        grade, op, data, u_before, config, return_aux=True
    )
    test_metrics = metrics(op, data, u_after)
    before_metrics = metrics(op, data, u_before)
    distance = (
        data["kappa"] / math.pi - torch.round(data["kappa"] / math.pi)
    ).abs()
    near_indices = torch.nonzero(
        distance <= config.resonance_width, as_tuple=False
    ).squeeze(-1)
    candidates = near_indices if len(near_indices) else torch.arange(
        data["u"].shape[0], device=op.device
    )
    candidate_rse = torch.as_tensor(
        test_metrics["rse"], dtype=data["u"].dtype, device=op.device
    )[candidates]
    median_rse = candidate_rse.median()
    representative = int(
        candidates[torch.argmin(torch.abs(candidate_rse - median_rse))]
    )
    i = representative
    kappa_over_pi = data["kappa"] / math.pi
    preconditioned_hat = dst1_ortho(
        auxiliary["preconditioned"][i : i + 1]
    ).squeeze(0)
    selected_k_tensor = torch.nonzero(
        auxiliary["gate"][i] > 0.5, as_tuple=False
    ).squeeze(-1)
    selected_coefficients = torch.zeros(
        len(selected_k_tensor),
        op.n,
        dtype=preconditioned_hat.dtype,
        device=preconditioned_hat.device,
    )
    if len(selected_k_tensor):
        rows = torch.arange(
            len(selected_k_tensor), device=preconditioned_hat.device
        )
        selected_coefficients[rows, selected_k_tensor] = (
            preconditioned_hat[selected_k_tensor]
        )
        selected_components = idst1_ortho(selected_coefficients)
        p_scale = torch.sqrt(
            torch.mean(
                auxiliary["preconditioned"][i].square()
            )
            + 1.0e-30
        )
        selected_components = selected_components / p_scale
    else:
        selected_components = torch.zeros(
            0, op.n, dtype=preconditioned_hat.dtype
        )
    selected_amplitude = preconditioned_hat[
        selected_k_tensor
    ].abs()
    alpha = selected_amplitude.square() / (
        selected_amplitude.square().sum() + 1.0e-30
    )
    if len(selected_amplitude) > 1:
        log_amplitude = torch.log(selected_amplitude + 1.0e-30)
        log_norm = (log_amplitude - log_amplitude.min()) / (
            log_amplitude.max() - log_amplitude.min() + 1.0e-30
        )
        entropy = -torch.sum(alpha * torch.log(alpha + 1.0e-30)) / math.log(
            len(selected_amplitude)
        )
    else:
        log_norm = torch.ones_like(selected_amplitude)
        entropy = torch.zeros((), dtype=selected_amplitude.dtype)
    correction = u_after[i] - u_before[i]
    target_correction = data["u"][i] - u_before[i]
    return {
        "grade": grade_id,
        "split": split,
        "status": status,
        "representative_id": representative,
        "representative_distance": float(distance[representative]),
        "x": op.x.cpu().numpy(),
        "truth": data["u"][i].cpu().numpy(),
        "before": u_before[i].cpu().numpy(),
        "after": u_after[i].cpu().numpy(),
        "truth_hat": dst1_ortho(
            data["u"][i : i + 1]
        ).squeeze(0).cpu().numpy(),
        "after_hat": dst1_ortho(
            u_after[i : i + 1]
        ).squeeze(0).cpu().numpy(),
        "res_before_hat": dst1_ortho(
            residual_before[i : i + 1]
        ).squeeze(0).cpu().numpy(),
        "res_after_hat": dst1_ortho(
            residual_after[i : i + 1]
        ).squeeze(0).cpu().numpy(),
        "precond_hat": preconditioned_hat.cpu().numpy(),
        "gate": auxiliary["gate"][i].cpu().numpy(),
        "gate_threshold": float(
            auxiliary["gate_threshold"][i].squeeze()
        ),
        "selected_k": (selected_k_tensor + 1).cpu().numpy(),
        "selected_components": selected_components.cpu().numpy(),
        "selected_amplitude": selected_amplitude.cpu().numpy(),
        "selected_alpha": alpha.cpu().numpy(),
        "selected_log_norm": log_norm.cpu().numpy(),
        "selection_entropy": float(entropy),
        "correction": correction.cpu().numpy(),
        "target_correction": target_correction.cpu().numpy(),
        "correction_hat": dst1_ortho(
            correction[None, :]
        ).squeeze(0).cpu().numpy(),
        "target_correction_hat": dst1_ortho(
            target_correction[None, :]
        ).squeeze(0).cpu().numpy(),
        "relative_error_before": float(
            torch.linalg.vector_norm(u_before[i] - data["u"][i])
            / (torch.linalg.vector_norm(data["u"][i]) + 1.0e-20)
        ),
        "relative_error_after": float(test_metrics["rse"][i]),
        "pinn_loss": float(torch.mean(residual_after[i].square())),
        "eta": float(auxiliary["eta"]),
        "test_rse": test_metrics["rse"],
        "test_rse_before": before_metrics["rse"],
        "mean_rse": test_metrics["mean_rse"],
        "max_rse": test_metrics["max_rse"],
        "mean_rel_res": test_metrics["mean_rel_res"],
        "kappa_over_pi": kappa_over_pi.cpu().numpy(),
        "distance_to_resonance": distance.cpu().numpy(),
        "curve": curve,
    }


@torch.no_grad()
def save_grade_data(
    out_dir,
    op,
    test,
    u_before,
    u_after,
    grade,
    grade_id,
    curve,
    validation,
    config,
):
    residual_before = op.residual(
        u_before, test["f"], test["kappa"]
    )
    residual_after = op.residual(
        u_after, test["f"], test["kappa"]
    )
    _, auxiliary = apply_grade(
        grade, op, test, u_before, config, return_aux=True
    )
    arrays = {
        "x": op.x.cpu().numpy(),
        "kappa": test["kappa"].cpu().numpy(),
        "truth": test["u"].cpu().numpy(),
        "forcing": test["f"].cpu().numpy(),
        "u_before": u_before.cpu().numpy(),
        "u_after": u_after.cpu().numpy(),
        "correction": (u_after - u_before).cpu().numpy(),
        "residual_before": residual_before.cpu().numpy(),
        "residual_after": residual_after.cpu().numpy(),
        "preconditioned_residual": auxiliary[
            "preconditioned"
        ].cpu().numpy(),
        "selected_residual_input": auxiliary[
            "selected_residual"
        ].cpu().numpy(),
        "selected_preconditioned_input": auxiliary[
            "selected_preconditioned"
        ].cpu().numpy(),
        "error_before": (test["u"] - u_before).cpu().numpy(),
        "error_after": (test["u"] - u_after).cpu().numpy(),
        "truth_hat": dst1_ortho(test["u"]).cpu().numpy(),
        "u_before_hat": dst1_ortho(u_before).cpu().numpy(),
        "u_after_hat": dst1_ortho(u_after).cpu().numpy(),
        "residual_before_hat": dst1_ortho(
            residual_before
        ).cpu().numpy(),
        "residual_after_hat": dst1_ortho(
            residual_after
        ).cpu().numpy(),
        "preconditioned_hat": dst1_ortho(
            auxiliary["preconditioned"]
        ).cpu().numpy(),
        "adaptive_gate": auxiliary["gate"].cpu().numpy(),
        "adaptive_gate_threshold": auxiliary[
            "gate_threshold"
        ].cpu().numpy(),
    }
    grade_npz = out_dir / f"grade_{grade_id:02d}_all_points.npz"
    np.savez_compressed(grade_npz, **arrays)

    batch, n = arrays["truth"].shape
    columns = np.column_stack(
        [
            np.full(batch * n, grade_id),
            np.repeat(np.arange(batch), n),
            np.repeat(arrays["kappa"], n),
            np.tile(arrays["x"], batch),
            arrays["truth"].reshape(-1),
            arrays["forcing"].reshape(-1),
            arrays["u_before"].reshape(-1),
            arrays["u_after"].reshape(-1),
            arrays["correction"].reshape(-1),
            arrays["residual_before"].reshape(-1),
            arrays["residual_after"].reshape(-1),
            arrays["preconditioned_residual"].reshape(-1),
            arrays["selected_residual_input"].reshape(-1),
            arrays["selected_preconditioned_input"].reshape(-1),
            np.abs(arrays["error_before"]).reshape(-1),
            np.abs(arrays["error_after"]).reshape(-1),
        ]
    )
    grade_csv = out_dir / f"grade_{grade_id:02d}_all_points.csv"
    np.savetxt(
        grade_csv,
        columns,
        delimiter=",",
        header=(
            "grade,sample_id,kappa,x,truth,forcing,u_before,u_after,"
            "correction,residual_before,residual_after,"
            "preconditioned_residual,selected_residual_input,"
            "selected_preconditioned_input,abs_error_before,abs_error_after"
        ),
        comments="",
        fmt=["%d", "%d"] + ["%.17e"] * 14,
    )

    epochs = np.arange(1, len(curve["train"]) + 1)
    metrics_csv = out_dir / f"grade_{grade_id:02d}_epoch_metrics.csv"
    scalar_columns = np.column_stack(
        [epochs] + [np.asarray(curve[key]) for key in SCALAR_CURVE_KEYS]
    )
    np.savetxt(
        metrics_csv,
        scalar_columns,
        delimiter=",",
        header="epoch," + ",".join(SCALAR_CURVE_KEYS),
        comments="",
        fmt=["%d"] + ["%.17e"] * len(SCALAR_CURVE_KEYS),
    )

    epoch_npz = out_dir / f"grade_{grade_id:02d}_every_epoch_points.npz"
    validation_u_before = torch.as_tensor(
        curve["val_u_before"],
        dtype=validation["u"].dtype,
        device=validation["u"].device,
    )
    _, validation_auxiliary = apply_grade(
        grade,
        op,
        validation,
        validation_u_before,
        config,
        return_aux=True,
    )
    epoch_arrays = {
        "grade": np.asarray(grade_id),
        "epoch": epochs,
        "x": op.x.cpu().numpy(),
        "sample_id": np.arange(validation["u"].shape[0]),
        "kappa": validation["kappa"].cpu().numpy(),
        "truth": validation["u"].cpu().numpy(),
        "forcing": validation["f"].cpu().numpy(),
        "u_before_grade": curve["val_u_before"],
        "val_adaptive_gate": validation_auxiliary["gate"].cpu().numpy(),
        "val_adaptive_gate_threshold": validation_auxiliary[
            "gate_threshold"
        ].cpu().numpy(),
        "val_selected_residual_input": validation_auxiliary[
            "selected_residual"
        ].cpu().numpy(),
        "val_selected_preconditioned_input": validation_auxiliary[
            "selected_preconditioned"
        ].cpu().numpy(),
        "val_u_after": np.stack(curve["val_u_after"]),
        "val_residual_after": np.stack(curve["val_residual_after"]),
        "val_abs_error": np.stack(curve["val_abs_error"]),
    }
    epoch_arrays.update(
        {key: np.asarray(curve[key]) for key in SCALAR_CURVE_KEYS}
    )
    np.savez_compressed(epoch_npz, **epoch_arrays)
    return grade_npz, grade_csv, metrics_csv, epoch_npz
