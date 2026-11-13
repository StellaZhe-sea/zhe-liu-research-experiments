"""Academic dark-theme plots matching 1D_H_spectral_v9 notebook."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


plt.rcParams.update(
    {
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.facecolor": "#0d0d0d",
        "axes.facecolor": "#181818",
        "axes.labelcolor": "#bbb",
        "axes.titlecolor": "#fff",
        "xtick.color": "#888",
        "ytick.color": "#888",
        "grid.color": "#2a2a2a",
        "text.color": "#ccc",
        "legend.framealpha": 0.3,
        "legend.facecolor": "#181818",
    }
)
CGRADE = plt.cm.plasma


def _grade_color(grade, n_grades=3):
    return CGRADE((grade - 1) / max(n_grades - 0.5, 1.0))


def _save_dark(fig, path, dpi=300):
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_grade_diagnostics(record, out_dir):
    """Notebook-compatible 6x2 per-grade scientific diagnostic."""
    out_dir = Path(out_dir)
    grade = record["grade"]
    color = _grade_color(grade)
    x = record["x"]
    frequency = np.arange(1, len(x) + 1)
    selected_k = np.asarray(record["selected_k"], dtype=int)
    n_selected = len(selected_k)

    fig = plt.figure(figsize=(16, 24), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Recurrent-DST-FNO Operator]  Grade {grade}  ·  "
        f"|K_n|={n_selected}  ·  H={record['selection_entropy']:.3f}  ·  "
        f"η={record['eta']:.3f}  ·  unseen mean RSE={record['mean_rse']:.2e}",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=0.99,
    )
    gs = gridspec.GridSpec(6, 2, fig, hspace=0.55, wspace=0.35)

    # Row 0: spectrum that actually produces the dynamic input gate.
    ax = fig.add_subplot(gs[0, :])
    amplitude = np.abs(record["precond_hat"])
    ax.semilogy(
        frequency,
        amplitude + 1.0e-30,
        color="#aaaaaa",
        lw=0.9,
        label=r"$|\widehat{p}_{n-1}(k)|$",
    )
    ax.axhline(
        record["gate_threshold"],
        color="#ff4455",
        lw=1.5,
        ls="--",
        label=f"dynamic threshold τ={record['gate_threshold']:.2e}",
    )
    ax.scatter(
        selected_k,
        amplitude[selected_k - 1] + 1.0e-30,
        color="#4499ff",
        s=25,
        zorder=5,
        label=rf"$K_n$: {n_selected} frequencies",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Amplitude (log)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        r"Dynamic Input Spectrum $|\widehat{p}_{n-1}(k)|$ — "
        r"Residual-Driven Threshold"
    )

    # Row 1: target correction spectrum and selected-frequency weights.
    ax = fig.add_subplot(gs[1, 0])
    target_amplitude = np.abs(record["target_correction_hat"])
    ax.semilogy(
        frequency,
        target_amplitude + 1.0e-30,
        color="#888888",
        lw=0.9,
        label=r"$|\widehat{\delta}_n(k)|$",
    )
    ax.scatter(
        selected_k,
        target_amplitude[selected_k - 1] + 1.0e-30,
        color="#ff8844",
        s=20,
        zorder=5,
        label=r"$K_n$",
    )
    ax.set_title(r"Target Correction $\delta_n$ Spectrum")
    ax.set_xlabel("k")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(
        selected_k,
        record["selected_log_norm"],
        color="#44ddff",
        alpha=0.85,
        width=1.5,
        label="log-amplitude norm",
    )
    ax2 = ax.twinx()
    ax2.bar(
        selected_k,
        record["selected_alpha"],
        color=color,
        alpha=0.4,
        width=1.5,
        label=r"$\alpha_k$",
    )
    ax.set_title(r"Selected Log-Amplitude vs $\alpha_k$")
    ax.set_xlabel("k")
    ax.set_ylabel("log-norm", color="#44ddff")
    ax2.set_ylabel(r"$\alpha_k$", color=color)
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=7)

    # Row 2: physical-space dynamic frequency inputs and grade output.
    ax = fig.add_subplot(gs[2, 0])
    components = np.asarray(record["selected_components"])
    n_show = min(8, len(components))
    component_colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, max(n_show, 1)))
    for index in range(n_show):
        ax.plot(
            x,
            components[index],
            color=component_colors[index],
            lw=0.9,
            alpha=0.85,
            label=f"k={selected_k[index]}",
        )
    ax.legend(fontsize=6, ncol=2)
    ax.set_title(r"Dynamic Frequency Input $s_n(x)$")
    ax.set_xlabel("x")

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(
        x,
        record["target_correction"],
        color="#888888",
        lw=1.2,
        ls="--",
        label=r"Target $\delta_n$",
    )
    ax.plot(
        x,
        record["correction"],
        color=color,
        lw=1.5,
        label=rf"Output $G_{{\theta_{grade}}}$",
    )
    ax.legend(fontsize=8)
    ax.set_title(f"Grade {grade} Output vs Target")
    ax.set_xlabel("x")

    # Row 3: accumulated solution and every-point error.
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(x, record["truth"], color="white", lw=2, label=r"$u^*$", zorder=3)
    ax.plot(
        x,
        record["before"],
        color="#666",
        lw=1.2,
        ls="--",
        alpha=0.7,
        label=rf"$u_{{{grade-1}}}$",
    )
    ax.plot(
        x,
        record["after"],
        color=color,
        lw=1.5,
        label=rf"$u_{{{grade}}}$",
    )
    ax.legend(fontsize=8)
    ax.set_title("Cumulative Solution")
    ax.set_xlabel("x")

    ax = fig.add_subplot(gs[3, 1])
    ax.semilogy(
        x,
        np.abs(record["before"] - record["truth"]) + 1.0e-30,
        "--",
        color="#666",
        lw=1.2,
        alpha=0.7,
        label=f"before {record['relative_error_before']:.3e}",
    )
    ax.semilogy(
        x,
        np.abs(record["after"] - record["truth"]) + 1.0e-30,
        color=color,
        lw=1.5,
        label=f"after  {record['relative_error_after']:.3e}",
    )
    ax.legend(fontsize=8)
    ax.set_title(r"Pointwise Error $|u^*-u_n|$")
    ax.set_xlabel("x")

    # Row 4: all recorded epochs and selected-mode energy.
    curve = record["curve"]
    epoch = np.arange(1, len(curve["train"]) + 1)
    ax = fig.add_subplot(gs[4, 0])
    ax.semilogy(
        epoch,
        curve["train"],
        color=color,
        lw=1.2,
        marker="o",
        ms=2,
        label="Total",
    )
    ax.semilogy(
        epoch,
        curve["train_h1"],
        color="#44ff88",
        lw=0.9,
        ls="--",
        marker="o",
        ms=1.5,
        alpha=0.7,
        label=r"$H^1$",
    )
    ax.semilogy(
        epoch,
        curve["train_pde"],
        color="#ff8844",
        lw=0.9,
        ls=":",
        marker="o",
        ms=1.5,
        alpha=0.7,
        label="PDE",
    )
    ax.legend(fontsize=7)
    ax.set_title("Training Loss — every epoch recorded")
    ax.set_xlabel("Epoch")

    ax = fig.add_subplot(gs[4, 1])
    ax.bar(
        selected_k,
        record["selected_alpha"],
        color=color,
        alpha=0.85,
        width=1.5,
    )
    ax.set_title(r"$\alpha_k$ Dynamic Input Energy Weights")
    ax.set_xlabel("k")

    # Row 5: output correction spectrum vs target correction spectrum.
    ax = fig.add_subplot(gs[5, :])
    output_amplitude = np.abs(record["correction_hat"])
    ax.semilogy(
        frequency,
        target_amplitude + 1.0e-30,
        color="#aaaaaa",
        lw=1,
        ls="--",
        label=r"$|\widehat{\delta}_n(k)|$ target",
    )
    ax.semilogy(
        frequency,
        output_amplitude + 1.0e-30,
        color=color,
        lw=1,
        label=r"$|\widehat{G}_{\theta_n}(k)|$ output",
    )
    ax.scatter(
        selected_k,
        output_amplitude[selected_k - 1] + 1.0e-30,
        color="#4499ff",
        s=20,
        zorder=5,
        label=r"$K_n$",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    ax.set_title("Output Spectrum vs Target")

    fig.subplots_adjust(top=0.965)
    diagnostics_path = out_dir / f"grade_{grade:02d}_diag.png"
    _save_dark(fig, diagnostics_path)

    # Keep the dedicated every-epoch figure as an additional artifact.
    epoch_path = _plot_grade_epoch_scatter(record, out_dir)
    return diagnostics_path, epoch_path


def _plot_grade_epoch_scatter(record, out_dir):
    grade = record["grade"]
    color = _grade_color(grade)
    curve = record["curve"]
    epoch = np.arange(1, len(curve["train"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Grade {grade}] Every Epoch · one marker = one stored epoch",
        color="white",
        fontweight="bold",
    )
    axes[0, 0].scatter(epoch, curve["train"], s=18, color=color, label="train")
    axes[0, 0].scatter(epoch, curve["val"], s=18, color="#44ddff", label="validation")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Hybrid Loss")
    axes[0, 0].legend()

    for key, label, c in (
        ("train_h1", "train H1", "#44ff88"),
        ("train_pde", "train PDE", "#ff8844"),
        ("val_h1", "validation H1", "#44ddff"),
        ("val_pde", "validation PDE", "#ff4455"),
    ):
        axes[0, 1].scatter(epoch, curve[key], s=13, label=label, color=c)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].legend(ncol=2, fontsize=7)

    for key, label, c in (
        ("val_mean_rse", "mean RSE", "#44ddff"),
        ("val_max_rse", "max RSE", "#ff8844"),
        ("val_mean_rel_res", "relative residual", "#44ff88"),
    ):
        axes[1, 0].scatter(epoch, curve[key], s=15, label=label, color=c)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Validation Accuracy")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].scatter(epoch, curve["lr"], s=15, color="#44ddff")
    axes[1, 1].set_ylabel("learning rate", color="#44ddff")
    twin = axes[1, 1].twinx()
    twin.scatter(epoch, curve["elapsed"], s=15, color="#ff8844")
    twin.set_ylabel("elapsed seconds", color="#ff8844")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
    path = Path(out_dir) / f"grade_{grade:02d}_epoch_scatter.png"
    _save_dark(fig, path, dpi=250)
    return path


def plot_summary(records, out_dir):
    """Three-row NLS-style summary matching the original notebook."""
    out_dir = Path(out_dir)
    n_grades = len(records)
    colors = [_grade_color(index + 1, n_grades) for index in range(n_grades)]
    x = records[0]["x"]
    final_rse = records[-1]["mean_rse"]

    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Recurrent-DST-FNO Operator]  Summary  ·  {n_grades} Grades  ·  "
        f"Final unseen mean RSE={final_rse:.2e}",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=0.99,
    )
    outer = gridspec.GridSpec(
        3,
        1,
        fig,
        top=0.96,
        bottom=0.05,
        hspace=0.48,
        height_ratios=[1.6, 1, 1],
    )

    ax_loss = fig.add_subplot(outer[0])
    ax_loss.set_title(
        "All Grades — Continuous Training Loss "
        "(solid=Total · dashed=H1 · dotted=PDE)",
        fontsize=10,
        color="#ccc",
    )
    offset = 0
    centers = []
    boundaries = [0]
    for index, record in enumerate(records):
        curve = record["curve"]
        n_epoch = len(curve["train"])
        epochs = np.arange(offset, offset + n_epoch)
        color = colors[index]
        ax_loss.semilogy(epochs, curve["train"], color=color, lw=1.1)
        ax_loss.semilogy(
            epochs, curve["train_h1"], color=color, lw=0.8, ls="--", alpha=0.65
        )
        ax_loss.semilogy(
            epochs, curve["train_pde"], color=color, lw=0.8, ls=":", alpha=0.55
        )
        centers.append(offset + n_epoch / 2)
        offset += n_epoch
        boundaries.append(offset)
    for boundary in boundaries[1:-1]:
        ax_loss.axvline(boundary, color="#444", lw=0.8, ls="--")
    lower = ax_loss.get_ylim()[0]
    for index, center in enumerate(centers):
        ax_loss.text(
            center,
            lower,
            f"G{index + 1}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors[index],
        )
    ax_loss.set_xlim(0, boundaries[-1])
    ax_loss.set_xlabel("Global Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(
        handles=[
            Line2D([0], [0], color="white", lw=1.5, label="Total"),
            Line2D([0], [0], color="white", lw=1, ls="--", label="H1"),
            Line2D([0], [0], color="white", lw=1, ls=":", label="PDE"),
        ],
        loc="upper right",
        fontsize=8,
    )

    middle = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.38
    )
    ax = fig.add_subplot(middle[0])
    ax.plot(x, records[-1]["truth"], color="white", lw=2, label=r"$u^*$")
    ax.plot(
        x,
        records[-1]["after"],
        color="#ff66cc",
        lw=1.3,
        ls="--",
        label=rf"$u_{{approx}}$ ({final_rse:.2e})",
    )
    ax.legend(fontsize=7)
    ax.set_title("Final Solution vs True")
    ax.set_xlabel("x")

    ax = fig.add_subplot(middle[1])
    ax.plot(x, records[0]["truth"], color="white", lw=2, label=r"$u^*$")
    for index, record in enumerate(records):
        ax.plot(x, record["after"], color=colors[index], lw=1, label=f"G{index+1}")
    ax.legend(fontsize=7)
    ax.set_title("Solution Evolution")
    ax.set_xlabel("x")

    ax = fig.add_subplot(middle[2])
    for index, record in enumerate(records):
        ax.semilogy(
            x,
            np.abs(record["after"] - record["truth"]) + 1.0e-30,
            color=colors[index],
            lw=1,
            label=f"G{index+1}",
        )
    ax.legend(fontsize=7)
    ax.set_title("Pointwise Error Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|u^*-u_n|$")

    bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.38
    )
    grade_index = np.arange(1, n_grades + 1)
    rse_values = [record["mean_rse"] for record in records]
    ax = fig.add_subplot(bottom[0])
    ax.semilogy(grade_index, rse_values, "o-", color="#44ddff", lw=2, ms=7)
    for gx, value in zip(grade_index, rse_values):
        ax.annotate(
            f"{value:.1e}",
            (gx, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#44ddff",
        )
    ax.set_title("Unseen Mean RSE per Grade")
    ax.set_xlabel("Grade")
    ax.set_ylabel("RSE")
    ax.set_xticks(grade_index)

    active_counts = [len(record["selected_k"]) for record in records]
    ax = fig.add_subplot(bottom[1])
    bars = ax.bar(grade_index, active_counts, color=colors, alpha=0.88)
    for bar, value in zip(bars, active_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(value),
            ha="center",
            color="white",
        )
    ax.set_title(r"$|K_n|$ Dynamic Input Frequencies")
    ax.set_xlabel("Grade")
    ax.set_xticks(grade_index)

    before_values = [1.0] + rse_values[:-1]
    improvements = [
        (before - after) / (before + 1.0e-30) * 100
        for before, after in zip(before_values, rse_values)
    ]
    ax = fig.add_subplot(bottom[2])
    bar_colors = ["#ff4455" if value < 0 else "#00cc88" for value in improvements]
    bars = ax.bar(grade_index, improvements, color=bar_colors, alpha=0.88)
    for bar, value in zip(bars, improvements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + 0.5,
            f"{value:.1f}%",
            ha="center",
            color="white",
        )
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_title("Unseen RSE Improvement per Grade")
    ax.set_xlabel("Grade")
    ax.set_ylabel("%")
    ax.set_xticks(grade_index)

    summary_path = out_dir / "all_grades_summary.png"
    fig.savefig(
        summary_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    fig.savefig(
        out_dir / "summary_nls.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return summary_path


def _decorate_grade_boundaries(ax, curves, offsets, colors):
    lower = ax.get_ylim()[0]
    for index, curve in enumerate(curves):
        x0, x1 = offsets[index], offsets[index + 1]
        if index:
            ax.axvline(x0, color="#444", lw=0.8, ls="--")
        ax.text(
            (x0 + x1) / 2,
            lower,
            f"G{index+1}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors[index],
        )


def save_and_plot_all_epoch_metrics(curves, out_dir, scalar_keys):
    """Save every epoch and reproduce the notebook's academic metric figures."""
    out_dir = Path(out_dir)
    rows = []
    global_offset = 0
    for grade, curve in enumerate(curves, start=1):
        for local_index in range(len(curve["train"])):
            rows.append(
                [grade, local_index + 1, global_offset + local_index + 1]
                + [curve[key][local_index] for key in scalar_keys]
            )
        global_offset += len(curve["train"])
    values = np.asarray(rows, dtype=float)
    csv_path = out_dir / "all_grades_every_epoch_metrics.csv"
    np.savetxt(
        csv_path,
        values,
        delimiter=",",
        header="grade,local_epoch,global_epoch," + ",".join(scalar_keys),
        comments="",
        fmt=["%d", "%d", "%d"] + ["%.17e"] * len(scalar_keys),
    )

    colors = [_grade_color(index + 1, len(curves)) for index in range(len(curves))]
    offsets = [0]
    for curve in curves:
        offsets.append(offsets[-1] + len(curve["train"]))

    # Combined four-panel figure, with one visible marker per stored epoch.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0d0d0d")
    fig.suptitle(
        "[All Grades] Every Epoch · one marker = one stored epoch",
        color="white",
        fontweight="bold",
    )
    for index, curve in enumerate(curves):
        epoch = np.arange(offsets[index], offsets[index + 1])
        color = colors[index]
        axes[0, 0].semilogy(
            epoch, curve["train"], color=color, marker="o", ms=2, lw=1
        )
        axes[0, 0].semilogy(
            epoch, curve["val"], color=color, marker="x", ms=2, lw=0.8, ls="--"
        )
        axes[0, 1].semilogy(epoch, curve["train_h1"], color=color, lw=1)
        axes[0, 1].semilogy(
            epoch, curve["train_pde"], color=color, lw=0.8, ls=":"
        )
        axes[1, 0].semilogy(
            epoch, curve["val_mean_rse"], color=color, marker="o", ms=2, lw=1
        )
        axes[1, 0].semilogy(
            epoch, curve["val_mean_rel_res"], color=color, lw=0.8, ls="--"
        )
        axes[1, 1].plot(epoch, curve["elapsed"], color=color, marker="o", ms=2)
    axes[0, 0].set_title("Train Total (solid) · Validation Total (dashed)")
    axes[0, 1].set_title("Train H1 (solid) · Train PDE (dotted)")
    axes[1, 0].set_title("Validation RSE (solid) · Relative Residual (dashed)")
    axes[1, 1].set_title("Wall Time within Each Grade")
    for ax in axes.flat:
        ax.set_xlabel("Global Epoch")
        _decorate_grade_boundaries(ax, curves, offsets, colors)
    combined_path = out_dir / "all_grades_every_epoch_metrics.png"
    _save_dark(fig, combined_path)

    # Original notebook-style individual scientific figures.
    specifications = [
        (
            "all_grades_loss.png",
            "[All Grades] Training Loss",
            "Loss (log)",
            lambda curve: (
                curve["train"],
                curve["train_h1"],
                curve["train_pde"],
            ),
            False,
        ),
        (
            "all_grades_loss_normalized.png",
            "[All Grades] Normalized Training Loss",
            r"$\widetilde{L}_n$ (log)",
            lambda curve: tuple(
                np.asarray(curve[key])
                / (float(curve[key][0]) + 1.0e-30)
                for key in ("train", "train_h1", "train_pde")
            ),
            False,
        ),
        (
            "all_grades_rse.png",
            "[All Grades] Validation RSE",
            "RSE (log)",
            lambda curve: (
                curve["val_mean_rse"],
                curve["val_max_rse"],
                curve["val_mean_rel_res"],
            ),
            False,
        ),
    ]
    for filename, title, ylabel, getter, _ in specifications:
        fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d0d0d")
        fig.suptitle(title, fontsize=12, color="white", fontweight="bold")
        for index, curve in enumerate(curves):
            epoch = np.arange(offsets[index], offsets[index + 1])
            series = getter(curve)
            for values_i, linestyle in zip(series, ("-", "--", ":")):
                ax.semilogy(
                    epoch,
                    values_i,
                    color=colors[index],
                    lw=1,
                    ls=linestyle,
                    marker="o",
                    ms=1.5,
                )
        ax.set_xlabel("Global Epoch")
        ax.set_ylabel(ylabel)
        _decorate_grade_boundaries(ax, curves, offsets, colors)
        _save_dark(fig, out_dir / filename)

    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d0d0d")
    fig.suptitle(
        "[All Grades] Training Wall Time",
        fontsize=12,
        color="white",
        fontweight="bold",
    )
    for index, curve in enumerate(curves):
        epoch = np.arange(offsets[index], offsets[index + 1])
        ax.plot(epoch, curve["elapsed"], color=colors[index], marker="o", ms=2)
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel("Elapsed seconds within grade")
    _decorate_grade_boundaries(ax, curves, offsets, colors)
    _save_dark(fig, out_dir / "all_grades_time.png")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    grade_index = np.arange(1, len(curves) + 1)
    best_pde = [float(np.min(curve["val_pde"])) for curve in curves]
    ax.semilogy(grade_index, best_pde, "o-", color="#44ddff", lw=2, ms=8)
    for gx, value in zip(grade_index, best_pde):
        ax.annotate(
            f"{value:.2e}",
            (gx, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color="#44ddff",
        )
    ax.set_xticks(grade_index)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Best validation PDE loss")
    ax.set_title("PINN/PDE Loss per Grade")
    _save_dark(fig, out_dir / "pinn_loss_per_grade.png")
    return csv_path, combined_path
