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


def grade_color(grade, n_grades=3):
    return CGRADE((grade - 1) / max(n_grades - 0.5, 1.0))


def save_dark(fig, path, dpi=300):
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_grade(record, out_dir, kappa):
    out_dir = Path(out_dir)
    grade = record["grade"]
    color = grade_color(grade)
    x = record["x"]
    k = np.arange(1, len(x) + 1)
    selected = record["selected_k"]
    amplitude = np.abs(record["preconditioned_hat"])
    target_hat = np.abs(record["target_correction_hat"])
    output_hat = np.abs(record["correction_hat"])

    fig = plt.figure(figsize=(16, 24), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Frozen Multigrade Recurrent-DST-FNO]  Grade {grade}  ·  "
        f"κ={kappa:.1f}  ·  |K_n|={len(selected)}  ·  "
        f"H={record['selection_entropy']:.3f}  ·  η={record['eta']:.3f}",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=0.99,
    )
    gs = gridspec.GridSpec(6, 2, fig, hspace=0.55, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(
        k, amplitude + 1.0e-30, color="#aaa", lw=0.9,
        label=r"$|\widehat p_{n-1}(k)|$",
    )
    ax.axhline(
        record["threshold"], color="#ff4455", lw=1.5, ls="--",
        label=f"dynamic τ={record['threshold']:.2e}",
    )
    ax.scatter(
        selected, amplitude[selected - 1], color="#4499ff", s=25,
        zorder=5, label=rf"$K_n$: {len(selected)} frequencies",
    )
    ax.set_title("Dynamic Residual-Frequency Input — Selection Threshold")
    ax.set_xlabel("k")
    ax.set_ylabel("Amplitude (log)")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(
        k, target_hat + 1.0e-30, color="#888", lw=0.9,
        label=r"$|\widehat\delta_n(k)|$",
    )
    ax.scatter(
        selected, target_hat[selected - 1] + 1.0e-30,
        color="#ff8844", s=20, label=r"$K_n$",
    )
    ax.set_title(r"Target Correction $\delta_n$ Spectrum")
    ax.set_xlabel("k")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(
        selected, record["selected_log_norm"], color="#44ddff",
        alpha=0.85, width=1.5, label="log-norm",
    )
    ax2 = ax.twinx()
    ax2.bar(
        selected, record["selected_alpha"], color=color,
        alpha=0.4, width=1.5, label=r"$\alpha_k$",
    )
    ax.set_title(r"Selected Log-Amplitude vs $\alpha_k$")
    ax.set_xlabel("k")
    ax.set_ylabel("log-norm", color="#44ddff")
    ax2.set_ylabel(r"$\alpha_k$", color=color)

    ax = fig.add_subplot(gs[2, 0])
    components = record["selected_components"]
    n_show = min(8, len(components))
    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, max(n_show, 1)))
    for index in range(n_show):
        ax.plot(
            x, components[index], color=colors[index], lw=0.9,
            label=f"k={selected[index]}",
        )
    ax.set_title(r"Dynamic Frequency Input $s_n(x)$")
    ax.set_xlabel("x")
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(
        x, record["target_correction"], color="#888", lw=1.2,
        ls="--", label=r"Target $\delta_n$",
    )
    ax.plot(
        x, record["correction"], color=color, lw=1.5,
        label=rf"Output $G_{{\theta_{grade}}}$",
    )
    ax.set_title(f"Grade {grade} Output vs Target")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(x, record["truth"], color="white", lw=2, label=r"$u^*$")
    ax.plot(
        x, record["before"], color="#666", lw=1.2, ls="--",
        label=rf"$u_{{{grade-1}}}$",
    )
    ax.plot(
        x, record["after"], color=color, lw=1.5,
        label=rf"$u_{{{grade}}}$",
    )
    ax.set_title("Cumulative Solution")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[3, 1])
    ax.semilogy(
        x, np.abs(record["before"] - record["truth"]) + 1.0e-30,
        color="#666", lw=1.2, ls="--",
        label=f"before {record['rse_before']:.3e}",
    )
    ax.semilogy(
        x, np.abs(record["after"] - record["truth"]) + 1.0e-30,
        color=color, lw=1.5,
        label=f"after {record['rse_after']:.3e}",
    )
    ax.set_title(r"Pointwise Error $|u^*-u_n|$")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    curve = record["curve"]
    epoch = np.arange(len(curve["loss"]))
    ax = fig.add_subplot(gs[4, 0])
    ax.semilogy(
        epoch, curve["loss"], color=color, lw=1.2,
        marker="o", ms=1.5, label="Grade-local objective",
    )
    ax.semilogy(
        epoch, curve["pde"], color="#44ff88", lw=0.9,
        ls="--", label="Local residual contraction",
    )
    ax.semilogy(
        epoch, np.asarray(curve["pinn"]) + 1.0e-30,
        color="#ff8844", lw=0.9, ls=":", label="PINN",
    )
    ax.set_title(
        "Grade-local Training Objective — not a cross-grade scale"
    )
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[4, 1])
    ax.bar(
        selected, record["selected_alpha"], color=color,
        alpha=0.85, width=1.5,
    )
    ax.set_title(r"$\alpha_k$ Dynamic Frequency Weights")
    ax.set_xlabel("k")

    ax = fig.add_subplot(gs[5, :])
    ax.semilogy(
        k, target_hat + 1.0e-30, color="#aaa", lw=1, ls="--",
        label=r"$|\widehat\delta_n(k)|$ target",
    )
    ax.semilogy(
        k, output_hat + 1.0e-30, color=color, lw=1,
        label=r"$|\widehat G_{\theta_n}(k)|$ output",
    )
    ax.scatter(
        selected, output_hat[selected - 1] + 1.0e-30,
        color="#4499ff", s=20, label=r"$K_n$",
    )
    ax.set_title("Output Spectrum vs Target")
    ax.set_xlabel("k")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    fig.subplots_adjust(top=0.965)
    path = out_dir / f"grade_{grade:02d}_diag.png"
    save_dark(fig, path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Grade {grade}] Every Epoch · one marker = one stored epoch",
        color="white", fontweight="bold",
    )
    axes[0, 0].scatter(epoch, curve["loss"], s=10, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Training Objective")
    axes[0, 1].scatter(epoch, curve["pinn"], s=10, color="#ff8844")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Absolute PINN Loss")
    axes[0, 2].scatter(epoch, curve["rse"], s=10, color="#44ddff")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("RSE (evaluation only)")
    axes[1, 0].scatter(
        epoch, curve["relative_residual"], s=10, color="#44ff88"
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Relative PDE Residual")
    axes[1, 1].step(epoch, curve["lr"], where="post", color="#ffcc44")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Learning Rate Used (warmup + dual plateau)")
    axes[1, 2].scatter(epoch, curve["elapsed"], s=10, color="#ff8844")
    axes[1, 2].set_title("Elapsed Seconds")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    epoch_path = out_dir / f"grade_{grade:02d}_epoch_scatter.png"
    save_dark(fig, epoch_path, dpi=250)
    return path, epoch_path


def plot_summary(records, out_dir):
    out_dir = Path(out_dir)
    ng = len(records)
    colors = [grade_color(i + 1, ng) for i in range(ng)]
    x = records[0]["x"]
    final_rse = records[-1]["rse_after"]
    fig = plt.figure(figsize=(18, 15), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Frozen Multigrade Recurrent-DST-FNO] Summary · {ng} Grades · "
        f"Final RSE={final_rse:.2e}",
        fontsize=13, fontweight="bold", color="white", y=0.985,
    )
    outer = gridspec.GridSpec(
        3, 1, fig, top=0.91, bottom=0.05, hspace=0.48,
        height_ratios=[1.6, 1, 1],
    )
    top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], wspace=0.34
    )
    ax_training = fig.add_subplot(top[0])
    ax_pinn = fig.add_subplot(top[1])
    ax_relative = fig.add_subplot(top[2])
    offset = 0
    boundaries = [0]
    all_pinn = []
    for index, record in enumerate(records):
        curve = record["curve"]
        ne = len(curve["loss"])
        ep = np.arange(offset, offset + ne)
        training_values = np.asarray(curve["loss"]) + 1.0e-30
        best_training_values = (
            np.asarray(curve["best_loss"]) + 1.0e-30
        )
        pinn_values = np.asarray(curve["pinn"]) + 1.0e-30
        relative_training = training_values / training_values[0]
        relative_best_training = (
            best_training_values / training_values[0]
        )
        all_pinn.append(pinn_values)
        ax_training.scatter(
            ep, training_values, color=colors[index], s=4, alpha=0.22,
        )
        ax_training.semilogy(
            ep, best_training_values, color=colors[index], lw=1.2,
            label=f"G{index+1}",
        )
        ax_pinn.scatter(
            ep, pinn_values, color=colors[index], s=5, alpha=0.24,
            label="Raw PINN epochs" if index == 0 else None,
        )
        ax_relative.scatter(
            ep, relative_training, color=colors[index], s=4, alpha=0.22,
        )
        ax_relative.semilogy(
            ep, relative_best_training, color=colors[index], lw=1.2,
            label=f"G{index+1}",
        )
        offset += ne
        boundaries.append(offset)
    all_pinn = np.concatenate(all_pinn)
    global_epoch = np.arange(len(all_pinn))
    global_best_pinn = np.minimum.accumulate(all_pinn)
    ax_pinn.semilogy(
        global_epoch,
        global_best_pinn,
        color="#44ddff",
        lw=1.6,
        label="Global best-so-far PINN (monotone)",
    )
    for boundary in boundaries[1:-1]:
        for axis in (ax_training, ax_pinn, ax_relative):
            axis.axvline(boundary, color="#555", lw=0.8, ls="--")
    ax_training.set_title(
        "Training Loss\n(raw points + grade-local best envelope)"
    )
    ax_training.set_ylabel(r"$\mathcal{L}_{train}^{(n)}$")
    ax_training.legend(fontsize=7, ncol=2)
    ax_pinn.set_title(
        "Absolute PINN Loss\n(raw points + monotone best envelope)"
    )
    ax_pinn.set_ylabel(r"$(1/N)\|L_\kappa u-f\|_2^2$")
    ax_pinn.legend(fontsize=7)
    ax_relative.set_title(
        "Relative Training Loss\n"
        r"raw + best: $\mathcal{L}_{train}^{(n)}(e)"
        r"/\mathcal{L}_{train}^{(n)}(0)$"
    )
    ax_relative.set_ylabel("Dimensionless ratio")
    ax_relative.legend(fontsize=7, ncol=2)
    for axis in (ax_training, ax_pinn, ax_relative):
        axis.set_xlabel("Global Epoch")

    middle = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.38
    )
    ax = fig.add_subplot(middle[0])
    ax.plot(x, records[-1]["truth"], color="white", lw=2, label=r"$u^*$")
    ax.plot(
        x, records[-1]["after"], color="#ff66cc", lw=1.3,
        ls="--", label=rf"$u_{{approx}}$ ({final_rse:.2e})",
    )
    ax.set_title("Final Solution vs True")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(middle[1])
    ax.plot(x, records[0]["truth"], color="white", lw=2, label=r"$u^*$")
    for index, record in enumerate(records):
        ax.plot(x, record["after"], color=colors[index], lw=1, label=f"G{index+1}")
    ax.set_title("Solution Evolution")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(middle[2])
    for index, record in enumerate(records):
        ax.semilogy(
            x, np.abs(record["after"] - record["truth"]) + 1.0e-30,
            color=colors[index], lw=1, label=f"G{index+1}",
        )
    ax.set_title("Pointwise Error Evolution")
    ax.legend(fontsize=7)

    bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.38
    )
    grades = np.arange(1, ng + 1)
    rse = [record["rse_after"] for record in records]
    ax = fig.add_subplot(bottom[0])
    ax.semilogy(grades, rse, "o-", color="#44ddff", lw=2)
    for gx, value in zip(grades, rse):
        ax.annotate(
            f"{value:.1e}", (gx, value), xytext=(0, 8),
            textcoords="offset points", ha="center", color="#44ddff",
        )
    ax.set_title("RSE per Grade")
    ax.set_xticks(grades)

    counts = [len(record["selected_k"]) for record in records]
    ax = fig.add_subplot(bottom[1])
    bars = ax.bar(grades, counts, color=colors)
    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.1,
            str(value), ha="center", color="white",
        )
    ax.set_title(r"$|K_n|$ Dynamic Frequencies")
    ax.set_xticks(grades)

    improvement = [
        (record["rse_before"] - record["rse_after"])
        / (record["rse_before"] + 1.0e-30)
        * 100
        for record in records
    ]
    ax = fig.add_subplot(bottom[2])
    bars = ax.bar(
        grades, improvement,
        color=["#00cc88" if value >= 0 else "#ff4455" for value in improvement],
    )
    for bar, value in zip(bars, improvement):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value, 0) + 0.5,
            f"{value:.1f}%", ha="center", color="white",
        )
    ax.set_title("RSE Improvement per Grade")
    ax.set_xticks(grades)
    ax.axhline(0, color="#555")
    path = out_dir / "helmholtz_multigrade_summary.png"
    save_dark(fig, path)
    return path


def save_all_epoch_metrics(curves, out_dir, scalar_keys):
    out_dir = Path(out_dir)
    rows = []
    offset = 0
    for grade, curve in enumerate(curves, start=1):
        for local in range(len(curve["loss"])):
            rows.append(
                [grade, local + 1, offset + local + 1]
                + [curve[key][local] for key in scalar_keys]
            )
        offset += len(curve["loss"])
    values = np.asarray(rows)
    csv_path = out_dir / "all_grades_every_epoch_metrics.csv"
    np.savetxt(
        csv_path, values, delimiter=",",
        header="grade,local_epoch,global_epoch," + ",".join(scalar_keys),
        comments="",
        fmt=["%d", "%d", "%d"] + ["%.17e"] * len(scalar_keys),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0d0d0d")
    offsets = np.cumsum([0] + [len(curve["loss"]) for curve in curves])
    all_pinn = np.concatenate(
        [np.asarray(curve["pinn"]) + 1.0e-30 for curve in curves]
    )
    global_best_pinn = np.minimum.accumulate(all_pinn)
    for index, curve in enumerate(curves):
        ep = np.arange(offsets[index], offsets[index + 1])
        color = grade_color(index + 1, len(curves))
        axes[0, 0].scatter(
            ep, curve["pinn"], s=7, color=color, alpha=0.3
        )
        axes[0, 1].scatter(ep, curve["rse"], s=7, color=color)
        axes[1, 0].scatter(
            ep, curve["relative_residual"], s=7, color=color
        )
        axes[1, 1].scatter(
            ep, curve["loss"], s=7, color=color, alpha=0.25
        )
        axes[1, 1].plot(
            ep, curve["best_loss"], lw=1.2, color=color
        )
    axes[0, 0].semilogy(
        np.arange(len(global_best_pinn)),
        global_best_pinn,
        color="#44ddff",
        lw=1.5,
        label="global best-so-far",
    )
    axes[0, 0].legend(fontsize=8)
    for axis, title in zip(
        axes.flat,
        (
            "Absolute PINN: raw + monotone best-so-far",
            "RSE (evaluation only)",
            "Relative PDE Residual",
            "Grade-local Objective: raw + best envelope",
        ),
    ):
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Global Epoch")
        for boundary in offsets[1:-1]:
            axis.axvline(boundary, color="#444", ls="--")
    fig.suptitle(
        "[All Grades] Every Epoch · one marker = one stored epoch",
        color="white", fontweight="bold",
    )
    plot_path = out_dir / "all_grades_every_epoch_metrics.png"
    save_dark(fig, plot_path)
    return csv_path, plot_path
