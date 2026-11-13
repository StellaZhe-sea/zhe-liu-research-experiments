"""Generate the summary figures and machine-readable metrics for the
public results pack.

All numbers are transcribed from saved execution logs / executed notebooks
in the local research archive. Each entry in metrics.csv/json carries a
``source`` field naming the originating experiment run so every figure
stays auditable.

Usage:
    python tools/make_summary_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ---------------------------------------------------------------------------
# Metric table (single source of truth for both CSV and JSON)
# ---------------------------------------------------------------------------

METRICS = [
    # experiment, metric, value, scope, source
    ("helmholtz_single_h1_8", "relative_l2_error", 2.2656e-14,
     "single synthetic PDE, manufactured solution", "helmholtz_h1_pde_single"),
    ("helmholtz_single_reviewed_v9", "relative_l2_error", 4.9660e-06,
     "single PDE, k=200pi, RSE-gated spectral multigrade", "helmholtz_spectral_v9"),
    ("helmholtz_single_recurrent_dst_fno", "relative_l2_error", 1.153697e-05,
     "single PDE, recurrent DST-FNO", "helmholtz_recurrent_dst_fno"),
    ("helmholtz_single_frozen_3grade", "relative_l2_error", 1.1612746e-10,
     "single PDE, frozen grades 1-3", "helmholtz_frozen_operator"),
    ("helmholtz_single_frozen_3grade", "relative_residual", 1.1594304e-10,
     "single PDE, frozen grades 1-3", "helmholtz_frozen_operator"),
    ("helmholtz_single_frozen_3grade", "pinn_residual", 1.4757e-09,
     "single PDE, frozen grades 1-3", "helmholtz_frozen_operator"),
    ("helmholtz_single_historical_dual4", "relative_l2_error", 2.5995406e-12,
     "single PDE, historical spectral adaptive dual 4-grade", "historical_log"),
    ("helmholtz_operator_recurrent", "unseen_mean_relative_l2_error", 2.013989e-04,
     "160 train / 40 val / 40 unseen test PDEs", "helmholtz_operator_recurrent"),
    ("helmholtz_operator_recurrent", "unseen_max_relative_l2_error", 5.626741e-03,
     "160 train / 40 val / 40 unseen test PDEs", "helmholtz_operator_recurrent"),
    ("helmholtz_operator_recurrent", "unseen_mean_relative_residual", 1.827555e-05,
     "160 train / 40 val / 40 unseen test PDEs", "helmholtz_operator_recurrent"),
    ("helmholtz_operator_historical_frozen", "unseen_mean_relative_l2_error", 1.347329e-04,
     "historical frozen operator run", "historical_log"),
    ("helmholtz_operator_historical_frozen", "unseen_max_relative_l2_error", 3.413798e-03,
     "historical frozen operator run", "historical_log"),
    ("helmholtz_operator_historical_frozen", "unseen_mean_relative_residual", 1.209998e-05,
     "historical frozen operator run", "historical_log"),
    ("helmholtz2d_fft_fft", "best_accepted_relative_l2_error", 1.6404e-09,
     "2D periodic synthetic problem; grades 7-8 rejected", "mgdl_spectral_suite"),
    ("helmholtz2d_fft_chebyshev", "best_accepted_relative_l2_error", 1.6405e-01,
     "negative result: grade 3 rejected, run not competitive", "mgdl_spectral_suite"),
    ("nls1d_cubic_historical", "relative_l2_error", 7.1940e-09,
     "manufactured nonlinear problem, 3 grades", "historical_log"),
    ("nls2d_split_step", "mass_relative_error", 4.29e-13,
     "conservation check after Strang fix", "nls2d_operator_validation"),
    ("nls2d_split_step", "hamiltonian_relative_error", 4.56e-12,
     "conservation check after Strang fix", "nls2d_operator_validation"),
    ("nls2d_split_step", "hamiltonian_relative_error_before_fix", 1.16,
     "same integrator before the Strang fix", "nls2d_operator_validation"),
    ("nls2d_grade_ablation", "baseline_final_rse", 4.9309e-02,
     "8 grades, 3000 epochs per grade", "nls2d_grade_ablation"),
    ("nls2d_grade_ablation", "method_b_final_rse", 4.9301e-02,
     "8 grades, 3000 epochs per grade", "nls2d_grade_ablation"),
    ("nls2d_grade_ablation", "pi_gnn_final_rse", 4.8256e-02,
     "8 grades; about 2.14% relative improvement over baseline", "nls2d_grade_ablation"),
    ("kolmogorov_fourier", "validation_relative_squared_error", 9.904e-03,
     "Fourier basis, Python training run", "kolmogorov_train_py"),
    ("kolmogorov_wavelet", "validation_relative_squared_error_printed", 0.0,
     "printed as 0.000000; treat as below print precision, not exact zero",
     "kolmogorov_train_py"),
    ("kolmogorov_lifting_historical", "validation_relative_squared_error", 2.431e-03,
     "historical lifting basis run", "historical_log"),
    ("kolmogorov_wavelet_serial_historical", "validation_relative_squared_error", 2.0e-06,
     "historical serial wavelet run (approximate)", "historical_log"),
    ("spectral_gnn_ablation_baseline", "final_relative_l2_error", 6.8709e-03,
     "1D Helmholtz N=256 kappa=200pi, 2000 epochs", "spectral_gnn_ablation"),
    ("spectral_gnn_ablation_energy_l2", "final_relative_l2_error", 4.0477e-02,
     "5.891x worse than baseline", "spectral_gnn_ablation"),
    ("spectral_gnn_ablation_distance_l2", "final_relative_l2_error", 3.5753e-02,
     "5.203x worse than baseline", "spectral_gnn_ablation"),
    ("spectral_gnn_ablation_mlp_l2", "final_relative_l2_error", 4.6213e-02,
     "6.726x worse than baseline", "spectral_gnn_ablation"),
    ("spectral_gnn_ablation_selection_l2", "final_relative_l2_error", 3.4302e-02,
     "4.992x worse than baseline", "spectral_gnn_ablation"),
    ("spectral_gnn_ablation_cascade_l2", "final_relative_l2_error", 1.6010e-02,
     "2.330x worse than baseline", "spectral_gnn_ablation"),
    ("spectral_gnn_ablation_energy_l0", "final_relative_l2_error", 2.9611e-02,
     "4.310x worse than baseline", "spectral_gnn_ablation"),
    ("reference_mgdl_regression", "test_relative_squared_error", 5.5488e-06,
     "third-party reference implementation, not an original contribution",
     "reference_mgdl_regression"),
    ("reference_mgdl_regression", "validation_relative_squared_error", 2.0093e-04,
     "third-party reference implementation", "reference_mgdl_regression"),
    ("reference_sgdl_regression", "test_relative_squared_error", 9.97109e-01,
     "third-party reference; only 5 epochs, not a fair MGDL comparison",
     "reference_sgdl_regression"),
    ("reference_sgdl_mnist", "test_accuracy", 8.1460e-01,
     "third-party SGDL classifier, 100 epochs", "reference_mgdl_classification"),
    ("reference_sgdl_mnist", "train_accuracy", 9.6052e-01,
     "third-party SGDL classifier, 100 epochs", "reference_mgdl_classification"),
]


def write_metrics() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "metric", "value", "scope", "source"])
        for row in METRICS:
            w.writerow([row[0], row[1], f"{row[2]:.6e}", row[3], row[4]])
    payload = [
        {"experiment": e, "metric": m, "value": v, "scope": s, "source": src}
        for e, m, v, s, src in METRICS
    ]
    (RES / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_helmholtz_versions() -> None:
    labels = ["H1-8\n(H1-PDE loss)", "Reviewed v9\n(spectral)", "Recurrent\nDST-FNO",
              "Frozen\n3-grade", "Historical\ndual 4-grade"]
    vals = [2.2656e-14, 4.9660e-06, 1.153697e-05, 1.1612746e-10, 2.5995406e-12]
    colors = ["#d62728", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Single-PDE Helmholtz: solver version progression (k = 200 pi)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6, f"{v:.2e}",
                ha="center", fontsize=9)
    ax.set_ylim(1e-15, 1e-3)
    fig.tight_layout()
    fig.savefig(FIG / "helmholtz_version_progression.png")
    plt.close(fig)


def fig_operator_unseen() -> None:
    groups = ["Recurrent DST-FNO", "Historical frozen"]
    mean_e = [2.013989e-04, 1.347329e-04]
    max_e = [5.626741e-03, 3.413798e-03]
    resid = [1.827555e-05, 1.209998e-05]
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (vals, lab, c) in enumerate([
            (mean_e, "mean unseen-test rel. L2", "#1f77b4"),
            (max_e, "max unseen-test rel. L2", "#d62728"),
            (resid, "mean rel. residual", "#2ca02c")]):
        b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v * 1.4, f"{v:.2e}",
                    ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Error (log scale)")
    ax.set_title("Helmholtz operator learning on 40 unseen test PDEs")
    ax.legend()
    ax.set_ylim(1e-6, 1e-1)
    fig.tight_layout()
    fig.savefig(FIG / "helmholtz_operator_unseen.png")
    plt.close(fig)


def fig_fft2d_grades() -> None:
    # Real per-grade "Rel L2 after" values from experiment_runs/2d_fft2d
    # and experiment_runs/2d_fftcheb run logs.
    g_fft = [1, 2, 3, 4, 5, 6, 7, 8]
    v_fft = [3.3156e-03, 2.9135e-04, 1.6460e-05, 6.1372e-08,
             3.1286e-09, 1.6404e-09, None, None]
    v_fft_rej = {7: 1.7440e-09, 8: 1.7431e-09}
    g_cheb = [1, 2, 3]
    v_cheb = [1.6405e-01, 1.6405e-01, None]
    v_cheb_rej = {3: 1.6405e-01}

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    xs = [g for g, v in zip(g_fft, v_fft) if v is not None]
    ys = [v for v in v_fft if v is not None]
    ax.plot(xs, ys, "o-", color="#1f77b4", label="FFT x FFT (accepted)")
    for g, v in v_fft_rej.items():
        ax.plot(g, v, "x", color="#d62728", ms=10, mew=2.5)
    ax.plot([], [], "x", color="#d62728", ms=10, mew=2.5,
            label="FFT x FFT (rejected)")

    xs = [g for g, v in zip(g_cheb, v_cheb) if v is not None]
    ys = [v for v in v_cheb if v is not None]
    ax.plot(xs, ys, "s--", color="#ff7f0e",
            label="FFT x Chebyshev (accepted)")
    for g, v in v_cheb_rej.items():
        ax.plot(g, v, "x", color="#ff7f0e", ms=10, mew=2.5)

    ax.set_yscale("log")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Relative L2 error after grade")
    ax.set_title("2D multi-grade Helmholtz: per-grade error and rejections\n"
                 "(FFT x FFT reaches 1.64e-09; FFT x Chebyshev stalls at 1.64e-01)")
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fft2d_grade_curve.png")
    plt.close(fig)


def fig_nls2d_conservation() -> None:
    labels = ["Mass", "Hamiltonian"]
    before = [np.nan, 1.16]
    after = [4.29e-13, 4.56e-12]
    x = np.arange(len(labels))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    b1 = ax.bar(x - w / 2, [v if not np.isnan(v) else 0 for v in before], w,
                label="before Strang fix", color="#d62728")
    b2 = ax.bar(x + w / 2, after, w, label="after Strang fix", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative conservation error (log scale)")
    ax.set_title("2D NLS split-step: conservation before / after integrator fix")
    for bb, v in zip(b1, before):
        if not np.isnan(v):
            ax.text(bb.get_x() + bb.get_width() / 2, v * 1.5, f"{v:.2e}",
                    ha="center", fontsize=9)
    for bb, v in zip(b2, after):
        ax.text(bb.get_x() + bb.get_width() / 2, v * 1.5, f"{v:.2e}",
                ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "nls2d_conservation.png")
    plt.close(fig)


def fig_kolmogorov_basis() -> None:
    labels = ["Fourier\n(this run)", "Wavelet\n(this run)*",
              "Lifting\n(historical)", "Wavelet serial\n(historical)"]
    vals = [9.904e-03, 1e-7, 2.431e-03, 2.0e-06]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Validation relative squared error")
    ax.set_title("Kolmogorov flow reconstruction: basis comparison")
    for b, v, lab in zip(bars, vals, ["9.904e-3", "< print precision", "2.431e-3", "~2e-6"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.5, lab, ha="center", fontsize=9)
    ax.set_ylim(1e-8, 1e-1)
    ax.text(0.01, 0.02,
            "* Wavelet printed 0.000000 in this run; plotted at an arbitrary\n"
            "  sub-precision value. Not evidence of exact zero.",
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "kolmogorov_basis_comparison.png")
    plt.close(fig)


def fig_gnn_ablation_bar() -> None:
    base = 6.8709e-03
    variants = ["energy (L=2)", "distance (L=2)", "mlp (L=2)",
                "selection (L=2)", "cascade (L=2)", "energy (L=0)"]
    vals = [4.0477e-02, 3.5753e-02, 4.6213e-02, 3.4302e-02, 1.6010e-02, 2.9611e-02]
    folds = [v / base for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(variants))
    bars = ax.bar(x, folds, color="#d62728")
    ax.axhline(1.0, color="#1f77b4", lw=2, label="baseline")
    for b, f, v in zip(bars, folds, vals):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.12, f"{f:.2f}x\n({v:.2e})",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("Fold error vs baseline")
    ax.set_title("SpectralGNN ablation: every GNN variant was worse (negative result)")
    ax.set_ylim(0, 8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "spectral_gnn_ablation_fold.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    write_metrics()
    fig_helmholtz_versions()
    fig_operator_unseen()
    fig_fft2d_grades()
    fig_nls2d_conservation()
    fig_kolmogorov_basis()
    fig_gnn_ablation_bar()
    print(f"wrote {len(METRICS)} metrics and 6 summary figures")


if __name__ == "__main__":
    main()
