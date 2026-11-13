"""
grbd.viz.plots
==============
All plotting functions for GRBD framework.
Dark-theme, publication-ready.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    "font.family":"monospace",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.2,
    "figure.facecolor":"#0d0d0d","axes.facecolor":"#181818",
    "axes.labelcolor":"#bbb","axes.titlecolor":"#fff",
    "xtick.color":"#888","ytick.color":"#888",
    "grid.color":"#2a2a2a","text.color":"#ccc",
    "legend.framealpha":0.3,"legend.facecolor":"#181818",
})
CGRADE = plt.cm.plasma


def plot_grade(n, op, info, out_dir="./grbd_output"):
    """Per-grade diagnostic figure (6-row layout)."""
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    color = CGRADE(n / 8.0); x = op.grid
    basis_name = info.get("basis_name", "")
    fig = plt.figure(figsize=(16,22), facecolor="#0d0d0d")
    fig.suptitle(
        f"[GRBD]  Grade {n}  ·  {basis_name}  ·  "
        f"|K|={info['n_atoms']}  H={info['entropy']:.3f}  lr={info['lr_used']:.2e}",
        fontsize=12, fontweight="bold", color="white", y=0.99)
    gs = gridspec.GridSpec(5, 2, fig, hspace=0.55, wspace=0.35)

    # row 0: residual spectrum (DST/FFT bases) or subband energy (wavelet)
    ax = fig.add_subplot(gs[0,:])
    amps_res = info.get("amps_res", None)
    if amps_res is not None:
        k_all = info.get("k_all", np.arange(1,len(amps_res)+1))
        mask  = info.get("mask", info.get("mask_flat", np.zeros(len(amps_res),bool)))
        tau   = info.get("tau", None)
        ax.semilogy(k_all, amps_res+1e-12, color="#aaa", lw=0.9, label="|ê_n(k)|")
        if tau is not None:
            ax.axhline(tau, color="#ff4455", lw=1.5, ls="--", label=f"τ={tau:.4f}")
        sel = k_all[mask] if len(mask)==len(k_all) else k_all[np.where(mask)[0]]
        ax.scatter(sel, amps_res[mask]+1e-12, color="#4499ff", s=25, zorder=5,
                   label=f"K_n: {info['n_atoms']} atoms")
        ax.set_title("Residual Spectrum / GD Threshold")
    else:
        ax.set_title("Residual (wavelet subband energy)")
    ax.legend(fontsize=8); ax.set_xlabel("atom index")

    # row 1: solution comparison
    ax = fig.add_subplot(gs[1,0])
    u_star  = info.get("u_star_ref", np.zeros_like(x))
    u_before= info.get("u_before",   np.zeros_like(x))
    u_after = info.get("u_after",    np.zeros_like(x))
    ax.plot(x, u_star,   color="white", lw=2,   label="u*",         zorder=3)
    ax.plot(x, u_before, color="#666",  lw=1.2, ls="--", alpha=0.7, label=f"u_{n-1}")
    ax.plot(x, u_after,  color=color,   lw=1.5, label=f"u_{n}")
    ax.legend(fontsize=8); ax.set_title("Solution"); ax.set_xlabel("x")

    # row 1: pointwise error
    ax = fig.add_subplot(gs[1,1])
    eb = info["rel_err_before"]; ea = info["rel_err_after"]
    ax.semilogy(x, np.abs(u_before-u_star)+1e-12, "--", color="#666",
                lw=1.2, alpha=0.7, label=f"before {eb:.3e}")
    ax.semilogy(x, np.abs(u_after -u_star)+1e-12,
                color=color, lw=1.5, label=f"after  {ea:.3e}")
    ax.legend(fontsize=8); ax.set_title("|u*−u_n|"); ax.set_xlabel("x")

    # row 2: g output vs target
    ax = fig.add_subplot(gs[2,0])
    target = info.get("target", np.zeros_like(x))
    g_np   = info.get("g_np",   np.zeros_like(x))
    ax.plot(x, target, color="#888", lw=1.2, ls="--", label="δ_n (target)")
    ax.plot(x, g_np,   color=color,  lw=1.5, label=f"g_{n} [best ckpt]")
    ax.legend(fontsize=8); ax.set_title("Output vs Target"); ax.set_xlabel("x")

    # row 2: alpha weights
    ax = fig.add_subplot(gs[2,1])
    alphas = info.get("alphas", np.array([1.0]))
    ax.bar(np.arange(len(alphas)), alphas, color=color, alpha=0.85)
    ax.set_title("α_k  (Sobolev/energy weights)"); ax.set_xlabel("atom index")

    # row 3: loss curves
    ax = fig.add_subplot(gs[3,:])
    losses    = info.get("losses",     [])
    losses_in = info.get("losses_in",  [])
    losses_oob= info.get("losses_oob", [])
    if losses:
        ax.semilogy(losses,     color=color,     lw=1.2, label="Total")
        ax.semilogy(losses_in,  color="#44ff88", lw=0.9, ls="--", alpha=0.7, label="In-band")
        ax.semilogy(losses_oob, color="#ff8844", lw=0.9, ls=":",  alpha=0.7, label="OOB")
    ax.legend(fontsize=7); ax.set_title(f"Loss  (best-checkpoint)"); ax.set_xlabel("Epoch")

    # row 4: input feature matrix s (first 8 atoms)
    ax = fig.add_subplot(gs[4,:])
    s = info.get("s_input", None)
    if s is not None:
        nshow = min(8, s.shape[1])
        cols  = plt.cm.coolwarm(np.linspace(0.05,0.95,nshow))
        for i in range(nshow):
            ax.plot(x, s[:,i], color=cols[i], lw=0.9, alpha=0.85)
    ax.set_title("Input features s_n (first 8 atoms)"); ax.set_xlabel("x")

    fname = out_dir / f"grade_{n:02d}_{basis_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {fname.name}")
    return fname


def plot_summary(infos, op, u_star, u_pred, out_dir="./grbd_output"):
    """Summary figure: RSE curve, loss, spectra, solution."""
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    NG   = len(infos)
    cols = [CGRADE(i/max(NG-0.5,1)) for i in range(NG)]
    x    = op.grid

    fig, axes = plt.subplots(2,3, figsize=(16,9), facecolor="#0d0d0d")
    fig.suptitle(
        f"[GRBD Summary]  {infos[0]['basis_name']}  {NG} grades  "
        f"final RSE={infos[-1]['rel_err_after']:.2e}",
        fontsize=12, fontweight="bold", color="white")

    ax = axes[0,0]
    ax.plot(x, u_star, color="white", lw=2.5, label="u* (true)", zorder=3)
    ax.plot(x, u_pred, color="#ff6633", lw=1.5, ls="--", label="u_N (predicted)")
    ax.legend(); ax.set_title("Final Solution")

    ax = axes[0,1]
    eb=[i["rel_err_before"] for i in infos]
    ea=[i["rel_err_after"]  for i in infos]
    ax.semilogy(range(1,NG+1), eb, "o--", color="#888",    lw=1.5, ms=5, label="Before")
    ax.semilogy(range(1,NG+1), ea, "s-",  color="#ff6633", lw=2,   ms=7, label="After")
    ax.set_title("RSE per Grade (monotone ↓)"); ax.legend(); ax.set_xlabel("Grade")

    ax = axes[0,2]
    ax.bar(range(1,NG+1), [i["n_atoms"] for i in infos], color=cols, alpha=0.85)
    ax.set_title("|K_n| per Grade"); ax.set_xlabel("Grade")

    ax = axes[1,0]; ax2 = ax.twinx()
    l1,=ax.plot(range(1,NG+1), [i["entropy"]  for i in infos],
                "^-", color="#44ff88", lw=2, ms=8, label="H_n")
    l2,=ax2.semilogy(range(1,NG+1), [i["lr_used"] for i in infos],
                     "D--", color="#ffdd44", lw=1.5, ms=7, label="lr")
    ax.legend(handles=[l1,l2]); ax.set_title("Entropy & lr")
    ax.set_ylabel("H_n", color="#44ff88"); ax2.set_ylabel("lr", color="#ffdd44")

    ax = axes[1,1]
    for i,info in enumerate(infos):
        ax.semilogy(info["losses"], color=cols[i], lw=1.0, alpha=0.9, label=f"G{i+1}")
    ax.legend(fontsize=8,ncol=2); ax.set_title("Loss per Grade"); ax.set_xlabel("Epoch")

    ax = axes[1,2]
    ax.semilogy(x, np.abs(u_pred-u_star)+1e-12, color="#ff6633", lw=1.5)
    ax.set_title("Final Pointwise Error |u*−u_N|"); ax.set_xlabel("x")

    fname = out_dir / "summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {fname.name}")
    return fname


def plot_comparison(solvers_dict, op, u_star, out_dir="./grbd_output"):
    """
    Compare multiple solvers (different bases) on same problem.

    Parameters
    ----------
    solvers_dict : dict  {label: GradeSolver}  e.g. {"DST": s1, "Wavelet": s2}
    """
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    x    = op.grid
    cols = plt.cm.tab10(np.linspace(0,0.8,len(solvers_dict)))
    fig, axes = plt.subplots(1,3, figsize=(16,5), facecolor="#0d0d0d")
    fig.suptitle("[GRBD Suite Comparison]", fontsize=13, color="white", fontweight="bold")

    ax = axes[0]
    ax.plot(x, u_star, color="white", lw=2.5, label="u* (true)", zorder=5)
    for (label,solver),c in zip(solvers_dict.items(),cols):
        ax.plot(x, solver.u, color=c, lw=1.5, ls="--", label=f"{label} (RSE={solver.final_rse:.2e})")
    ax.legend(fontsize=8); ax.set_title("Final Solutions")

    ax = axes[1]
    for (label,solver),c in zip(solvers_dict.items(),cols):
        ea=[i["rel_err_after"] for i in solver.infos]
        ax.semilogy(range(1,len(ea)+1), ea, "s-", color=c, lw=2, ms=7, label=label)
    ax.set_title("RSE per Grade"); ax.legend(); ax.set_xlabel("Grade")

    ax = axes[2]
    for (label,solver),c in zip(solvers_dict.items(),cols):
        ax.semilogy(x, np.abs(solver.u-u_star)+1e-12, color=c, lw=1.5, label=label)
    ax.set_title("Final Pointwise Error"); ax.legend(); ax.set_xlabel("x")

    fname = out_dir / "comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {fname.name}")
    return fname
