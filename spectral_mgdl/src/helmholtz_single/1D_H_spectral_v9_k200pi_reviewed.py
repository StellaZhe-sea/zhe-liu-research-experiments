#!/usr/bin/env python
# coding: utf-8

# # MGDL Helmholtz v9 — best-ckpt + ReduceLROnPlateau
#


#
# ---


# |---|------|


#

# In[ ]:


import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import dst, idst
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path("./helmholtz_out_reviewed"); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2,
    "figure.facecolor": "#0d0d0d", "axes.facecolor": "#181818",
    "axes.labelcolor": "#bbb", "axes.titlecolor": "#fff",
    "xtick.color": "#888", "ytick.color": "#888",
    "grid.color": "#2a2a2a", "text.color": "#ccc",
    "legend.framealpha": 0.3, "legend.facecolor": "#181818",
})
CGRADE = plt.cm.plasma

print("[*] imports OK")


# In[ ]:


# Cell 2 — HelmholtzSpectralOperator


class HelmholtzSpectralOperator:
    def __init__(self, N, kappa):
        self.N = N; self.kappa = kappa
        self.h = 1.0 / (N + 1)
        self.x = np.arange(1, N + 1) * self.h
        k = np.arange(1, N + 1)
        self.k_all = k
        self.lam_k = (np.pi * k)**2 - kappa**2
        self._safe = np.where(
            np.abs(self.lam_k) > 1e-8, self.lam_k,
            np.sign(self.lam_k + 1e-12) * 1e-8
        )
        self._norm_dst = 2.0 / (N + 1)

    def make_f(self, u):
        return idst(self.lam_k * dst(u, type=1), type=1).astype(np.float32)

    def residual(self, u, f):
        u_hat = dst(u, type=1).astype(np.float32)
        f_hat = dst(f, type=1).astype(np.float32)
        return idst(self.lam_k * u_hat - f_hat, type=1).astype(np.float32)

    def inv_L(self, rhs, null_thresh=10.0):
        rhs_hat = dst(rhs, type=1).astype(np.float32)
        rhs_hat[np.abs(self.lam_k) < null_thresh] = 0.0
        return idst(rhs_hat / self._safe, type=1).astype(np.float32)

print("[*] HelmholtzSpectralOperator defined")


# In[ ]:


def gd_threshold(amps, floor_ratio=0.01):
    """Gd threshold."""
    s = np.sort(amps)[::-1]
    mad = np.median(np.abs(s - np.median(s)))
    sigma = mad / 0.6745 + 1e-12
    n = len(amps); beta = 1 / n
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    return float(max(omega * sigma, floor_ratio * s[0]))


def compute_Kn(residual_e, op, gd_floor=0.01, exclude_mask=None, select_amps=None):
    """Compute Kn."""
    N = op.N; k_all = np.arange(1, N + 1)
    e_hat = dst(residual_e, type=1).astype(np.float32)
    amps_res = np.abs(e_hat)
    score = amps_res if select_amps is None else select_amps
    tau = gd_threshold(score, floor_ratio=gd_floor)
    mask = score > tau

    if exclude_mask is not None and exclude_mask.any():
        candidate = mask & ~exclude_mask
        if candidate.any():
            mask = candidate


    if not mask.any():
        mask[np.argmax(score)] = True
    k_sel = k_all[mask]
    return mask, k_sel, e_hat, amps_res, tau


def build_spectral_input(residual_e, target_delta, op, gd_floor=0.01, exclude_mask=None,
                          select_by="target"):
    """Build spectral input."""
    N = op.N; k_all = np.arange(1, N + 1)

    t_hat = dst(target_delta, type=1).astype(np.float32)
    amps_delta = np.abs(t_hat)
    select_amps = amps_delta if select_by == "target" else None
    mask, k_sel, e_hat, amps_res, tau = compute_Kn(
        residual_e, op, gd_floor, exclude_mask=exclude_mask, select_amps=select_amps)

    a_sel = amps_delta[mask]
    signs = np.sign(t_hat[mask])


    energy = a_sel**2
    alphas = energy / (energy.sum() + 1e-12)
    H = float(-np.sum(alphas * np.log(alphas + 1e-12)))

    log_amps = np.log(1.0 + a_sel)
    log_max  = np.log(1.0 + a_sel.max() + 1e-12)
    log_norm = log_amps / log_max

    lam_sel  = op.lam_k[mask].astype(np.float32)
    lam_norm = (lam_sel / (op.kappa**2 + 1e-12)).astype(np.float32)
    k_norm   = (k_sel.astype(np.float32) / N)

    feat = np.stack([
        log_norm * signs,
        lam_norm,
        k_norm,
        np.zeros_like(k_norm),
        np.zeros_like(k_norm),
    ], axis=1).astype(np.float32)


    j_ = np.arange(1, N + 1)[:, None]
    basis = (np.sqrt(2.0 / (N + 1)) *
             np.sin(np.pi * k_sel[None, :] * j_ / (N + 1))).astype(np.float32)
    s_input = (log_norm[None, :] * signs[None, :] * basis).astype(np.float32)

    target_coef = (target_delta @ basis).astype(np.float32)

    meta = dict(
        k_sel=k_sel, mask=mask, tau=tau, k_all=k_all,
        n_freqs=int(mask.sum()), entropy=H,
        amps_res=amps_res, e_hat=e_hat,
        amps_delta=amps_delta, target_hat=t_hat,
        alphas=alphas, signs=signs, log_norm=log_norm,
        target=target_delta, s_input=s_input,
        lam_sel=lam_sel, target_coef=target_coef,
        basis=basis, feat=feat,
        selection_score=amps_delta if select_by == "target" else amps_res,
        selection_label=(r"$|\hat{\delta}_n(k)|$" if select_by == "target"
                         else r"$|\hat{e}_n(k)|$"),
        select_by=select_by, gd_floor=gd_floor,
    )
    return feat, meta

print("[*] spectral selection functions defined  (v13: fully spectral-space input/target)")


# In[ ]:


class LrAdaptor:
    """LrAdaptor."""
    def __init__(self,
                 base_lr    = 2e-3,
                 lr_min     = 1e-4,
                 lr_max     = 1e-3,
                 tail_frac  = 0.20,
                 slope_eps  = -1e-10,
                 decay_bad  = 0.50,
                 decay_mild = 0.70,
                 boost_good = 0.70):
        self.base_lr    = base_lr
        self.lr_min     = lr_min
        self.lr_max     = lr_max
        self.tail_frac  = tail_frac
        self.slope_eps  = slope_eps
        self.decay_bad  = decay_bad
        self.decay_mild = decay_mild
        self.boost_good = boost_good
        self._current_lr = base_lr

    def update(self, losses: list, grade_n: int) -> float:
        losses = np.array(losses, dtype=np.float64)
        n = len(losses)
        tail_start = max(0, int(n * (1 - self.tail_frac)))
        tail = losses[tail_start:]
        head = losses[:max(1, tail_start)]
        x = np.arange(len(tail), dtype=np.float64)
        slope = np.polyfit(x, tail, 1)[0] / (tail[0] + 1e-30)
        drop  = (head.mean() - tail.mean()) / (head.mean() + 1e-30)
        lr = self._current_lr
        if slope < self.slope_eps:
            reason = f'still descending (slope={slope:.2e})'
        elif drop < 0.05:
            lr = min(lr * 1.5, self.lr_max)
            reason = f'no progress (drop={drop:.2f}) x1.5'
        elif drop > 0.90:
            lr *= self.decay_mild
            reason = f'great (drop={drop:.2f}) x{self.decay_mild}'
        else:
            reason = f'moderate (drop={drop:.2f}) hold'
        lr = float(np.clip(lr, self.lr_min, self.lr_max))
        self._current_lr = lr
        print(f'  [LrAdaptor] {reason} lr={lr:.2e}')
        return lr

    @property
    def current_lr(self):
        return self._current_lr

print("[*] LrAdaptor defined")


# In[ ]:


#


#   → Tanh MLP × depth           → h_mlp  (|K_n|, width)

#


class GradeNet(nn.Module):
    def __init__(self, n_freqs, feat_dim, width, depth,
                 skip_gain_init=0.1, mem_gain_init=1.0):
        super().__init__()
        self.n_freqs = n_freqs


        self._W_raw = nn.Parameter(torch.zeros(n_freqs, n_freqs))
        self.mix_act = nn.Tanh()


        self.proj = nn.Linear(feat_dim, width)
        self.mlp  = nn.Sequential(*[
            layer
            for _ in range(depth)
            for layer in (nn.Linear(width, width), nn.Tanh())
        ])
        self.out = nn.Linear(width, 1)


        self.skip_gain = nn.Parameter(torch.tensor(float(skip_gain_init)))


        self.mem_gain = nn.Parameter(torch.tensor(float(mem_gain_init)))

        nn.init.normal_(self._W_raw, mean=0.0, std=0.01)
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)
        for m in list(self.mlp.modules()) + [self.out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    @property
    def W_sym(self):
        """W sym."""
        return 0.5 * (self._W_raw + self._W_raw.t())

    def forward(self, feat):
        """Forward."""
        h_mix = self.mix_act(self.W_sym @ feat) + feat
        h = self.mlp(self.proj(h_mix))
        c_mlp  = self.out(h)
        c_skip = self.skip_gain * feat[:, 0:1]
        c_mem  = self.mem_gain  * feat[:, 4:5]
        return c_mlp + c_skip + c_mem

print("[*] GradeNet defined  (v14: + mem_gain cross-grade memory skip)")


# In[ ]:


# loss = Σ α_k (c_k - target_coef_k)²


class SpectralLoss(nn.Module):
    def __init__(self, alphas):
        super().__init__()
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))

    def forward(self, c, target_coef):
        """Forward."""
        diff2 = (c - target_coef) ** 2
        loss  = (self.alphas * diff2).sum()
        return loss, loss.item(), 0.0

print("[*] SpectralLoss defined  (v13: pure spectral-coefficient regression)")


# In[ ]:


def plot_grade(n, op, meta, g_np, u_before, u_after, u_star,
               losses, losses_in, losses_oob, lr_used, accepted=True):
    color = CGRADE(n / 7.0); x = op.x
    reject_tag = "" if accepted else "  [REJECTED]"

    target_scale = float(meta.get("target_scale", 1.0))
    target_phys = np.asarray(meta["target"]) * target_scale
    output_phys = np.asarray(g_np)
    target_coef_phys = np.asarray(meta["target_coef"]) * target_scale
    output_coef_phys = output_phys @ np.asarray(meta["basis"])
    denom = np.linalg.norm(target_phys) * np.linalg.norm(output_phys)
    corr = float(np.dot(target_phys, output_phys) / (denom + 1e-30))
    coef_rel = float(np.linalg.norm(output_coef_phys - target_coef_phys) /
                     (np.linalg.norm(target_coef_phys) + 1e-30))
    sign_match = float(np.mean(np.sign(output_coef_phys) == np.sign(target_coef_phys)))

    fig = plt.figure(figsize=(16, 24), facecolor="#0d0d0d")
    fig.suptitle(
        f"[MGDL Helmholtz reviewed]  Grade {n}{reject_tag}  ·  κ={op.kappa:.1f}  "
        f"·  |K_n|={meta['n_freqs']}  ·  corr={corr:.6f}  ·  signed-coef err={coef_rel:.2e}",
        fontsize=13, fontweight="bold",
        color="#ff4455" if not accepted else "white", y=0.99)
    gs = gridspec.GridSpec(6, 2, fig, hspace=0.55, wspace=0.35)

    # Row 0: show the quantity that was actually thresholded.
    ax = fig.add_subplot(gs[0, :])
    score = np.asarray(meta.get("selection_score", meta["amps_res"]))
    score_label = meta.get("selection_label", r"selection score")
    ax.semilogy(meta["k_all"], score + 1e-12,
                color="#aaaaaa", lw=0.9, label=score_label)
    ax.axhline(meta["tau"], color="#ff4455", lw=1.5, ls="--",
               label=f"GD τ={meta['tau']:.4g}  (floor={meta.get('gd_floor', float('nan')):.3g})")
    ax.scatter(meta["k_all"][meta["mask"]], score[meta["mask"]],
               color="#4499ff", s=25, zorder=5,
               label=f"K_n: {meta['n_freqs']} freqs")
    ax.set_xlabel("k"); ax.set_ylabel("Selection amplitude (log)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Actual Selection Spectrum — select_by={meta.get('select_by', 'unknown')}")

    # Row 1: target spectrum and weights.
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(meta["k_all"], meta["amps_delta"] + 1e-12,
                color="#888888", lw=0.9, label=r"$|\hat{\delta}^{norm}_n(k)|$")
    ax.scatter(meta["k_all"][meta["mask"]], meta["amps_delta"][meta["mask"]],
               color="#ff8844", s=20, zorder=5, label="$K_n$")
    ax.set_title(r"Normalized target spectrum"); ax.set_xlabel("k"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(meta["k_sel"], meta["log_norm"], color="#44ddff", alpha=0.85, width=1.5,
           label="log-norm")
    ax2 = ax.twinx()
    ax2.bar(meta["k_sel"], meta["alphas"], color=color, alpha=0.4, width=1.5,
            label=r"$\alpha_k$")
    ax.set_title(r"Log-Amp Norm vs $\alpha_k$"); ax.set_xlabel("k")
    ax.set_ylabel("log-norm", color="#44ddff"); ax2.set_ylabel(r"$\alpha_k$", color=color)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=7)

    # Row 2: legacy diagnostic encoding and scale-consistent physical comparison.
    ax = fig.add_subplot(gs[2, 0])
    s_in = meta["s_input"]; nshow = min(8, s_in.shape[1])
    cols = plt.cm.coolwarm(np.linspace(0.05, 0.95, nshow))
    for i in range(nshow):
        ax.plot(x, s_in[:, i], color=cols[i], lw=0.9, alpha=0.85,
                label=f"k={meta['k_sel'][i]}")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title(r"Legacy diagnostic encoding $s_n$ (not network input)"); ax.set_xlabel("x")

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, target_phys, color="#aaaaaa", lw=1.5, ls="--",
            label=r"Physical target $\delta_n$")
    ax.plot(x, output_phys, color=color, lw=1.5, label=f"Physical output $g_{n}(x)$")
    ax.legend(fontsize=8)
    ax.set_title(f"Same-scale Output vs Target  ·  corr={corr:.6f}"); ax.set_xlabel("x")

    # Row 3: cumulative solution and pointwise error.
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(x, u_star, color="white", lw=2, label="$u^*$", zorder=3)
    ax.plot(x, u_before, color="#666", lw=1.2, ls="--", alpha=0.7, label=f"$u_{{{n-1}}}$")
    ax.plot(x, u_after, color=color, lw=1.5, label=f"$u_{{{n}}}$")
    ax.legend(fontsize=8); ax.set_title("Cumulative Solution"); ax.set_xlabel("x")

    ax = fig.add_subplot(gs[3, 1])
    eb = np.linalg.norm(u_before - u_star) / np.linalg.norm(u_star)
    ea = np.linalg.norm(u_after - u_star) / np.linalg.norm(u_star)
    ax.semilogy(x, np.abs(u_before - u_star) + 1e-12, "--",
                color="#666", lw=1.2, alpha=0.7, label=f"before {eb:.3e}")
    ax.semilogy(x, np.abs(u_after - u_star) + 1e-12,
                color=color, lw=1.5, label=f"after  {ea:.3e}")
    ax.legend(fontsize=8); ax.set_title(r"Pointwise Error $|u^*-u_n|$"); ax.set_xlabel("x")

    # Row 4: loss and weights.
    ax = fig.add_subplot(gs[4, 0])
    ax.semilogy(losses, color=color, lw=1.2, label="Total/In-band")
    ax.legend(fontsize=7); ax.set_title(f"Training Loss  (lr={lr_used:.2e})"); ax.set_xlabel("Epoch")
    ax = fig.add_subplot(gs[4, 1])
    ax.bar(meta["k_sel"], meta["alphas"], color=color, alpha=0.85, width=1.5)
    ax.set_title(r"$\alpha_k$ Weights"); ax.set_xlabel("k")

    # Row 5: signed coefficients retain phase/sign information.
    ax = fig.add_subplot(gs[5, :])
    ks = np.asarray(meta["k_sel"]); bar_w = 0.32
    ax.bar(ks - bar_w / 2, target_coef_phys, width=bar_w,
           color="#aaaaaa", alpha=0.75, label="target signed coefficient")
    ax.bar(ks + bar_w / 2, output_coef_phys, width=bar_w,
           color=color, alpha=0.75, label="output signed coefficient")
    ax.axhline(0.0, color="#666", lw=0.8)
    ax.set_xlabel("k"); ax.set_ylabel("Orthonormal coefficient (physical scale)")
    ax.legend(fontsize=8)
    ax.set_title(f"Signed Spectral Coefficients — sign match={sign_match:.0%}, rel error={coef_rel:.2e}")

    fname = OUT / f"grade_{n:02d}_diag.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show(); plt.close(fig); print(f"  [saved] {fname.name}")
    return fname

print("[*] plot_grade defined (reviewed: same-scale + signed phase diagnostics)")


# In[ ]:


def plot_summary_nls(infos, op, u_star, u_pred, beta=1.0, out_dir=None):
    if out_dir is None:
        out_dir = OUT
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)

    NG     = len(infos)
    colors = [CGRADE(i / max(NG - 0.5, 1)) for i in range(NG)]
    x      = op.x
    final_rel_l2 = infos[-1]["rel_err_after"]

    fig = plt.figure(figsize=(18, 14), facecolor="#0d0d0d")
    fig.suptitle(
        f"[MGDL_Helmholtz v8.1]  Summary  ·  β={beta:.1f}  ·  {NG} Grades"
        f"  ·  Final rel L2={final_rel_l2:.2e}",
        fontsize=13, fontweight="bold", color="white", y=0.99)
    gs_outer = gridspec.GridSpec(3, 1, fig, top=0.96, bottom=0.05,
                                 hspace=0.48, height_ratios=[1.6, 1, 1])


    ax_loss = fig.add_subplot(gs_outer[0])
    ax_loss.set_title(
        "All Grades — Continuous Training Loss  (solid=Total · dashed=In-Band · dotted=OOB)",
        fontsize=10, color="#ccc")
    epoch_offset = 0
    grade_centers = []; grade_boundaries = [0]
    for i, info in enumerate(infos):
        c  = colors[i]
        ls = np.array(info["losses"])
        li = np.array(info["losses_in"])
        lo = np.array(info["losses_oob"])
        n_ep = len(ls)
        eps  = np.arange(epoch_offset, epoch_offset + n_ep)
        ax_loss.semilogy(eps, ls, color=c, lw=1.1, alpha=0.9)
        ax_loss.semilogy(eps, li, color=c, lw=0.8, ls="--", alpha=0.65)
        ax_loss.semilogy(eps, lo, color=c, lw=0.8, ls=":",  alpha=0.55)
        grade_centers.append(epoch_offset + n_ep / 2)
        epoch_offset += n_ep
        grade_boundaries.append(epoch_offset)
    for xb in grade_boundaries[1:-1]:
        ax_loss.axvline(xb, color="#444", lw=0.8, ls="--")
    ylim = ax_loss.get_ylim()
    for i, cx in enumerate(grade_centers):
        label = f"G{i+1}" + (" ✗" if not infos[i].get("accepted", True) else "")
        ax_loss.text(cx, ylim[0], label, ha="center", va="bottom",
                     fontsize=9, color=colors[i], alpha=0.85)
    ax_loss.set_xlim(0, grade_boundaries[-1])
    ax_loss.set_xlabel(f"Epoch  (each grade = {len(infos[0]['losses'])} epochs)",
                       fontsize=9, color="#aaa")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(handles=[
        Line2D([0],[0], color="white", lw=1.5,          label="Total"),
        Line2D([0],[0], color="white", lw=1.0, ls="--", label="In-Band"),
        Line2D([0],[0], color="white", lw=1.0, ls=":",  label="OOB"),
    ], loc="upper right", fontsize=8)


    gs_mid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], wspace=0.38)

    ax_sol = fig.add_subplot(gs_mid[0])
    ax_sol.plot(x, u_star, color="white", lw=2.0, label=r"$u^*$", zorder=3)
    ax_sol.plot(x, u_pred, color="#ff66cc", lw=1.3, ls="--",
                label=rf"$u_{{approx}}$ ({final_rel_l2:.2e})")
    ax_sol.legend(fontsize=7); ax_sol.set_title("Final Solution vs True", fontsize=10)
    ax_sol.set_xlabel("x")

    ax_evo = fig.add_subplot(gs_mid[1])
    ax_evo.plot(x, u_star, color="white", lw=2.0, label=r"$u^*$", zorder=3)
    u_acc = np.zeros_like(u_star)
    for i, info in enumerate(infos):
        u_acc = u_acc + info["g_np"]
        ax_evo.plot(x, u_acc.copy(), color=colors[i], lw=1.0, alpha=0.8,
                    label=f"G{i+1}")
    ax_evo.legend(fontsize=7); ax_evo.set_title("Solution Evolution", fontsize=10)
    ax_evo.set_xlabel("x")

    ax_pw = fig.add_subplot(gs_mid[2])
    u_acc = np.zeros_like(u_star)
    for i, info in enumerate(infos):
        u_acc = u_acc + info["g_np"]
        ax_pw.semilogy(x, np.abs(u_acc - u_star) + 1e-12,
                       color=colors[i], lw=1.0, alpha=0.85, label=f"G{i+1}")
    ax_pw.legend(fontsize=7); ax_pw.set_title("Pointwise Error Evolution", fontsize=10)
    ax_pw.set_xlabel("x"); ax_pw.set_ylabel(r"$|u^* - u_n|$")


    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[2], wspace=0.38)
    grades_idx = np.arange(1, NG + 1)

    ax_l2 = fig.add_subplot(gs_bot[0])
    ea_vals = [info["rel_err_after"] for info in infos]
    ax_l2.semilogy(grades_idx, ea_vals, "o-", color="#44ddff", lw=2, ms=7, zorder=3)
    for xi, yi in zip(grades_idx, ea_vals):
        ax_l2.annotate(f"{yi:.1e}", (xi, yi),
                       textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=7.5, color="#44ddff")
    ax_l2.set_title("Rel L2 per Grade", fontsize=10)
    ax_l2.set_xlabel("Grade"); ax_l2.set_ylabel("Rel L2")
    ax_l2.set_xticks(grades_idx)

    ax_freq = fig.add_subplot(gs_bot[1])
    n_freqs = [info["n_freqs"] for info in infos]
    bars = ax_freq.bar(grades_idx, n_freqs, color=colors, alpha=0.88)
    for bar, val in zip(bars, n_freqs):
        ax_freq.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1, str(val),
                     ha="center", va="bottom", fontsize=9, color="white")
    ax_freq.set_title(r"$|K_n|$ per Grade", fontsize=10)
    ax_freq.set_xlabel("Grade"); ax_freq.set_xticks(grades_idx)

    ax_imp = fig.add_subplot(gs_bot[2])
    improvements = [(info["rel_err_before"] - info["rel_err_after"]) /
                    (info["rel_err_before"] + 1e-30) * 100
                    for info in infos]
    bar_colors = ["#ff4455" if v < 0 else "#00cc88" for v in improvements]
    bars = ax_imp.bar(grades_idx, improvements, color=bar_colors, alpha=0.88)
    for bar, val in zip(bars, improvements):
        ax_imp.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), 0) + 0.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=9, color="white")
    ax_imp.set_title("Rel L2 Improvement per Grade", fontsize=10)
    ax_imp.set_xlabel("Grade"); ax_imp.set_ylabel("%")
    ax_imp.set_xticks(grades_idx)
    ax_imp.axhline(0, color="#555", lw=0.8)

    fname = out_dir / "summary_nls.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig); print(f"  [saved] {fname}")
    return fname

print("[*] plot_summary_nls defined")


# In[ ]:


# Cell 9 — plot_all_grades_metrics


def plot_all_grades_metrics(infos, out_dir=None):
    if out_dir is None:
        out_dir = OUT
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)

    NG     = len(infos)
    colors = [CGRADE(i / max(NG - 0.5, 1)) for i in range(NG)]

    offsets = [0]
    for info in infos:
        offsets.append(offsets[-1] + len(info["losses"]))

    def _decorate(ax, infos, offsets, colors):
        ylim = ax.get_ylim()
        for i, info in enumerate(infos):
            x0, x1 = offsets[i], offsets[i+1]
            cx = (x0 + x1) / 2
            if i > 0:
                ax.axvline(x0, color="#444", lw=0.8, ls="--")
            if not info.get("accepted", True):
                ax.axvspan(x0, x1, color="#ff2244", alpha=0.07)
                ax.text(cx, ylim[0], f"G{i+1}\n✗",
                        ha="center", va="bottom", fontsize=8,
                        color="#ff6677", alpha=0.9)
            else:
                ax.text(cx, ylim[0], f"G{i+1}",
                        ha="center", va="bottom", fontsize=8,
                        color=colors[i], alpha=0.85)
        ax.set_ylim(ylim)

    saved = []


    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d0d0d")
    fig.suptitle(f"[All Grades]  Training Loss  ·  {NG} Grades",
                 fontsize=12, color="white", fontweight="bold")
    ax.set_title("solid=Total · dashed=In-Band · dotted=OOB",
                 fontsize=9, color="#ccc")
    for i, info in enumerate(infos):
        c  = colors[i]
        ep = np.arange(offsets[i], offsets[i+1])
        ax.semilogy(ep, info["losses"],     color=c, lw=1.0, alpha=0.9)
        ax.semilogy(ep, info["losses_in"],  color=c, lw=0.7, ls="--", alpha=0.6)
        ax.semilogy(ep, info["losses_oob"], color=c, lw=0.7, ls=":",  alpha=0.5)
    ax.set_xlim(0, offsets[-1])
    ax.set_xlabel("Global Epoch"); ax.set_ylabel("Loss (log)")
    ax.legend(handles=[
        Line2D([0],[0],color="white",lw=1.2,label="Total"),
        Line2D([0],[0],color="white",lw=0.8,ls="--",label="In-Band"),
        Line2D([0],[0],color="white",lw=0.8,ls=":",label="OOB"),
    ], fontsize=8, loc="upper right")
    _decorate(ax, infos, offsets, colors)
    fname = out_dir / "all_grades_loss.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig); print(f"  [saved] {fname.name}")
    saved.append(fname)


    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d0d0d")
    fig.suptitle(
        f"[All Grades]  Normalized Training Loss  ·  {NG} Grades",
        fontsize=12, color="white", fontweight="bold")
    ax.set_title(
        r"$\tilde{L}_n(\mathrm{ep}) = L_n(\mathrm{ep})\,/\,L_n(0)$"
        "  —  unified scale, each grade starts at 1",
        fontsize=9, color="#ccc")
    for i, info in enumerate(infos):
        c  = colors[i]
        ls = np.array(info["losses"],     dtype=np.float64)
        li = np.array(info["losses_in"],  dtype=np.float64)
        lo = np.array(info["losses_oob"], dtype=np.float64)
        l0  = ls[0] + 1e-30
        li0 = li[0] + 1e-30
        lo0 = lo[0] + 1e-30
        ep = np.arange(offsets[i], offsets[i+1])
        ax.semilogy(ep, ls / l0, color=c, lw=1.0, alpha=0.9)
        ax.semilogy(ep, li / li0, color=c, lw=0.7, ls="--", alpha=0.6)
        if np.any(lo > 0):
            ax.semilogy(ep, lo / lo0, color=c, lw=0.7, ls=":", alpha=0.5)

        ax.annotate(f"{ls[-1]/l0:.1e}",
                    (ep[-1], ls[-1]/l0),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=7, color=c)
    ax.axhline(1.0, color="#555", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlim(0, offsets[-1])
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel(r"$\tilde{L}_n$ (log)")
    ax.legend(handles=[
        Line2D([0],[0],color="white",lw=1.2,label="Total"),
        Line2D([0],[0],color="white",lw=0.8,ls="--",label="In-Band"),
        Line2D([0],[0],color="white",lw=0.8,ls=":",label="OOB"),
        Line2D([0],[0],color="#555",lw=0.8,ls=":",label="start = 1"),
    ], fontsize=8, loc="upper right")
    _decorate(ax, infos, offsets, colors)
    fname = out_dir / "all_grades_loss_normalized.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig); print(f"  [saved] {fname.name}")
    saved.append(fname)


    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d0d0d")
    fig.suptitle(f"[All Grades]  RSE (Relative Solution Error)  ·  {NG} Grades",
                 fontsize=12, color="white", fontweight="bold")
    ax.set_title(r"$\|u_{pred}-u^*\|_2 / \|u^*\|_2$  per epoch",
                 fontsize=9, color="#ccc")
    for i, info in enumerate(infos):
        rse = info.get("rse_history", [])
        if not rse: continue
        c  = colors[i]
        ep = np.arange(offsets[i], offsets[i] + len(rse))
        ax.semilogy(ep, rse, color=c, lw=1.0, alpha=0.9)
        best_idx = int(np.argmin(rse))
        ax.scatter(offsets[i] + best_idx, rse[best_idx], color=c, s=35, zorder=5)
        ax.annotate(f"{rse[best_idx]:.2e}",
                    (offsets[i] + best_idx, rse[best_idx]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color=c)
    ax.set_xlim(0, offsets[-1])
    ax.set_xlabel("Global Epoch"); ax.set_ylabel("RSE (log)")
    _decorate(ax, infos, offsets, colors)
    fname = out_dir / "all_grades_rse.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig); print(f"  [saved] {fname.name}")
    saved.append(fname)


    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0d0d0d")
    fig.suptitle(f"[All Grades]  Training Wall Time  ·  {NG} Grades",
                 fontsize=12, color="white", fontweight="bold")
    ax.set_title("Cumulative seconds from each grade's start",
                 fontsize=9, color="#ccc")
    for i, info in enumerate(infos):
        t = info.get("time_history", [])
        if not t: continue
        c  = colors[i]
        ep = np.arange(offsets[i], offsets[i] + len(t))
        ax.plot(ep, t, color=c, lw=1.0, alpha=0.9)
        ax.annotate(f"{t[-1]:.1f}s", (ep[-1], t[-1]),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=7, color=c)
    ax.set_xlim(0, offsets[-1])
    ax.set_xlabel("Global Epoch"); ax.set_ylabel("Elapsed (s)")
    _decorate(ax, infos, offsets, colors)
    fname = out_dir / "all_grades_time.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig); print(f"  [saved] {fname.name}")
    saved.append(fname)


    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    fig.suptitle(f"[All Grades]  PINN Loss per Grade  ·  {NG} Grades",
                 fontsize=12, color="white", fontweight="bold")
    ax.set_title(r"$(1/N)\|L[u_n] - f\|^2$ after each grade",
                 fontsize=9, color="#ccc")
    grades_idx = np.arange(1, NG + 1)
    pinn_vals  = [info.get("pinn_loss", float("nan")) for info in infos]
    ax.semilogy(grades_idx, pinn_vals, "o-", color="#44ddff", lw=2, ms=8, zorder=3)
    for xi, yi in zip(grades_idx, pinn_vals):
        if not np.isnan(yi):
            ax.annotate(f"{yi:.2e}", (xi, yi),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color="#44ddff")
    ax.set_xticks(grades_idx)
    ax.set_xlabel("Grade"); ax.set_ylabel(r"$(1/N)\|L[u_n]-f\|^2$ (log)")
    ax.grid(True, alpha=0.2)
    fname = out_dir / "pinn_loss_per_grade.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig); print(f"  [saved] {fname.name}")
    saved.append(fname)

    return saved

print("[*] plot_all_grades_metrics defined (v9+: raw + normalized loss)")


# In[ ]:


# Cell 10 — HelmholtzGradeSolver  (v9 patch)
#
# v9 vs v8.1：


class HelmholtzGradeSolver:
    def __init__(self, op, f, u_star,
                 base_width        = 64,
                 base_depth        = 3,
                 lambda_oob        = 0.0,
                 gd_floor          = 0.01,
                 stop_delta_H      = 0.001,
                 stop_min_improve  = 0.20,
                 base_lr           = 1e-3,
                 lr_min            = 1e-4,
                 lr_max            = 5e-3,
                 ledger_min_improve = 0.30,
                 ledger_patience    = 2,
                 difficulty_eta     = 1.0,
                 mem_feat_cap       = 10.0,
                 device            = "cpu"):
        self.op               = op
        self.f                = f
        self.u_star           = u_star
        self.base_width       = base_width
        self.base_depth       = base_depth
        self.lambda_oob       = lambda_oob
        self.gd_floor         = gd_floor
        self.stop_delta_H     = stop_delta_H
        self.stop_min_improve = stop_min_improve
        self.device           = torch.device(device)
        self.grades           = []
        self.infos            = []
        self._u               = np.zeros(op.N, np.float32)
        self._next_residual   = None
        self.lr_adaptor       = LrAdaptor(
            base_lr=base_lr, lr_min=lr_min, lr_max=lr_max)


        self.freq_ledger = dict(
            times_selected  = np.zeros(op.N, dtype=int),
            stall_count     = np.zeros(op.N, dtype=int),
            quarantined     = np.zeros(op.N, dtype=bool),
            last_error_phys = np.zeros(op.N, dtype=np.float64),   # v14-B
            last_fit_err    = np.zeros(op.N, dtype=np.float64),   # v14-A
        )
        self.ledger_min_improve = ledger_min_improve
        self.ledger_patience    = ledger_patience
        self.difficulty_eta     = difficulty_eta
        self.mem_feat_cap       = mem_feat_cap


        self._last_skip_gain = 0.1
        self._last_mem_gain  = 1.0


    def _update_freq_ledger(self, k_sel, amps_before, amps_after):
        """Update freq ledger."""
        ledger = self.freq_ledger
        newly_quarantined = []
        for k in k_sel:
            idx = int(k) - 1
            ab = float(amps_before[idx]); aa = float(amps_after[idx])
            improve_ratio = aa / (ab + 1e-12)
            if ledger['times_selected'][idx] > 0:
                if improve_ratio > (1.0 - self.ledger_min_improve):
                    ledger['stall_count'][idx] += 1
                else:
                    ledger['stall_count'][idx] = 0
                if (not ledger['quarantined'][idx]) and \
                   ledger['stall_count'][idx] >= self.ledger_patience:
                    ledger['quarantined'][idx] = True
                    newly_quarantined.append(int(k))
            ledger['times_selected'][idx] += 1
        return newly_quarantined


    def _compute_history_gate(self, k_sel):
        """Compute history gate."""
        stall = self.freq_ledger['stall_count'][np.asarray(k_sel, dtype=int) - 1]
        return (0.5 ** stall).astype(np.float32)


    def _config(self, H, nk, rel_err):
        depth = max(2, min(self.base_depth + int(H * 2.0), 10))
        if rel_err < 1e-3: width_cap = 256
        elif rel_err < 1e-2: width_cap = 512
        else: width_cap = 1024
        width = min(max(self.base_width, 4 * nk), width_cap)
        return width, depth


    def _should_stop(self):
        if len(self.infos) < 2:
            return False
        eb = self.infos[-1]["rel_err_before"]
        ea = self.infos[-1]["rel_err_after"]
        improve_rate = (eb - ea) / (eb + 1e-30)
        prev_eb = self.infos[-2]["rel_err_before"]
        prev_ea = self.infos[-2]["rel_err_after"]
        prev_impr = (prev_eb - prev_ea) / (prev_eb + 1e-30)


        cond_A = (improve_rate < 0.10) and (prev_impr < 0.10)

        cond_B = (improve_rate < self.stop_min_improve) and (prev_impr < self.stop_min_improve)

        cond_C = (improve_rate < 0) and (prev_impr < 0)

        stop = cond_A or cond_B or cond_C
        if stop:
            reasons = []
            if cond_A: reasons.append(f"RSE stagnant: {improve_rate*100:.1f}% & {prev_impr*100:.1f}% < 10%")
            if cond_B: reasons.append(f"low improve: {improve_rate*100:.1f}% & {prev_impr*100:.1f}% < {self.stop_min_improve*100:.0f}%")
            if cond_C: reasons.append("consecutive divergence")
            print(f"\n  [EARLY STOP v9]  " + "  |  ".join(reasons))
        return stop


    def add_grade(self, n_epochs=1500):
        n = len(self.grades) + 1
        print(f"\n{'═'*60}\n  GRADE {n}\n{'═'*60}")


        if self._next_residual is not None:
            e_n = self._next_residual; self._next_residual = None
            print(f"  [v9] Pre-computed residual from grade {n-1}")
        else:
            e_n = self.op.residual(self._u, self.f)
        target       = -self.op.inv_L(e_n)
        target_scale = float(np.linalg.norm(target)) + 1e-12
        target_norm  = (target / target_scale).astype(np.float32)
        rel_b  = float(np.linalg.norm(self._u - self.u_star) /
                       np.linalg.norm(self.u_star))
        pinn_b = float(np.linalg.norm(e_n) ** 2) / self.op.N
        print(f"  Residual L2  : {np.linalg.norm(e_n):.4e}")
        print(f"  Rel L2 error : {rel_b:.4e}")
        print(f"  PINN loss(before): {pinn_b:.4e}")


        if n > 1 and self.infos:
            lr = self.lr_adaptor.update(self.infos[-1]["losses"], grade_n=n-1)
        else:
            lr = self.lr_adaptor.base_lr
        print(f"  lr (adaptive): {lr:.2e}")


        n_quarantined = int(self.freq_ledger['quarantined'].sum())
        if n_quarantined > 0:
            print(f"  [ledger] active quarantine: {n_quarantined} freq(s) excluded from candidates")
        feat_np, meta = build_spectral_input(e_n, target_norm, self.op,
                                              gd_floor=self.gd_floor,
                                              exclude_mask=self.freq_ledger['quarantined'],
                                              select_by="target")
        meta['target_scale'] = target_scale


        hist_gate = self._compute_history_gate(meta["k_sel"])
        feat_np[:, 3] = hist_gate
        meta['hist_gate'] = hist_gate


        #   target_coef_now[k] = -(target_scale_prev/target_scale_now) · err
        #                      = -last_error_phys[k] / target_scale_now


        k_idx = meta["k_sel"].astype(int) - 1
        mem_feat = -self.freq_ledger['last_error_phys'][k_idx] / (target_scale + 1e-12)
        mem_feat = np.clip(mem_feat, -self.mem_feat_cap, self.mem_feat_cap).astype(np.float32)
        feat_np[:, 4] = mem_feat
        meta['mem_feat'] = mem_feat
        n_with_memory = int((self.freq_ledger['times_selected'][k_idx] > 0).sum())


        difficulty = self.freq_ledger['last_fit_err'][k_idx]
        boost = 1.0 + self.difficulty_eta * np.clip(difficulty, 0.0, 3.0)
        alphas_boosted = meta["alphas"] * boost
        alphas_boosted = (alphas_boosted / (alphas_boosted.sum() + 1e-12)).astype(np.float32)
        meta['alphas_raw'] = meta['alphas']
        meta['alphas'] = alphas_boosted
        meta['difficulty_boost'] = boost

        width, depth = self._config(meta["entropy"], meta["n_freqs"], rel_b)
        print(f"  |K_n|={meta['n_freqs']}  H={meta['entropy']:.3f}  "
              f"w={width}  d={depth}  "
              f"gated={int((hist_gate < 1.0).sum())}  "
              f"mem={n_with_memory}/{meta['n_freqs']}  "
              f"skip0={self._last_skip_gain:.3f}  mem0={self._last_mem_gain:.3f}")


        net           = GradeNet(meta["n_freqs"], feat_np.shape[1],
                                  width, depth,
                                  skip_gain_init=self._last_skip_gain,
                                  mem_gain_init=self._last_mem_gain).to(self.device)
        feat_t        = torch.tensor(feat_np, dtype=torch.float32).to(self.device)
        target_coef_t = torch.tensor(meta["target_coef"], dtype=torch.float32).to(self.device)
        criterion     = SpectralLoss(alphas=meta["alphas"]).to(self.device)
        opt  = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        max_ep = n_epochs * 2 if meta['n_freqs'] > 20 else n_epochs


        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=300,
            min_lr=lr * 5e-3)


        lr_floor = max(lr * 5e-3, 1e-6)
        max_backtrack = 6
        rel_tol = 1e-6

        def _synth_physical(c_np):
            """Synth physical."""
            return meta["basis"] @ c_np   # (N,)

        def _forward_eval():
            """Forward eval."""
            with torch.no_grad():
                c_out = net(feat_t).squeeze(-1)
                loss_t, l_in, l_oob = criterion(c_out, target_coef_t)
                c_np = c_out.cpu().numpy()
            g_np_ep = _synth_physical(c_np) * target_scale
            rse = float(np.linalg.norm(self._u + g_np_ep - self.u_star) /
                        np.linalg.norm(self.u_star))
            return float(loss_t.item()), l_in, l_oob, rse

        def _snapshot():
            """Snapshot."""
            return ({k: v.detach().clone() for k, v in net.state_dict().items()},
                    copy.deepcopy(opt.state_dict()))

        best_loss = float('inf')
        best_rse  = float('inf')
        plateau_cnt = 0
        log_every = 1000 if max_ep > 3000 else 500
        losses = []; losses_in = []; losses_oob = []
        rse_history = []; time_history = []
        t0 = time.perf_counter()


        net.eval()
        prev_loss, prev_l_in, prev_l_oob, prev_rse = _forward_eval()
        best_loss, best_rse = prev_loss, prev_rse
        prev_net_state, prev_opt_state = _snapshot()

        net.train()
        for ep in range(max_ep):
            accepted_step = False
            lv = l_in = l_oob = rse_ep = None
            for attempt in range(max_backtrack + 1):
                opt.zero_grad()
                c_out = net(feat_t).squeeze(-1)
                loss, _, _ = criterion(c_out, target_coef_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

                net.eval()
                lv, l_in, l_oob, rse_ep = _forward_eval()
                net.train()

                ok_loss = lv     <= prev_loss * (1 + rel_tol)
                ok_rse  = rse_ep <= prev_rse  * (1 + rel_tol)
                if ok_loss and ok_rse:
                    accepted_step = True
                    break


                cur_lr = opt.param_groups[0]['lr']
                net.load_state_dict(prev_net_state)
                opt.load_state_dict(prev_opt_state)
                for pg in opt.param_groups:
                    pg['lr'] = max(cur_lr * 0.5, lr_floor)

            if not accepted_step:


                cur_lr = opt.param_groups[0]['lr']
                net.load_state_dict(prev_net_state)
                opt.load_state_dict(prev_opt_state)
                for pg in opt.param_groups:
                    pg['lr'] = cur_lr
                lv, l_in, l_oob, rse_ep = prev_loss, prev_l_in, prev_l_oob, prev_rse

            sched.step(lv)

            losses.append(lv); losses_in.append(l_in); losses_oob.append(l_oob)
            rse_history.append(rse_ep)
            time_history.append(time.perf_counter() - t0)


            prev_loss, prev_l_in, prev_l_oob, prev_rse = lv, l_in, l_oob, rse_ep
            prev_net_state, prev_opt_state = _snapshot()


            if lv < best_loss * (1 - 5e-5):
                best_loss = lv; plateau_cnt = 0
            else:
                plateau_cnt += 1
            if rse_ep < best_rse:
                best_rse = rse_ep
            if plateau_cnt >= 600 and ep > max_ep // 4:
                print(f'  [plateau] ep{ep+1}  loss={lv:.4e} stop'); break

            if (ep + 1) % log_every == 0:
                print(f'  ep{ep+1:5d}  total={lv:.4e}  in={l_in:.4e}  '
                      f'oob={l_oob:.4e}  RSE={rse_ep:.4e}  bestRSE={best_rse:.4e}'
                      f'  (accepted={accepted_step})')


        net.eval()
        print(f"  [v10] Monotone training done  final loss={losses[-1]:.4e}  "
              f"final RSE={rse_history[-1]:.4e}")

        u_before = self._u.copy()
        net.eval()
        with torch.no_grad():
            c_final = net(feat_t).squeeze(-1).cpu().numpy()
        g_np = _synth_physical(c_final) * target_scale
        u_candidate = self._u + g_np
        rel_a  = float(np.linalg.norm(u_candidate - self.u_star) /
                       np.linalg.norm(self.u_star))
        pinn_a = float(np.linalg.norm(self.op.residual(u_candidate, self.f)) ** 2) / self.op.N


        if rel_a < rel_b and pinn_a <= pinn_b:
            self._u = u_candidate
            accepted = True
            arrow = f"↓{abs(rel_a-rel_b)/rel_b*100:.1f}%"
        else:
            accepted = False
            reasons = []
            if not (rel_a < rel_b):   reasons.append("RSE worsened")
            if not (pinn_a <= pinn_b): reasons.append("PINN loss worsened")
            arrow = f"[REJECTED — u unchanged: {' & '.join(reasons)}]"
        rel_actual = float(np.linalg.norm(self._u - self.u_star) /
                           np.linalg.norm(self.u_star))
        print(f"  Rel L2 candidate: {rel_a:.4e}  ({arrow})")
        print(f"  Rel L2 actual   : {rel_actual:.4e}")


        pinn_res    = self.op.residual(self._u, self.f)
        pinn_loss_n = float(np.linalg.norm(pinn_res) ** 2) / self.op.N
        print(f"  PINN loss    : {pinn_loss_n:.4e}  (before={pinn_b:.4e}  candidate={pinn_a:.4e})")


        err_now = c_final - meta["target_coef"]
        if accepted:
            self.freq_ledger['last_error_phys'][k_idx] = err_now * target_scale
            self.freq_ledger['last_fit_err'][k_idx] = np.abs(err_now)
            self._last_skip_gain = float(net.skip_gain.detach().cpu())
            self._last_mem_gain = float(net.mem_gain.detach().cpu())
        print(f"  [v14] mem_gain={self._last_mem_gain:.4f}  "
              f"skip_gain={self._last_skip_gain:.4f}  "
              f"mean|err|={np.abs(err_now).mean():.4e}  memory_updated={accepted}")

        net.requires_grad_(False)
        print(f"  Grade {n} FROZEN.")

        # e_{{n+1}}
        e_next = self.op.residual(self._u, self.f)
        self._next_residual = e_next
        next_mask, next_ksel, next_ehat, amps_next, next_tau = compute_Kn(
            e_next, self.op, self.gd_floor, exclude_mask=self.freq_ledger['quarantined'])
        print(f"  [v9] e_{{n+1}} L2={np.linalg.norm(e_next):.4e}  "
              f"|K_{{n+1}}|={next_ksel.shape[0]}")


        newly_q = self._update_freq_ledger(meta["k_sel"], meta["amps_res"], amps_next)
        if newly_q:
            print(f"  [ledger] quarantined {len(newly_q)} freq(s) "
                  f"(stalled >= {self.ledger_patience}x, "
                  f"<{self.ledger_min_improve*100:.0f}% improve): {newly_q}")
        print(f"  [ledger] total quarantined so far: "
              f"{int(self.freq_ledger['quarantined'].sum())}")

        info = {
            **meta,
            "depth": depth, "width": width,
            "losses": losses, "losses_in": losses_in, "losses_oob": losses_oob,
            "rse_history":  rse_history,
            "time_history": time_history,
            "g_candidate": g_np,
            "g_np": g_np if accepted else np.zeros_like(g_np),
            "rel_err_before": rel_b, "rel_err_after": rel_actual,
            "rel_err_candidate": rel_a,
            "pinn_before": pinn_b, "pinn_after": pinn_loss_n,
            "pinn_candidate": pinn_a,   # v14-D
            "lr_used": lr,
            "accepted": accepted,
            "pinn_loss": pinn_loss_n,
            "n_quarantined_before": n_quarantined,          # v11
            "n_quarantined_after": int(self.freq_ledger['quarantined'].sum()),
            "newly_quarantined": newly_q,
        }
        self.grades.append(net)
        self.infos.append(info)
        plot_grade(n, self.op, info, g_np, u_before, self._u, self.u_star,
                   losses, losses_in, losses_oob, lr, accepted)
        return info

    def solve(self, n_grades=3, n_epochs=1500):
        t_total_start = time.perf_counter()
        for i in range(n_grades):
            self.add_grade(n_epochs)
            if self._should_stop():
                print(f"  Stopping after {i+1} grades.")
                break
        t_total = time.perf_counter() - t_total_start
        self._total_time = t_total
        plot_summary_nls(self.infos, self.op, self.u_star, self._u, beta=1.0)
        plot_all_grades_metrics(self.infos)
        self._table()

    def _table(self):
        print(f"\n{'═'*100}")
        print(f"{'G':^3}  {'|Kn|':^5}  {'H':^6}  {'w':^4}  {'d':^3}  "
              f"{'lr':^8}  {'err_before':^11}  {'err_after':^11}  "
              f"{'Δ%':^8}  {'acc':^5}  {'frozen':^6}")
        print("─" * 100)
        for i, info in enumerate(self.infos):
            eb, ea = info["rel_err_before"], info["rel_err_after"]
            frozen = "✓" if not any(
                p.requires_grad for p in self.grades[i].parameters()) else "✗"
            acc = "✓" if info.get("accepted", True) else "✗"
            print(f" {i+1:^3}  {info['n_freqs']:^5}  {info['entropy']:^6.3f}  "
                  f"{info['width']:^4}  {info['depth']:^3}  {info['lr_used']:^8.2e}  "
                  f"{eb:^11.4e}  {ea:^11.4e}  {(eb-ea)/eb*100:^+7.1f}%  "
                  f"{acc:^5}  {frozen:^6}")
        print("═" * 100)
        print(f"Final rel L2 : {self.infos[-1]['rel_err_after']:.4e}")
        total_t = getattr(self, '_total_time', None)
        if total_t is not None:
            m, s = divmod(total_t, 60)
            print(f"Total wall time: {int(m):02d}m {s:.1f}s  ({total_t:.1f}s)")

print("[*] HelmholtzGradeSolver defined  (v14: +A difficulty-weighted alphas, +B mem_feat, +C warm-start)")


# In[ ]:


if __name__ == "__main__" or True:
    np.random.seed(20260728)
    torch.manual_seed(20260728)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(20260728)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this script, but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver and install a CUDA-enabled PyTorch build."
        )
    device = torch.device("cuda")
    print(f"[*] device={device}")


    N = 256; kappa = 200.0*torch.pi
    op     = HelmholtzSpectralOperator(N, kappa)
    x      = op.x
    u_star = (np.sin(3*np.pi*x) + 0.5*np.sin(7*np.pi*x)
              + 0.3*np.sin(11*np.pi*x)).astype(np.float32)
    f      = op.make_f(u_star)


    _e = (np.linalg.norm(-op.inv_L(op.residual(np.zeros(N, np.float32), f)) - u_star)
          / np.linalg.norm(u_star))
    print(f"[check] inv_L err={_e:.2e}  (OK if < 1e-5)")
    null_modes = op.k_all[np.abs(op.lam_k) < 10.0]
    if len(null_modes):
        print(f"[warning] resonant/null modes {null_modes.tolist()} are projected out by inv_L; "
              "the PDE is non-unique unless a zero-nullspace gauge is imposed.")


    solver = HelmholtzGradeSolver(
        op, f, u_star,
        base_width        = 64,
        base_depth        = 3,
        lambda_oob        = 0.1,
        gd_floor          = 0.01,
        stop_delta_H      = 0.001,
        stop_min_improve  = 0.01,
        base_lr           = 3e-3,
        lr_min            = 5e-5,
        lr_max            = 1e-2,
        ledger_min_improve = 0.30,
        ledger_patience    = 2,
        device            = str(device),
    )
    solver.solve(n_grades=3, n_epochs=1000)
