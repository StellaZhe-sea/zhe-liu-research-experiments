import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)
torch.manual_seed(20260729)
np.random.seed(20260729)

OUT = Path("./helmholtz_out_recurrent_dst_fno")
OUT.mkdir(exist_ok=True)


def dst1_ortho(x):
    """Differentiable orthonormal DST-I along dimension 1.

    x: (batch, N, channels) or (batch, N)
    """
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(-1)
    batch, n, channels = x.shape
    z = torch.zeros(batch, 1, channels, dtype=x.dtype, device=x.device)
    odd = torch.cat([z, x, z, -torch.flip(x, dims=[1])], dim=1)
    spectrum = torch.fft.fft(odd, dim=1)
    coeff = -spectrum.imag[:, 1:n + 1] / math.sqrt(2.0 * (n + 1))
    return coeff.squeeze(-1) if squeeze else coeff


def idst1_ortho(x_hat):
    # The orthonormal DST-I matrix is symmetric and self-inverse.
    return dst1_ortho(x_hat)


class Helmholtz1D:
    def __init__(self, n=256, kappa=200.0 * math.pi, device="cpu"):
        self.n = n
        self.device = torch.device(device)
        self.x = torch.arange(1, n + 1, device=self.device) / (n + 1)
        self.k = torch.arange(1, n + 1, device=self.device)
        self.kappa = float(kappa)
        self.lam = (math.pi * self.k) ** 2 - self.kappa ** 2
        self.null_mask = self.lam.abs() < 10.0

    def apply(self, u):
        return idst1_ortho(dst1_ortho(u) * self.lam.unsqueeze(0))

    def residual(self, u, f):
        return self.apply(u) - f

    def shifted_precondition(self, residual, shift_ratio=0.02):
        """Tikhonov/shifted-Laplacian correction in physical coordinates.

        Approximates -L^{-1}r while remaining bounded at resonance.
        """
        r_hat = dst1_ortho(residual)
        sigma = shift_ratio * self.kappa ** 2
        gain = -self.lam / (self.lam.square() + sigma ** 2)
        gain = torch.where(self.null_mask, torch.zeros_like(gain), gain)
        return idst1_ortho(r_hat * gain.unsqueeze(0))

    def null_coefficient(self, u):
        u_hat = dst1_ortho(u)
        if not self.null_mask.any():
            return torch.zeros(u.shape[0], dtype=u.dtype, device=u.device)
        return u_hat[:, self.null_mask].square().sum(dim=1).sqrt()


class DSTFNOBlock(nn.Module):
    def __init__(self, n, width):
        super().__init__()
        self.spectral_weight = nn.Parameter(
            0.02 * torch.randn(n, width, width) / math.sqrt(width))
        self.local = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()

    def forward(self, v):
        v_hat = dst1_ortho(v)
        spectral = torch.einsum("bki,kio->bko", v_hat, self.spectral_weight)
        spectral = idst1_ortho(spectral)
        update = self.local(v) + spectral
        return self.norm(v + self.act(update))


class RecurrentPhysicalDSTFNO(nn.Module):
    """Shared recurrent backbone; no hidden feature is passed across grades."""
    def __init__(self, n, n_grades=6, in_channels=7, width=20, depth=3):
        super().__init__()
        self.n_grades = n_grades
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([DSTFNOBlock(n, width) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)
        # Start from a stable preconditioned Richardson iteration.
        self.precond_gain = nn.Parameter(torch.tensor(1.0))
        self.step_logits = nn.Parameter(torch.full((n_grades,), 1.4))

    def forward(self, features, p_normalized, grade):
        v = self.lift(features)
        for block in self.blocks:
            v = block(v)
        learned = self.project(v).squeeze(-1)
        correction_normalized = learned + self.precond_gain * p_normalized
        eta = torch.sigmoid(self.step_logits[grade])
        return eta * correction_normalized, eta


def rms(x):
    return torch.sqrt(torch.mean(x.square()) + 1e-30)


def residual_driven_gate(preconditioned, floor_ratio=0.01):
    """Hard adaptive output gate derived only from the current physical state."""
    amp = dst1_ortho(preconditioned).abs()
    tau_floor = floor_ratio * amp.amax(dim=1, keepdim=True)
    tau_noise = 3.0 * amp.median(dim=1, keepdim=True).values
    tau = torch.maximum(tau_floor, tau_noise)
    gate = (amp > tau).to(preconditioned.dtype)
    # Always keep at least the strongest frequency.
    strongest = amp.argmax(dim=1, keepdim=True)
    gate.scatter_(1, strongest, 1.0)
    return gate.detach(), tau.detach()


def build_features(op, u, residual, f, precond, grade, n_grades):
    u_scale = rms(u).detach().clamp_min(1e-10)
    r_scale = rms(residual).detach().clamp_min(1e-10)
    f_scale = rms(f).detach().clamp_min(1e-10)
    p_scale = rms(precond).detach().clamp_min(1e-12)
    x_channel = 2.0 * op.x.unsqueeze(0) - 1.0
    kappa_channel = torch.full_like(x_channel, op.kappa / (256.0 * math.pi))
    grade_channel = torch.full_like(x_channel, grade / max(n_grades - 1, 1))
    features = torch.stack([
        x_channel,
        u / u_scale,
        residual / r_scale,
        f / f_scale,
        precond / p_scale,
        kappa_channel,
        grade_channel,
    ], dim=-1)
    return features, precond / p_scale, p_scale


def unroll(model, op, f, return_history=False):
    u = torch.zeros_like(f)
    f_scale = rms(f).detach()
    history = []
    total_loss = torch.zeros((), dtype=f.dtype, device=f.device)
    grade_weights = torch.linspace(0.5, 2.0, model.n_grades, device=f.device)
    grade_weights = grade_weights / grade_weights.sum()

    for grade in range(model.n_grades):
        residual_before = op.residual(u, f)
        p = op.shifted_precondition(residual_before)
        features, p_normalized, p_scale = build_features(
            op, u, residual_before, f, p, grade, model.n_grades)
        correction_normalized, eta = model(features, p_normalized, grade)
        adaptive_gate, gate_tau = residual_driven_gate(p)
        correction_hat = dst1_ortho(correction_normalized)
        correction_normalized = idst1_ortho(correction_hat * adaptive_gate)
        correction = correction_normalized * p_scale
        u_next = u + correction
        residual_after = op.residual(u_next, f)

        pde = torch.mean((residual_after / f_scale).square())
        pde_before = torch.mean((residual_before / f_scale).square())
        corr_hat = dst1_ortho(correction)
        h1_weight = 1.0 + (math.pi * op.k) ** 2
        h1_weight = h1_weight / h1_weight.mean()
        h1_step = torch.mean(corr_hat.square() * h1_weight.unsqueeze(0))
        null_penalty = op.null_coefficient(u_next).square().mean()
        monotone_penalty = torch.relu(pde - pde_before)

        grade_loss = (
            pde
            + 1e-7 * h1_step
            + 10.0 * null_penalty
            + 10.0 * monotone_penalty
        )
        total_loss = total_loss + grade_weights[grade] * grade_loss
        if return_history:
            history.append({
                "grade": grade + 1,
                "u": u_next.detach().clone(),
                "residual": residual_after.detach().clone(),
                "preconditioned_residual": p.detach().clone(),
                "selected_k": (
                    torch.nonzero(adaptive_gate.squeeze(0), as_tuple=False)
                    .squeeze(-1).cpu().numpy() + 1).tolist(),
                "gate_tau": float(gate_tau.squeeze()),
                "pinn": float(torch.mean(residual_after.square()).detach()),
                "relative_residual": float(torch.sqrt(pde).detach()),
                "eta": float(eta.detach()),
                "null": float(op.null_coefficient(u_next).detach()),
            })
        u = u_next
    return u, total_loss, history


def adaptive_frequency_set(signal, floor_ratio=0.01):
    amp = dst1_ortho(signal).abs().squeeze(0).detach().cpu().numpy()
    threshold = max(np.median(amp) * 3.0, floor_ratio * amp.max())
    selected = np.flatnonzero(amp > threshold) + 1
    if selected.size == 0:
        selected = np.array([int(np.argmax(amp)) + 1])
    return selected, amp, threshold


def plot_results(op, u_star, history, truth_k):
    x = op.x.cpu().numpy()
    u_true = u_star.squeeze(0).cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(x, u_true, "k", lw=2, label="truth")
    for item in history:
        axes[0, 0].plot(x, item["u"].squeeze(0).cpu(), lw=1,
                        label=f"G{item['grade']}")
    axes[0, 0].set_title("Physical-space recurrent solution")
    axes[0, 0].legend(ncol=2)

    grades = [h["grade"] for h in history]
    pinn = [h["pinn"] for h in history]
    axes[0, 1].semilogy(grades, pinn, "o-")
    axes[0, 1].set_title("Actual PINN loss per grade")
    axes[0, 1].set_xlabel("grade")

    selection_matrix = np.zeros((len(history), op.n))
    labels = []
    for i, item in enumerate(history):
        selected = np.asarray(item["selected_k"], dtype=int)
        selection_matrix[i, selected - 1] = 1.0
        labels.append(selected.tolist())
    axes[1, 0].imshow(selection_matrix, aspect="auto", origin="lower",
                      extent=[1, op.n, 1, len(history)], cmap="magma")
    for k in truth_k:
        axes[1, 0].axvline(k, color="cyan", lw=0.5, alpha=0.6)
    axes[1, 0].set_title("Adaptive preconditioned-residual frequencies (cyan=true modes)")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("grade")

    final_err = history[-1]["u"].squeeze(0).cpu().numpy() - u_true
    axes[1, 1].semilogy(x, np.abs(final_err) + 1e-18)
    axes[1, 1].set_title("Final pointwise error")
    fig.tight_layout()
    fname = OUT / "recurrent_dst_fno_summary.png"
    fig.savefig(fname, dpi=220)
    plt.close(fig)
    return fname, labels


def main():
    device = torch.device("cpu")
    op = Helmholtz1D(n=256, kappa=200.0 * math.pi, device=device)
    truth_k = np.array([3, 7, 11, 19, 31, 47, 73, 103,
                        137, 173, 199, 201, 229, 247])
    truth_a = np.array([1.0, 0.55, -0.32,
                        7e-3, -5e-3, 3.5e-3, -2.5e-3,
                        8e-5, -6e-5, 4.5e-5,
                        -3.5e-5, 3e-5, -2e-5, 1.5e-5])
    u_star = torch.sum(
        torch.tensor(truth_a)[:, None]
        * torch.sin(math.pi * torch.tensor(truth_k)[:, None] * op.x[None, :]),
        dim=0, keepdim=True)
    f = op.apply(u_star).detach()

    reconstruction = op.shifted_precondition(-f, shift_ratio=1e-10)
    print(f"[*] device={device} dtype={u_star.dtype}")
    print(f"[*] exact resonant modes={op.k[op.null_mask].tolist()}")
    print(f"[*] multiscale truth={list(zip(truth_k.tolist(), truth_a.tolist()))}")
    print(f"[*] initial PINN loss={float(torch.mean(f.square())):.6e}")
    print(f"[*] near-exact inverse check RSE="
          f"{float(torch.linalg.vector_norm(reconstruction-u_star)/torch.linalg.vector_norm(u_star)):.3e}")

    model = RecurrentPhysicalDSTFNO(
        n=op.n, n_grades=8, in_channels=7, width=20, depth=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=80, min_lr=2e-5)

    best = float("inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(401):
        optimizer.zero_grad()
        u_pred, loss, _ = unroll(model, op, f)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        scheduler.step(value)
        if value < best:
            best = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            with torch.no_grad():
                residual = op.residual(u_pred, f)
                pinn = float(torch.mean(residual.square()))
                rse = float(torch.linalg.vector_norm(u_pred-u_star)
                            / torch.linalg.vector_norm(u_star))
            print(f"epoch={epoch:4d} loss={value:.3e} "
                  f"PINN={pinn:.3e} RSE(eval)={rse:.3e}")

    model.load_state_dict(best_state)
    with torch.no_grad():
        final_u, final_loss, history = unroll(model, op, f, return_history=True)
    final_rse = float(torch.linalg.vector_norm(final_u-u_star)
                      / torch.linalg.vector_norm(u_star))
    fname, selected_by_grade = plot_results(op, u_star, history, truth_k)
    print("\n[per-grade diagnostics]")
    seen = set()
    for item, selected in zip(history, selected_by_grade):
        newly_discovered = sorted(set(selected) - seen)
        seen.update(selected)
        print(f"G{item['grade']}: PINN={item['pinn']:.6e} "
              f"rel_res={item['relative_residual']:.3e} eta={item['eta']:.3f} "
              f"null={item['null']:.3e} active_K={selected} "
              f"new_K={newly_discovered}")
    print(f"\nFINAL PINN={history[-1]['pinn']:.6e}")
    print(f"FINAL RSE(eval only)={final_rse:.6e}")
    print(f"wall={time.perf_counter()-start:.1f}s")
    print(f"saved={fname}")


if __name__ == "__main__":
    main()
