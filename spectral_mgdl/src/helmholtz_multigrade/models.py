"""The single model definition: modularized from recurrent physical DST-FNO."""

import math

import torch
import torch.nn as nn


def dst1_ortho(x):
    """Differentiable orthonormal DST-I along spatial dimension 1."""
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
    """The orthonormal DST-I matrix is symmetric and self-inverse."""
    return dst1_ortho(x_hat)


class BatchedHelmholtz1D:
    """A batch of -u_xx-kappa^2 u=f problems with homogeneous Dirichlet BCs."""

    def __init__(self, n=256, shift_ratio=1.0e-3, device="cpu"):
        self.n = n
        self.shift_ratio = shift_ratio
        self.device = torch.device(device)
        self.x = torch.arange(1, n + 1, device=self.device) / (n + 1)
        self.k = torch.arange(1, n + 1, device=self.device)

    def eigenvalues(self, kappa):
        return (math.pi * self.k[None, :]) ** 2 - kappa[:, None] ** 2

    def apply(self, u, kappa):
        return idst1_ortho(dst1_ortho(u) * self.eigenvalues(kappa))

    def residual(self, u, f, kappa):
        return self.apply(u, kappa) - f

    def shifted_precondition(self, residual, kappa):
        """Stable approximation of -L_kappa^{-1} residual."""
        lam = self.eigenvalues(kappa)
        sigma = self.shift_ratio * kappa[:, None].square()
        gain = -lam / (lam.square() + sigma.square())
        gain = torch.where(lam.abs() < 10.0, torch.zeros_like(gain), gain)
        return idst1_ortho(dst1_ortho(residual) * gain)


class DSTFNOBlock(nn.Module):
    """The spectral/local residual block used by the recurrent prototype."""

    def __init__(self, n, width):
        super().__init__()
        self.spectral_weight = nn.Parameter(
            0.02 * torch.randn(n, width, width) / math.sqrt(width)
        )
        self.local = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.GELU()

    def forward(self, v):
        v_hat = dst1_ortho(v)
        spectral = torch.einsum(
            "bki,kio->bko", v_hat, self.spectral_weight
        )
        spectral = idst1_ortho(spectral)
        update = self.local(v) + spectral
        return self.norm(v + self.act(update))


def sample_rms(x, eps=1.0e-30):
    return torch.sqrt(torch.mean(x.square(), dim=1, keepdim=True) + eps)


def residual_driven_gate(preconditioned, floor_ratio=0.01):
    """Hard adaptive selection derived only from the current physical residual."""
    amplitude = dst1_ortho(preconditioned).abs()
    tau_floor = floor_ratio * amplitude.amax(dim=1, keepdim=True)
    tau_noise = 3.0 * amplitude.median(dim=1, keepdim=True).values
    threshold = torch.maximum(tau_floor, tau_noise)
    gate = (amplitude > threshold).to(preconditioned.dtype)
    strongest = amplitude.argmax(dim=1, keepdim=True)
    gate.scatter_(1, strongest, 1.0)
    return gate.detach(), threshold.detach()


def build_physical_features(
    op,
    u,
    residual,
    f,
    preconditioned,
    selected_residual,
    selected_preconditioned,
    kappa,
):
    """Physical state plus dynamically selected spectral content.

    The selected fields are obtained by inverse-DST reconstruction, so the
    network still receives coordinate-aligned physical-space channels.
    """
    u_scale = sample_rms(u).detach().clamp_min(1.0e-10)
    residual_scale = sample_rms(residual).detach().clamp_min(1.0e-10)
    forcing_scale = sample_rms(f).detach().clamp_min(1.0e-10)
    preconditioned_scale = (
        sample_rms(preconditioned).detach().clamp_min(1.0e-12)
    )
    x_channel = (2.0 * op.x[None, :] - 1.0).expand_as(u)
    kappa_channel = (kappa / (op.n * math.pi))[:, None].expand_as(u)
    features = torch.stack(
        [
            x_channel,
            u / u_scale,
            residual / residual_scale,
            f / forcing_scale,
            preconditioned / preconditioned_scale,
            selected_residual / residual_scale,
            selected_preconditioned / preconditioned_scale,
            kappa_channel,
        ],
        dim=-1,
    )
    return features, preconditioned / preconditioned_scale, preconditioned_scale


class GradeDSTFNO(nn.Module):
    """One independent copy of the recurrent residual-DST-FNO update."""

    def __init__(self, n=256, width=20, depth=3):
        super().__init__()
        self.lift = nn.Linear(8, width)
        self.blocks = nn.ModuleList(
            [DSTFNOBlock(n, width) for _ in range(depth)]
        )
        self.project = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)
        self.precond_gain = nn.Parameter(torch.tensor(1.0))
        self.step_logit = nn.Parameter(torch.tensor(1.4))

    def forward(self, features, preconditioned_normalized):
        v = self.lift(features)
        for block in self.blocks:
            v = block(v)
        learned = self.project(v).squeeze(-1)
        correction_normalized = (
            learned + self.precond_gain * preconditioned_normalized
        )
        eta = torch.sigmoid(self.step_logit)
        return eta * correction_normalized, eta


def grade_correction(
    grade, op, u_previous, residual_previous, f, kappa, gate_floor_ratio
):
    """One physical residual update; no hidden feature crosses grade boundaries."""
    preconditioned = op.shifted_precondition(residual_previous, kappa)
    gate, threshold = residual_driven_gate(
        preconditioned, floor_ratio=gate_floor_ratio
    )
    selected_residual = idst1_ortho(
        dst1_ortho(residual_previous) * gate
    )
    selected_preconditioned = idst1_ortho(
        dst1_ortho(preconditioned) * gate
    )
    features, p_normalized, p_scale = build_physical_features(
        op,
        u_previous,
        residual_previous,
        f,
        preconditioned,
        selected_residual,
        selected_preconditioned,
        kappa,
    )
    correction_normalized, eta = grade(features, p_normalized)
    correction_normalized = idst1_ortho(
        dst1_ortho(correction_normalized) * gate
    )
    correction = correction_normalized * p_scale
    return correction, {
        "preconditioned": preconditioned,
        "selected_residual": selected_residual,
        "selected_preconditioned": selected_preconditioned,
        "gate": gate,
        "gate_threshold": threshold,
        "eta": eta,
    }
