"""Single-PDE frozen multigrade model derived from recurrent DST-FNO."""

import math

import torch
import torch.nn as nn


def dst1_ortho(x):
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(-1)
    batch, n, channels = x.shape
    z = torch.zeros(batch, 1, channels, dtype=x.dtype, device=x.device)
    odd = torch.cat([z, x, z, -torch.flip(x, dims=[1])], dim=1)
    spectrum = torch.fft.fft(odd, dim=1)
    coeff = -spectrum.imag[:, 1 : n + 1] / math.sqrt(2.0 * (n + 1))
    return coeff.squeeze(-1) if squeeze else coeff


def idst1_ortho(x_hat):
    return dst1_ortho(x_hat)


class Helmholtz1D:
    def __init__(self, n=256, kappa=200.0 * math.pi, shift_ratio=0.02):
        self.n = n
        self.kappa = float(kappa)
        self.shift_ratio = shift_ratio
        self.x = torch.arange(1, n + 1) / (n + 1)
        self.k = torch.arange(1, n + 1)
        self.lam = (math.pi * self.k) ** 2 - self.kappa**2
        self.null_mask = self.lam.abs() < 10.0

    def apply(self, u):
        return idst1_ortho(dst1_ortho(u) * self.lam.unsqueeze(0))

    def residual(self, u, f):
        return self.apply(u) - f

    def shifted_precondition(self, residual):
        sigma = self.shift_ratio * self.kappa**2
        gain = -self.lam / (self.lam.square() + sigma**2)
        gain = torch.where(self.null_mask, torch.zeros_like(gain), gain)
        return idst1_ortho(dst1_ortho(residual) * gain.unsqueeze(0))

    def null_coefficient(self, u):
        if not self.null_mask.any():
            return torch.zeros(u.shape[0], dtype=u.dtype, device=u.device)
        return dst1_ortho(u)[:, self.null_mask].square().sum(dim=1).sqrt()


class DSTFNOBlock(nn.Module):
    """Exact spectral/local residual block from the recurrent implementation."""

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
        spectral = torch.einsum("bki,kio->bko", v_hat, self.spectral_weight)
        spectral = idst1_ortho(spectral)
        return self.norm(v + self.act(self.local(v) + spectral))


def rms(x):
    return torch.sqrt(torch.mean(x.square(), dim=1, keepdim=True) + 1.0e-30)


def residual_driven_gate(preconditioned, floor_ratio=0.01):
    amplitude = dst1_ortho(preconditioned).abs()
    threshold = torch.maximum(
        floor_ratio * amplitude.amax(dim=1, keepdim=True),
        3.0 * amplitude.median(dim=1, keepdim=True).values,
    )
    gate = (amplitude > threshold).to(preconditioned.dtype)
    gate.scatter_(1, amplitude.argmax(dim=1, keepdim=True), 1.0)
    return gate.detach(), threshold.detach()


def build_features(
    op,
    u_previous,
    residual,
    f,
    preconditioned,
    selected_residual,
    selected_preconditioned,
):
    u_scale = rms(u_previous).detach().clamp_min(1.0e-10)
    r_scale = rms(residual).detach().clamp_min(1.0e-10)
    f_scale = rms(f).detach().clamp_min(1.0e-10)
    p_scale = rms(preconditioned).detach().clamp_min(1.0e-12)
    x_channel = (2.0 * op.x[None, :] - 1.0).expand_as(u_previous)
    kappa_channel = torch.full_like(
        x_channel, op.kappa / (op.n * math.pi)
    )
    features = torch.stack(
        [
            x_channel,
            u_previous / u_scale,
            residual / r_scale,
            f / f_scale,
            preconditioned / p_scale,
            selected_residual / r_scale,
            selected_preconditioned / p_scale,
            kappa_channel,
        ],
        dim=-1,
    )
    return features, preconditioned / p_scale, p_scale


class GradeDSTFNO(nn.Module):
    """One independent copy of the recurrent physical DST-FNO update."""

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
        eta = torch.sigmoid(self.step_logit)
        return eta * (
            learned + self.precond_gain * preconditioned_normalized
        ), eta


def grade_correction(grade, op, u_previous, residual, f, floor_ratio):
    preconditioned = op.shifted_precondition(residual)
    gate, threshold = residual_driven_gate(preconditioned, floor_ratio)
    selected_residual = idst1_ortho(dst1_ortho(residual) * gate)
    selected_preconditioned = idst1_ortho(
        dst1_ortho(preconditioned) * gate
    )
    features, p_normalized, p_scale = build_features(
        op,
        u_previous,
        residual,
        f,
        preconditioned,
        selected_residual,
        selected_preconditioned,
    )
    correction_normalized, eta = grade(features, p_normalized)
    correction_normalized = idst1_ortho(
        dst1_ortho(correction_normalized) * gate
    )
    return correction_normalized * p_scale, {
        "preconditioned": preconditioned,
        "selected_residual": selected_residual,
        "selected_preconditioned": selected_preconditioned,
        "gate": gate,
        "threshold": threshold,
        "eta": eta,
    }
