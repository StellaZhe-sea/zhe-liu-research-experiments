"""
grbd.network.spectral_loss
==========================
Basis-agnostic loss:
    L = Σ_k α_k (ĝ_k − δ̂_k)²  +  λ_oob · mean(ĝ_k² for k ∉ K_n)

make_loss(basis, meta, lambda_oob) auto-selects the right variant.
"""

import torch
import torch.nn as nn
import numpy as np


class BasisLoss(nn.Module):
    def __init__(self, transform, mask, alphas, lambda_oob=0.1):
        super().__init__()
        self.transform  = transform
        self.lambda_oob = lambda_oob
        self.register_buffer("mask_in",  torch.tensor(mask,  dtype=torch.bool))
        self.register_buffer("mask_out", torch.tensor(~mask, dtype=torch.bool))
        self.register_buffer("alphas",   torch.tensor(alphas, dtype=torch.float32))

    def forward(self, g, target):
        g_hat   = self.transform(g)
        tgt_hat = self.transform(target)
        diff2   = (g_hat - tgt_hat)**2
        loss_in  = (self.alphas * diff2[self.mask_in]).sum()
        loss_oob = self.lambda_oob * (g_hat[self.mask_out]**2).mean() \
                   if self.mask_out.any() else torch.zeros(1, device=g.device).squeeze()
        return loss_in + loss_oob, loss_in.item(), loss_oob.detach().item() if hasattr(loss_oob, "detach") else float(loss_oob)


def make_dst_loss(N, mask, alphas, lambda_oob=0.1):
    norm = 2.0 / (N+1)
    def _dst(x):
        try:
            return torch.fft.dst(x, type=1, dim=-1) * norm
        except Exception:
            k = torch.arange(1, N+1, dtype=x.dtype, device=x.device)
            j = torch.arange(1, N+1, dtype=x.dtype, device=x.device)
            M = torch.sin(torch.pi * k[:,None] * j[None,:] / (N+1))
            return (x @ M.T) * norm
    return BasisLoss(_dst, mask, alphas, lambda_oob)


def make_fourier_loss(N, mask, alphas, lambda_oob=0.1):
    def _rfft(x):
        return torch.view_as_real(torch.fft.rfft(x) / N).reshape(-1)
    return BasisLoss(_rfft, mask, alphas, lambda_oob)


def make_chebyshev_loss(N, mask, alphas, lambda_oob=0.1):
    def _dct(x):
        k = torch.arange(N, dtype=x.dtype, device=x.device).view(-1, 1)
        n = torch.arange(N, dtype=x.dtype, device=x.device).view(1, -1)
        mat = 2.0 * torch.cos(torch.pi * k * (2.0 * n + 1.0) / (2.0 * N)) / N
        return mat @ x
    return BasisLoss(_dct, mask, alphas, lambda_oob)


class _WaveletLoss(nn.Module):
    """Wavelet loss: numpy transform wrapped for autograd-free coefficient loss."""
    def __init__(self, basis, meta, alphas, lambda_oob=0.1):
        super().__init__()
        self.basis      = basis
        self.meta       = meta
        self.lambda_oob = lambda_oob
        mf = meta.get("mask_flat", meta.get("mask"))
        self.register_buffer("alphas",   torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("mask_in",  torch.tensor(mf,    dtype=torch.bool))
        self.register_buffer("mask_out", torch.tensor(~mf,   dtype=torch.bool))

    def _coeff(self, x):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        _, m = self.basis.decompose(x_np)
        return torch.tensor(m["all_coeffs"], dtype=torch.float32, device=x.device)

    def forward(self, g, target):
        g_hat   = self._coeff(g)
        tgt_hat = self._coeff(target)
        diff2   = (g_hat - tgt_hat)**2
        loss_in  = (self.alphas * diff2[self.mask_in]).sum()
        loss_oob = self.lambda_oob * (g_hat[self.mask_out]**2).mean() \
                   if self.mask_out.any() else torch.zeros(1).squeeze()
        return loss_in + loss_oob, loss_in.item(), loss_oob.detach().item() if hasattr(loss_oob, "detach") else float(loss_oob)


def make_loss(basis, meta, lambda_oob=0.1):
    """Auto-select loss given basis type."""
    alphas   = meta["alphas"]
    N        = basis.op.N
    cls      = type(basis).__name__
    if cls == "DSTBasis":
        return make_dst_loss(N, meta["mask"], alphas, lambda_oob)
    elif cls == "FourierBasis":
        return make_fourier_loss(N, meta["mask_use"], alphas, lambda_oob)
    elif cls == "ChebyshevBasis":
        return make_chebyshev_loss(N, meta["mask"], alphas, lambda_oob)
    elif cls in ("WaveletBasis", "WaveletPacketBasis"):
        return _WaveletLoss(basis, meta, alphas, lambda_oob)
    else:
        raise ValueError(f"Unknown basis: {cls}")
