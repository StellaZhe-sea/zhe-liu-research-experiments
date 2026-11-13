"""
operators.py
HelmholtzOp: -u'' + kappa^2 u = f, Dirichlet BC, uniform grid.
inv_L is exact via DST — shared by all three basis frameworks.
"""

import math
import numpy as np
import torch


def dst1(x: torch.Tensor) -> torch.Tensor:
    """DST Type-I via FFT, along last dim."""
    N = x.shape[-1]
    z = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
    ext = torch.cat([z, x, z, -x.flip(-1)], dim=-1)
    fc = torch.fft.rfft(ext, dim=-1)
    return -fc[..., 1:N+1].imag


def idst1(c: torch.Tensor) -> torch.Tensor:
    """DST Type-I inverse (symmetric, divide by 2(N+1))."""
    N = c.shape[-1]
    return dst1(c) / (2 * (N + 1))


class HelmholtzOp:
    """
    -u'' + kappa^2 u = f on [0,1], u(0)=u(1)=0.
    Grid: x_i = i/(N+1), i=1..N  (N uniform interior points).
    Eigenvalues: lam_k = (pi*k)^2 - kappa^2.
    """

    def __init__(self, N: int = 128, kappa: float = 10.0):
        self.N = N
        self.kappa = kappa
        k = torch.arange(1, N + 1, dtype=torch.float64)
        self.x = k / (N + 1)                          # (N,)
        self.lam = (math.pi * k) ** 2 - kappa ** 2    # (N,)
        # guard against near-zero eigenvalues
        self._safe = torch.where(
            self.lam.abs() > 1e-8, self.lam,
            torch.full_like(self.lam, 1e-8)
        )

    def make_f(self, u: torch.Tensor) -> torch.Tensor:
        """Forward operator: f = L[u]."""
        return idst1(self.lam * dst1(u.double())).to(u.dtype)

    def residual(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """e_n = L[u_n] - f  (in equation space)."""
        return idst1(self.lam * dst1(u.double()) - dst1(f.double())).to(u.dtype)

    def inv_L(self, r: torch.Tensor) -> torch.Tensor:
        """Exact spectral inverse: inv_L(r) = u_n - u*  when r = e_n."""
        return idst1(dst1(r.double()) / self._safe).to(r.dtype)
