"""
grbd.operators.helmholtz
========================
1-D Helmholtz  L[u] = -u_xx - κ²u  on (0,1) with Dirichlet BC.

Solved exactly via DST Type-I:
  eigenvalues  λ_k = (πk)² - κ²,   k = 1..N
  grid         x_j = j/(N+1),       j = 1..N
"""

import numpy as np
from scipy.fft import dst, idst
from .base import PDEOperator


class HelmholtzOperator(PDEOperator):

    def __init__(self, N: int, kappa: float):
        self._N     = N
        self.kappa  = kappa
        self._h     = 1.0 / (N + 1)
        self._x     = np.arange(1, N + 1) * self._h
        k           = np.arange(1, N + 1, dtype=np.float64)
        self._lam   = (np.pi * k) ** 2 - kappa ** 2
        # safe inverse: avoid division by zero near resonance
        self._safe  = np.where(
            np.abs(self._lam) > 1e-8,
            self._lam,
            np.sign(self._lam + 1e-12) * 1e-8,
        )
        self._norm  = 2.0 / (N + 1)   # DST-I orthonormality factor

    # ── PDEOperator interface ────────────────────────────────────────────
    @property
    def N(self) -> int:
        return self._N

    @property
    def grid(self) -> np.ndarray:
        return self._x

    def eigenvalues(self) -> np.ndarray:
        return self._lam.copy()

    def make_f(self, u: np.ndarray) -> np.ndarray:
        return idst(self._lam * dst(u, type=1), type=1).astype(np.float32)

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        u_hat = dst(u, type=1).astype(np.float32)
        f_hat = dst(f, type=1).astype(np.float32)
        return idst(self._lam * u_hat - f_hat, type=1).astype(np.float32)

    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        return idst(
            dst(rhs, type=1).astype(np.float32) / self._safe,
            type=1,
        ).astype(np.float32)
