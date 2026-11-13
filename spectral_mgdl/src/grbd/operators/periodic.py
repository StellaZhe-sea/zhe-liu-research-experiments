"""
grbd.operators.periodic
=======================
1-D periodic diffusion  L[u] = -ν u_xx  on [0,1) (unit torus).
Used for Burgers / NLS / heat equation problems where BC is periodic.

Solved via FFT:
  eigenvalues  λ_k = ν (2πk)²,  k = 0..N-1  (wavenumbers)
  grid         x_j = j/N,       j = 0..N-1
"""

import numpy as np
from .base import PDEOperator


class PeriodicDiffusionOperator(PDEOperator):

    def __init__(self, N: int, nu: float = 1.0):
        self._N   = N
        self.nu   = nu
        self._x   = np.linspace(0, 1, N, endpoint=False).astype(np.float32)
        k         = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float64)  # wavenumbers
        self._k   = k
        self._lam = nu * (2 * np.pi * k) ** 2                          # eigenvalues for rfft

    @property
    def N(self) -> int:
        return self._N

    @property
    def grid(self) -> np.ndarray:
        return self._x

    def eigenvalues(self) -> np.ndarray:
        # Return full-spectrum eigenvalues (length N) via irfft convention
        k_full = np.fft.fftfreq(self._N, d=1.0 / self._N)
        return self.nu * (2 * np.pi * k_full) ** 2

    def make_f(self, u: np.ndarray) -> np.ndarray:
        u_hat = np.fft.rfft(u)
        return np.fft.irfft(self._lam * u_hat, n=self._N).astype(np.float32)

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        return self.make_f(u) - f.astype(np.float32)

    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        rhs_hat = np.fft.rfft(rhs)
        safe    = np.where(np.abs(self._lam) > 1e-10, self._lam, 1e-10)
        return np.fft.irfft(rhs_hat / safe, n=self._N).astype(np.float32)
