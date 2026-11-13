"""
grbd.operators.base
===================
Abstract PDE operator interface.

Every concrete operator must implement:
  - residual(u, f)   : L[u] - f  in physical space
  - inv_L(rhs)       : L^{-1}[rhs]  (exact or approximate)
  - make_f(u_star)   : compute RHS f = L[u_star]
  - sobolev_decay(alpha, tau) : spectral weights λ_k^{-alpha} for GRF / Sobolev weighting
  - grid              : 1-D spatial coordinates (N,)
  - N                 : number of interior DOFs
"""

from abc import ABC, abstractmethod
import numpy as np


class PDEOperator(ABC):

    # ── required properties ──────────────────────────────────────────────
    @property
    @abstractmethod
    def N(self) -> int:
        """Number of interior degrees of freedom."""

    @property
    @abstractmethod
    def grid(self) -> np.ndarray:
        """1-D spatial grid, shape (N,)."""

    # ── required methods ─────────────────────────────────────────────────
    @abstractmethod
    def make_f(self, u_star: np.ndarray) -> np.ndarray:
        """Compute RHS f = L[u_star]."""

    @abstractmethod
    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute operator residual L[u] - f."""

    @abstractmethod
    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        """Apply L^{-1} to rhs."""

    @abstractmethod
    def eigenvalues(self) -> np.ndarray:
        """
        Return operator eigenvalues λ_k for k=1..N.
        Used by BasisSolver.sobolev_weights() and GRF sampling.
        """

    # ── convenience ──────────────────────────────────────────────────────
    def sobolev_decay(self, alpha: float = 2.0, tau: float = 3.0) -> np.ndarray:
        """
        Sobolev spectral weights  w_k = (τ² + λ_k)^{-alpha/2}.

        This is the FNO / GRF logic generalised to any operator:
        the covariance operator C = σ²(τ²I + L)^{-alpha} defines
        a Gaussian measure on H^alpha.  The same weights serve as
        per-coefficient importance in the spectral loss.
        """
        lam = np.abs(self.eigenvalues())          # ensure positive
        return (tau**2 + lam) ** (-alpha / 2.0)

    def check_inverse(self, tol: float = 1e-4) -> float:
        """Sanity check: relative error of inv_L ∘ L on a smooth test vector."""
        x = self.grid
        v = np.sin(3 * np.pi * x).astype(np.float32)
        f = self.make_f(v)
        v_rec = self.inv_L(self.residual(np.zeros_like(v), f))
        # inv_L(L[v] - 0) should equal -v  (since residual = L[0]-f = -f = -L[v])
        # actually inv_L(residual(0,f)) = inv_L(-f) = -inv_L(L[v]) = -v
        err = float(np.linalg.norm(-v_rec - v) / (np.linalg.norm(v) + 1e-30))
        return err
