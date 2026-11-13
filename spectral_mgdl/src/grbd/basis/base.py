"""
grbd.basis.base
===============
Abstract BasisSolver interface.

Every concrete basis must implement:

  decompose(residual)
      residual (N,) → coeffs (M,), meta dict
      Select the M "active" atoms from the full decomposition.

  reconstruct(coeffs, meta)
      coeffs (M,) + meta → physical field (N,)
      Inverse transform restricted to active atoms.

  atom_features(residual, target, meta)
      Build the input feature matrix s (N, M) fed to GradeNet,
      and enrich meta with:
        - 'alphas'  (M,)  per-atom loss weights  (Sobolev or energy)
        - 'n_atoms' int
        - 'entropy' float  spectral entropy of alphas

  sobolev_weights(meta)
      Return per-atom Sobolev importance weights w_k (M,).
      Default: energy of selected coefficients (override for richer priors).

The GradeSolver and GradeNet are completely basis-agnostic;
they only interact through this interface.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasisSolver(ABC):

    # ── core interface ───────────────────────────────────────────────────

    @abstractmethod
    def decompose(self, residual: np.ndarray) -> tuple:
        """
        Parameters
        ----------
        residual : (N,) float32

        Returns
        -------
        coeffs : (M,) float32   selected atom coefficients
        meta   : dict           bookkeeping for reconstruct / atom_features
        """

    @abstractmethod
    def reconstruct(self, coeffs: np.ndarray, meta: dict) -> np.ndarray:
        """
        Parameters
        ----------
        coeffs : (M,) float32
        meta   : dict  from decompose

        Returns
        -------
        field : (N,) float32
        """

    @abstractmethod
    def atom_features(
        self,
        residual: np.ndarray,
        target:   np.ndarray,
        meta:     dict,
    ) -> tuple:
        """
        Build GradeNet input features and enrich meta.

        Parameters
        ----------
        residual : (N,) float32   e_n  = L[u_n] - f
        target   : (N,) float32   δ_n  = -L^{-1}[e_n]
        meta     : dict           from decompose(residual)

        Returns
        -------
        s    : (N, M) float32   feature matrix for GradeNet
        meta : dict             enriched with 'alphas', 'n_atoms', 'entropy'
        """

    # ── optional override ────────────────────────────────────────────────

    def sobolev_weights(self, meta: dict) -> np.ndarray:
        """
        Per-atom Sobolev importance weights.
        Default: energy of selected target coefficients (same as v8 logic).
        Override in spectral bases to use operator eigenvalues.
        """
        a = np.abs(meta.get("target_coeffs", np.ones(meta["n_atoms"])))
        energy = a ** 2
        return energy / (energy.sum() + 1e-12)

    # ── shared helpers ───────────────────────────────────────────────────

    @staticmethod
    def _spectral_entropy(alphas: np.ndarray) -> float:
        p = alphas / (alphas.sum() + 1e-12)
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _log_norm_encode(
        amps:  np.ndarray,   # (M,) positive
        signs: np.ndarray,   # (M,) ±1
        basis: np.ndarray,   # (N, M)
    ) -> np.ndarray:
        """
        Log-amplitude normalised signed feature encoding (from v8):
            s[:, i] = log(1 + a_i) / log(1 + a_max)  ×  sign_i  ×  basis[:, i]
        Returns s (N, M) float32.
        """
        log_a   = np.log(1.0 + amps)
        log_max = np.log(1.0 + amps.max() + 1e-12)
        log_norm = log_a / log_max                    # (M,)
        return (log_norm[None, :] * signs[None, :] * basis).astype(np.float32)
