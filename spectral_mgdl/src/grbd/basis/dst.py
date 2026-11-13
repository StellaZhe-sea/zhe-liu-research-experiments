"""
grbd.basis.dst
==============
DST Type-I basis for Dirichlet BC problems (Helmholtz, Poisson, …).

Atoms    : sin(πkx/(N+1)),  k = 1..N
Selection: Gavish-Donoho thresholding on residual spectrum
Weights  : Sobolev weights via operator eigenvalues  λ_k = (πk)² − κ²
           (or plain energy if operator not supplied)
"""

import numpy as np
from scipy.fft import dst, idst
from .base import BasisSolver


# ── Gavish-Donoho threshold ──────────────────────────────────────────────────
def _gd_threshold(amps: np.ndarray, floor_ratio: float = 0.10) -> float:
    s   = np.sort(amps)[::-1]
    mad = np.median(np.abs(s - np.median(s)))
    sig = mad / 0.6745 + 1e-12
    n   = len(amps); beta = 1.0 / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return float(max(omega * sig, floor_ratio * s[0]))


class DSTBasis(BasisSolver):
    """
    Parameters
    ----------
    op         : PDEOperator  (must expose .N, .grid, .eigenvalues())
    gd_floor   : float        GD threshold floor as fraction of max amplitude
    sobolev_alpha : float     exponent for Sobolev weights (0 → pure energy)
    sobolev_tau   : float     shift in (τ² + λ)^{-α/2}
    """

    def __init__(self, op, gd_floor: float = 0.10,
                 sobolev_alpha: float = 0.0,
                 sobolev_tau:   float = 0.0):
        self.op            = op
        self.gd_floor      = gd_floor
        self.sobolev_alpha = sobolev_alpha
        self.sobolev_tau   = sobolev_tau
        self._N            = op.N
        self._k_all        = np.arange(1, op.N + 1)
        # precompute DST sine basis matrix  (N, N)
        j = np.arange(1, op.N + 1)[:, None]   # (N,1)
        k = self._k_all[None, :]               # (1,N)
        self._B = np.sin(np.pi * k * j / (op.N + 1)).astype(np.float32)  # (N,N)

    # ── BasisSolver interface ────────────────────────────────────────────

    def decompose(self, residual: np.ndarray) -> tuple:
        e_hat   = dst(residual, type=1).astype(np.float32)
        amps    = np.abs(e_hat)
        tau     = _gd_threshold(amps, self.gd_floor)
        mask    = amps > tau
        if not mask.any():
            mask[np.argmax(amps)] = True
        k_sel   = self._k_all[mask]
        coeffs  = e_hat[mask]
        meta    = dict(
            mask=mask, k_sel=k_sel, k_all=self._k_all,
            e_hat=e_hat, amps_res=amps, tau=tau,
            n_atoms=int(mask.sum()),
        )
        return coeffs, meta

    def reconstruct(self, coeffs: np.ndarray, meta: dict) -> np.ndarray:
        full              = np.zeros(self._N, dtype=np.float32)
        full[meta["mask"]] = coeffs
        return idst(full, type=1).astype(np.float32)

    def atom_features(self, residual: np.ndarray, target: np.ndarray,
                      meta: dict) -> tuple:
        mask  = meta["mask"]
        t_hat = dst(target, type=1).astype(np.float32)
        a_sel = np.abs(t_hat[mask])
        signs = np.sign(t_hat[mask])

        # Sobolev or energy weights
        alphas = self.sobolev_weights({**meta, "target_coeffs": t_hat[mask]})

        H = self._spectral_entropy(alphas)
        s = self._log_norm_encode(a_sel, signs, self._B[:, mask])

        meta = {
            **meta,
            "target_hat":    t_hat,
            "amps_delta":    np.abs(t_hat),
            "target_coeffs": t_hat[mask],
            "alphas":        alphas,
            "entropy":       H,
            "log_norm":      np.log(1 + a_sel) / (np.log(1 + a_sel.max()) + 1e-12),
            "signs":         signs,
            "target":        target,
            "s_input":       s,
        }
        return s, meta

    def sobolev_weights(self, meta: dict) -> np.ndarray:
        """
        If sobolev_alpha > 0: use operator eigenvalues for Sobolev weighting.
        Otherwise: fall back to energy of target coefficients (v8 behaviour).
        """
        tc = meta.get("target_coeffs", None)
        if tc is None:
            return np.ones(meta["n_atoms"]) / meta["n_atoms"]

        if self.sobolev_alpha > 0:
            lam   = np.abs(self.op.eigenvalues())[meta["mask"]]
            w     = (self.sobolev_tau**2 + lam) ** (-self.sobolev_alpha / 2.0)
            # blend with energy: weight = sobolev × energy
            energy = tc**2
            w      = w * energy
        else:
            w = tc**2

        w = np.abs(w)
        return w / (w.sum() + 1e-12)
