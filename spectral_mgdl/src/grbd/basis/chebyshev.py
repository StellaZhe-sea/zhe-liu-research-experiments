"""
grbd.basis.chebyshev
====================
Chebyshev Type-I basis for non-periodic problems on (−1, 1) or (0, 1).

Atoms    : T_k(x) = cos(k·arccos(x)),  k = 0..N-1
Transform: DCT-II (scipy) on Chebyshev nodes
Selection: GD threshold on |ĉ_k|
Weights  : Sobolev weights  (τ² + k²)^{-α/2}  (Chebyshev order proxy)

Note: the grid is the Chebyshev-Gauss-Lobatto (CGL) grid mapped to (0,1):
      x_j = ½(1 − cos(πj/(N+1))),  j = 1..N  (interior nodes only)
The DCT-II / III pair gives the forward / inverse Chebyshev transform.
"""

import numpy as np
from scipy.fft import dct, idct
from .base import BasisSolver


def _gd_threshold(amps: np.ndarray, floor_ratio: float = 0.10) -> float:
    s   = np.sort(amps)[::-1]
    mad = np.median(np.abs(s - np.median(s)))
    sig = mad / 0.6745 + 1e-12
    n   = len(amps); beta = 1.0 / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return float(max(omega * sig, floor_ratio * s[0]))


class ChebyshevBasis(BasisSolver):
    """
    Parameters
    ----------
    op            : PDEOperator  must expose .N, .grid
    gd_floor      : float
    sobolev_alpha : float   >0 → weight by Chebyshev order  (k+1)^{-α}
    sobolev_tau   : float
    domain        : tuple   physical domain (a, b) default (0, 1)
    """

    def __init__(self, op, gd_floor=0.10,
                 sobolev_alpha=0.0, sobolev_tau=1.0,
                 domain=(0.0, 1.0)):
        self.op            = op
        self._N            = op.N
        self.gd_floor      = gd_floor
        self.sobolev_alpha = sobolev_alpha
        self.sobolev_tau   = sobolev_tau
        self._a, self._b   = domain
        self._k_all        = np.arange(self._N)   # orders 0..N-1
        # Chebyshev nodes on (a,b): interior only
        j    = np.arange(1, self._N + 1)
        xi   = np.cos(np.pi * j / (self._N + 1))    # on (-1,1)
        self._x = 0.5 * (self._b - self._a) * (1 - xi) + self._a  # map to (a,b)
        # Chebyshev polynomial matrix  T_k(xi_j)  shape (N,N)
        self._B = np.cos(
            np.outer(np.arccos(xi), self._k_all)
        ).astype(np.float32)   # (N, N)

    # ── transform helpers ────────────────────────────────────────────────

    def _fwd(self, u: np.ndarray) -> np.ndarray:
        """Physical → Chebyshev coefficients (DCT-II, unnormalised)."""
        return dct(u, type=2, norm=None).astype(np.float32) / self._N

    def _inv(self, c: np.ndarray) -> np.ndarray:
        """Chebyshev coefficients → physical (DCT-III)."""
        return idct(c * self._N, type=2, norm=None).astype(np.float32)

    # ── BasisSolver interface ────────────────────────────────────────────

    def decompose(self, residual: np.ndarray) -> tuple:
        c_hat  = self._fwd(residual)
        amps   = np.abs(c_hat)
        tau    = _gd_threshold(amps, self.gd_floor)
        mask   = amps > tau
        if not mask.any():
            mask[np.argmax(amps)] = True
        k_sel  = self._k_all[mask]
        coeffs = c_hat[mask]
        meta   = dict(
            mask=mask, k_sel=k_sel, k_all=self._k_all,
            c_hat=c_hat, amps_res=amps, tau=tau,
            n_atoms=int(mask.sum()),
        )
        return coeffs, meta

    def reconstruct(self, coeffs: np.ndarray, meta: dict) -> np.ndarray:
        full             = np.zeros(self._N, dtype=np.float32)
        full[meta["mask"]] = coeffs
        return self._inv(full)

    def atom_features(self, residual: np.ndarray, target: np.ndarray,
                      meta: dict) -> tuple:
        mask   = meta["mask"]
        t_hat  = self._fwd(target)
        a_sel  = np.abs(t_hat[mask])
        signs  = np.sign(t_hat[mask])
        signs[signs == 0] = 1.0

        alphas = self.sobolev_weights({**meta, "target_coeffs": t_hat[mask]})
        H      = self._spectral_entropy(alphas)
        s      = self._log_norm_encode(a_sel, signs, self._B[:, mask])

        meta = {
            **meta,
            "target_hat":    t_hat,
            "amps_delta":    np.abs(t_hat),
            "target_coeffs": t_hat[mask],
            "alphas":        alphas,
            "entropy":       H,
            "log_norm":      np.log(1+a_sel) / (np.log(1+a_sel.max())+1e-12),
            "signs":         signs,
            "target":        target,
            "s_input":       s,
        }
        return s, meta

    def sobolev_weights(self, meta: dict) -> np.ndarray:
        tc = meta.get("target_coeffs", None)
        if tc is None:
            return np.ones(meta["n_atoms"]) / meta["n_atoms"]

        if self.sobolev_alpha > 0:
            k_sel = meta["k_sel"].astype(np.float64) + 1.0   # avoid k=0
            w     = (self.sobolev_tau**2 + k_sel**2) ** (-self.sobolev_alpha / 2.0)
            w     = w * tc**2
        else:
            w = tc**2

        w = np.abs(w)
        return w / (w.sum() + 1e-12)
