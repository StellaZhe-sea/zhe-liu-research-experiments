"""
grbd.basis.fourier
==================
Real Fourier basis for periodic BC problems (Burgers, NLS, heat, …).

Atoms    : cos(2πkx), sin(2πkx),  k = 0..K-1  (rfft representation)
Selection: GD threshold on |â_k| of residual rfft
Weights  : Sobolev weights  (τ² + (2πk)²)^{-α/2}  or plain energy
"""

import numpy as np
from .base import BasisSolver


def _gd_threshold(amps: np.ndarray, floor_ratio: float = 0.10) -> float:
    s   = np.sort(amps)[::-1]
    mad = np.median(np.abs(s - np.median(s)))
    sig = mad / 0.6745 + 1e-12
    n   = len(amps); beta = 1.0 / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return float(max(omega * sig, floor_ratio * s[0]))


class FourierBasis(BasisSolver):
    """
    Parameters
    ----------
    op             : PDEOperator  (periodic, must expose .N, .grid)
    n_modes        : int | None   max Fourier modes to consider (None → N//2)
    gd_floor       : float
    sobolev_alpha  : float        >0 activates Sobolev weighting
    sobolev_tau    : float
    """

    def __init__(self, op, n_modes=None, gd_floor=0.10,
                 sobolev_alpha=0.0, sobolev_tau=1.0):
        self.op            = op
        self._N            = op.N
        self._n_modes      = n_modes or (op.N // 2)
        self.gd_floor      = gd_floor
        self.sobolev_alpha = sobolev_alpha
        self.sobolev_tau   = sobolev_tau
        self._x            = op.grid                          # (N,)
        # wavenumbers for rfft output (length N//2+1)
        self._k_rfft = np.fft.rfftfreq(op.N, d=1.0 / op.N)  # 0,1,2,…,N/2

    # ── helpers ──────────────────────────────────────────────────────────

    def _rfft_amps(self, u: np.ndarray) -> np.ndarray:
        """Amplitude of rfft coefficients, scaled to physical units."""
        c = np.fft.rfft(u) / self._N
        return np.abs(c).astype(np.float32)

    def _build_basis_cols(self, k_sel: np.ndarray) -> np.ndarray:
        """
        Real Fourier atoms at selected wavenumbers.
        For k=0: constant 1.  For k>0: [cos, sin] pair → 2 columns.
        Returns (N, M) float32 where M = sum of active columns.
        """
        cols = []
        x    = self._x[:, None]           # (N,1)
        for k in k_sel:
            if k == 0:
                cols.append(np.ones((self._N, 1), dtype=np.float32))
            else:
                cols.append(np.cos(2 * np.pi * k * x).astype(np.float32))
                cols.append(np.sin(2 * np.pi * k * x).astype(np.float32))
        return np.concatenate(cols, axis=1) if cols else np.zeros((self._N, 0), np.float32)

    # ── BasisSolver interface ────────────────────────────────────────────

    def decompose(self, residual: np.ndarray) -> tuple:
        c_full   = np.fft.rfft(residual) / self._N        # complex (N//2+1,)
        amps     = np.abs(c_full).astype(np.float32)
        amps_use = amps[:self._n_modes + 1]               # restrict to n_modes

        tau      = _gd_threshold(amps_use, self.gd_floor)
        mask_use = amps_use > tau
        if not mask_use.any():
            mask_use[np.argmax(amps_use)] = True

        k_sel    = self._k_rfft[:self._n_modes + 1][mask_use]
        # coefficients = complex rfft values at selected k
        coeffs_c = c_full[:self._n_modes + 1][mask_use]

        # n_atoms counts real DOFs: k=0 → 1, k>0 → 2
        n_atoms  = sum(1 if k == 0 else 2 for k in k_sel)

        meta = dict(
            mask_use=mask_use, k_sel=k_sel,
            amps_rfft=amps, tau=tau,
            coeffs_c=coeffs_c,
            n_atoms=n_atoms,
        )
        # flatten to real coefficients for consistency
        coeffs_r = np.concatenate([
            np.array([c.real]) if k == 0 else np.array([c.real, c.imag])
            for k, c in zip(k_sel, coeffs_c)
        ]).astype(np.float32)
        return coeffs_r, meta

    def reconstruct(self, coeffs_r: np.ndarray, meta: dict) -> np.ndarray:
        c_full = np.zeros(self._N // 2 + 1, dtype=np.complex64)
        idx    = 0
        for k, k_idx in zip(meta["k_sel"],
                             np.where(meta["mask_use"])[0]):
            if k == 0:
                c_full[k_idx] = coeffs_r[idx]; idx += 1
            else:
                c_full[k_idx] = coeffs_r[idx] + 1j * coeffs_r[idx + 1]; idx += 2
        return np.fft.irfft(c_full * self._N, n=self._N).astype(np.float32)

    def atom_features(self, residual: np.ndarray, target: np.ndarray,
                      meta: dict) -> tuple:
        k_sel    = meta["k_sel"]
        c_target = np.fft.rfft(target) / self._N
        c_sel    = c_target[:len(meta["mask_use"])][meta["mask_use"]]

        # amplitude and sign per real DOF
        amps_list, signs_list = [], []
        for c in c_sel:
            a = abs(c)
            if c.real == 0 and c.imag == 0:
                amps_list += [0.0, 0.0]; signs_list += [1.0, 1.0]
            else:
                amps_list += [abs(c.real), abs(c.imag)]
                signs_list += [np.sign(c.real) or 1.0, np.sign(c.imag) or 1.0]

        amps_r  = np.array(amps_list,  dtype=np.float32)
        signs_r = np.array(signs_list, dtype=np.float32)

        # target coefficients (real DOFs) for Sobolev weight computation
        tc = np.concatenate([
            np.array([c.real]) if k == 0 else np.array([c.real, c.imag])
            for k, c in zip(k_sel, c_sel)
        ]).astype(np.float32)

        alphas = self.sobolev_weights({**meta, "target_coeffs": tc})
        H      = self._spectral_entropy(alphas)
        B      = self._build_basis_cols(k_sel)          # (N, n_atoms)
        s      = self._log_norm_encode(amps_r, signs_r, B)

        meta = {
            **meta,
            "target_coeffs": tc,
            "amps_delta":    amps_r,
            "alphas":        alphas,
            "entropy":       H,
            "target":        target,
            "s_input":       s,
        }
        return s, meta

    def sobolev_weights(self, meta: dict) -> np.ndarray:
        tc = meta.get("target_coeffs", None)
        if tc is None:
            return np.ones(meta["n_atoms"]) / meta["n_atoms"]

        if self.sobolev_alpha > 0:
            # expand k_sel to real DOF wavenumbers
            k_dofs = np.concatenate([
                [0] if k == 0 else [k, k] for k in meta["k_sel"]
            ]).astype(np.float64)
            lam = (2 * np.pi * k_dofs) ** 2
            w   = (self.sobolev_tau**2 + lam) ** (-self.sobolev_alpha / 2.0)
            w   = w * tc**2
        else:
            w = tc**2

        w = np.abs(w)
        return w / (w.sum() + 1e-12)
