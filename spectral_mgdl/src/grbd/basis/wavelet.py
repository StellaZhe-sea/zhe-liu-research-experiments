"""
grbd.basis.wavelet
==================
Discrete Wavelet Transform (DWT) basis — Suite 2.

Multi-resolution decomposition: residual split into
  approximation (scaling) coefficients  cA_j
  detail (wavelet) coefficients         cD_j, cD_{j-1}, …, cD_1

Atoms    : wavelet & scaling basis functions at selected (level, position)
Selection: GD threshold on per-subband energy (energy-adaptive)
Weights  : 2^{-j·α}  (coarser scales weighted higher for smooth correction)
           — the wavelet analogue of Sobolev weighting

Physical motivation
-------------------
When the residual eₙ has localised features (boundary layer, spike,
or partial singularity), global Fourier modes are inefficient — the
wavelet multi-resolution representation concentrates information where
the error lives.  The network then learns level-dependent corrections.

Requirements: PyWavelets  (pip install PyWavelets)
"""

import numpy as np
from .base import BasisSolver

try:
    import pywt
    _HAS_PWT = True
except ImportError:
    _HAS_PWT = False


def _gd_threshold(amps: np.ndarray, floor_ratio: float = 0.10) -> float:
    s   = np.sort(amps)[::-1]
    if s[0] < 1e-30:
        return 1e-30
    mad = np.median(np.abs(s - np.median(s)))
    sig = mad / 0.6745 + 1e-12
    n   = len(amps); beta = 1.0 / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return float(max(omega * sig, floor_ratio * s[0]))


class WaveletBasis(BasisSolver):
    """
    Parameters
    ----------
    op            : PDEOperator   must expose .N, .grid
    family        : str           PyWavelets wavelet name, e.g. 'db4', 'sym6'
    levels        : int           decomposition depth  (None → max)
    mode          : str           signal extension mode ('periodization' for periodic,
                                  'symmetric' for Dirichlet-like)
    gd_floor      : float         GD threshold floor ratio
    sobolev_alpha : float         scale weight exponent  2^{-j·α}
    include_approx: bool          include coarsest approximation subband
    """

    def __init__(self, op, family='db4', levels=None, mode='symmetric',
                 gd_floor=0.10, sobolev_alpha=1.0, include_approx=True):
        if not _HAS_PWT:
            raise ImportError("PyWavelets required: pip install PyWavelets")
        self.op             = op
        self._N             = op.N
        self.wavelet        = pywt.Wavelet(family)
        self.levels         = levels or pywt.dwt_max_level(op.N, self.wavelet)
        self.mode           = mode
        self.gd_floor       = gd_floor
        self.sobolev_alpha  = sobolev_alpha
        self.include_approx = include_approx

    # ── helpers ──────────────────────────────────────────────────────────

    def _dwt(self, u: np.ndarray):
        """Forward DWT → list of arrays [cA_J, cD_J, cD_{J-1}, ..., cD_1]."""
        return pywt.wavedec(u, self.wavelet, mode=self.mode, level=self.levels)

    def _idwt(self, coeffs_list) -> np.ndarray:
        """Inverse DWT."""
        rec = pywt.waverec(coeffs_list, self.wavelet, mode=self.mode)
        return rec[:self._N].astype(np.float32)

    def _flatten(self, coeffs_list) -> tuple:
        """
        Flatten DWT coefficient list to 1-D array.
        Returns (flat, slices) where slices is list of (start,stop) per subband.
        """
        flat, slices = [], []
        pos = 0
        for arr in coeffs_list:
            n = len(arr)
            flat.append(arr.astype(np.float32))
            slices.append((pos, pos + n))
            pos += n
        return np.concatenate(flat), slices

    def _unflatten(self, flat: np.ndarray, slices: list,
                   orig_list: list) -> list:
        """Reconstruct DWT coefficient list from flat array."""
        result = []
        for (s, e), ref in zip(slices, orig_list):
            result.append(flat[s:e].copy())
        return result

    def _subband_levels(self, coeffs_list) -> list:
        """Return resolution level for each subband (J, J-1, …, 1 for details; 0 for approx)."""
        J = self.levels
        # coeffs_list = [cA_J, cD_J, cD_{J-1}, ..., cD_1]
        levels_list = [0]  # approximation = level 0 (coarsest)
        for j in range(J, 0, -1):
            levels_list.append(j)
        return levels_list

    # ── BasisSolver interface ────────────────────────────────────────────

    def decompose(self, residual: np.ndarray) -> tuple:
        coeffs_list  = self._dwt(residual.astype(np.float64))
        flat, slices = self._flatten(coeffs_list)
        J            = self.levels

        # Per-subband GD selection
        mask         = np.zeros(len(flat), dtype=bool)
        subband_info = []
        for i, (arr, (s, e)) in enumerate(zip(coeffs_list, slices)):
            amps = np.abs(arr)
            if i == 0 and not self.include_approx:
                # skip approximation subband
                subband_info.append(dict(level=0, s=s, e=e, selected=np.zeros(len(arr), bool)))
                continue
            tau_sb = _gd_threshold(amps, self.gd_floor)
            sel_sb = amps > tau_sb
            if not sel_sb.any():
                sel_sb[np.argmax(amps)] = True
            mask[s:e] = sel_sb
            subband_info.append(dict(level=(0 if i == 0 else J - i + 1),
                                     s=s, e=e, selected=sel_sb))

        coeffs = flat[mask]
        meta   = dict(
            mask=mask, flat=flat, slices=slices,
            orig_list=coeffs_list,
            subband_info=subband_info,
            n_atoms=int(mask.sum()),
            levels=J,
        )
        return coeffs, meta

    def reconstruct(self, coeffs: np.ndarray, meta: dict) -> np.ndarray:
        full            = meta["flat"].copy()
        full[meta["mask"]] = coeffs
        coeffs_list     = self._unflatten(full, meta["slices"], meta["orig_list"])
        return self._idwt(coeffs_list)

    def atom_features(self, residual: np.ndarray, target: np.ndarray,
                      meta: dict) -> tuple:
        target_list, slices = self._flatten(self._dwt(target.astype(np.float64)))
        mask    = meta["mask"]
        tc      = target_list[mask].astype(np.float32)
        a_sel   = np.abs(tc)
        signs   = np.sign(tc); signs[signs == 0] = 1.0

        alphas  = self.sobolev_weights({**meta, "target_coeffs": tc})
        H       = self._spectral_entropy(alphas)

        # Build spatial basis columns: reconstruct indicator vectors for each atom
        # (expensive but correct; cache if needed for large N)
        B = self._build_atom_basis(meta)   # (N, M)
        s = self._log_norm_encode(a_sel, signs, B)

        meta = {
            **meta,
            "target_flat":   target_list,
            "target_coeffs": tc,
            "amps_delta":    a_sel,
            "alphas":        alphas,
            "entropy":       H,
            "target":        target,
            "s_input":       s,
        }
        return s, meta

    def _build_atom_basis(self, meta: dict) -> np.ndarray:
        """
        Build (N, M) matrix of wavelet atom spatial profiles.
        Column i = physical-space representation of the i-th selected atom.
        """
        mask    = meta["mask"]
        flat_sz = len(meta["flat"])
        M       = mask.sum()
        B       = np.zeros((self._N, M), dtype=np.float32)
        atom_idx = 0
        for j_flat, is_sel in enumerate(mask):
            if not is_sel:
                continue
            indicator = np.zeros(flat_sz, dtype=np.float32)
            indicator[j_flat] = 1.0
            coeffs_l = self._unflatten(indicator, meta["slices"], meta["orig_list"])
            rec      = self._idwt(coeffs_l)
            B[:, atom_idx] = rec
            atom_idx += 1
        return B

    def sobolev_weights(self, meta: dict) -> np.ndarray:
        """
        Wavelet Sobolev weights: atoms at finer scales (higher level j)
        get weight 2^{-j·α}.  Combined with target energy.
        """
        tc   = meta.get("target_coeffs", None)
        if tc is None:
            return np.ones(meta["n_atoms"]) / meta["n_atoms"]

        # Build per-atom scale levels
        mask  = meta["mask"]
        flat  = meta["flat"]
        sinfo = meta["subband_info"]
        atom_levels = np.zeros(mask.sum(), dtype=np.float64)
        ai = 0
        for si in sinfo:
            s, e = si["s"], si["e"]
            for k in range(s, e):
                if mask[k]:
                    atom_levels[ai] = si["level"]
                    ai += 1

        scale_w = 2.0 ** (-atom_levels * self.sobolev_alpha)
        w       = scale_w * (tc ** 2)
        w       = np.abs(w)
        return w / (w.sum() + 1e-12)
