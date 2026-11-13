"""
basis/wavelet.py
T_wavelet: multi-resolution input encoding based on DWT of inv_L(e_n).

Physical meaning of s_n:
  - Each column = one wavelet basis function psi_{j,k}
  - Amplitude = energy of that scale/location in the solution error
  - Captures WHERE in space and at WHAT scale the error is concentrated

Natural advantage over Fourier:
  - Besov space approximation: handles discontinuities and local singularities
  - When a(x) has jumps, error concentrates locally — DWT sees this, DST cannot

Truncation: keep coefficients whose cumulative energy >= energy_ratio of total.
inv_L: still exact DST (shared with Fourier) — wavelets only affect T_basis.
"""

import math
import numpy as np
import torch
import pywt
from typing import Tuple, List, Dict
from basis_mgdl.operators import HelmholtzOp, dst1


def _dwt_coeffs(signal: np.ndarray, wavelet: str, level: int):
    """
    Return DWT coefficients as list [(cA_J, cD_J, cD_{J-1}, ..., cD_1)].
    Uses periodic extension to avoid boundary issues on uniform grid.
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)
    return coeffs  # [cA_J, cD_J, cD_{J-1}, ..., cD_1]


def _energy_threshold(coeffs: list, energy_ratio: float = 0.99):
    """
    Select wavelet coefficients by cumulative energy.
    Returns list of (level_idx, position_idx) tuples (0-indexed).
    level_idx: 0 = approx, 1..J = detail (finest first in pywt output)
    """
    # Flatten all detail coefficients with their level/position labels
    entries = []
    for lvl, c in enumerate(coeffs):
        for pos, val in enumerate(c):
            entries.append((abs(val)**2, lvl, pos))

    total_energy = sum(e[0] for e in entries) + 1e-12
    entries.sort(reverse=True)

    selected = []
    cumulative = 0.0
    for energy, lvl, pos in entries:
        selected.append((lvl, pos))
        cumulative += energy
        if cumulative / total_energy >= energy_ratio:
            break

    return selected


def _reconstruct_basis_vector(
    lvl: int, pos: int, coeffs: list, wavelet: str, N: int
) -> np.ndarray:
    """
    Reconstruct a single wavelet basis vector psi_{lvl,pos} in physical space.
    Sets coefficient (lvl, pos) to 1, everything else to 0, then idwt.
    """
    zero_coeffs = [np.zeros_like(c) for c in coeffs]
    zero_coeffs[lvl][pos] = 1.0
    basis_vec = pywt.waverec(zero_coeffs, wavelet, mode='periodization')
    # trim or pad to exactly N
    if len(basis_vec) > N:
        basis_vec = basis_vec[:N]
    elif len(basis_vec) < N:
        basis_vec = np.pad(basis_vec, (0, N - len(basis_vec)))
    return basis_vec


def T_wavelet(
    e_n: torch.Tensor,
    op: HelmholtzOp,
    wavelet: str = 'db4',
    energy_ratio: float = 0.99,
    max_level: int = None,
) -> Tuple[torch.Tensor, List[Tuple[int,int]], np.ndarray]:
    """
    Build wavelet input matrix s_n from residual e_n.

    Args:
        e_n:          residual L[u_n] - f,  shape (N,)
        op:           HelmholtzOp
        wavelet:      pywt wavelet name (default 'db4')
        energy_ratio: fraction of energy to retain
        max_level:    DWT decomposition levels (default: log2(N)-2)

    Returns:
        s_n:      (N, m) float32 — network input
        sel_info: list of (level, position) pairs selected
        amps:     (N_coeffs,) flattened coefficient amplitudes
    """
    N = op.N
    if max_level is None:
        max_level = int(math.log2(N)) - 2

    # delta = inv_L(e_n) = u_n - u*
    delta = op.inv_L(e_n).numpy()                  # (N,)

    # DWT decomposition
    coeffs = _dwt_coeffs(delta, wavelet, max_level)

    # Energy-based selection
    sel_info = _energy_threshold(coeffs, energy_ratio)

    # Build s_n: each column is a normalised wavelet basis vector
    # weighted by the signed coefficient value (log-normalised amplitude)
    coeff_vals = np.array([coeffs[lvl][pos] for lvl, pos in sel_info])
    amps_sel = np.abs(coeff_vals)
    a_max = amps_sel.max() + 1e-12
    signs = np.sign(coeff_vals)

    cols = []
    for i, (lvl, pos) in enumerate(sel_info):
        log_norm = math.log(1 + amps_sel[i]) / math.log(1 + a_max)
        basis_vec = _reconstruct_basis_vector(lvl, pos, coeffs, wavelet, N)
        col = log_norm * signs[i] * basis_vec
        cols.append(col)

    s_n = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)

    # For logging: full flattened amplitude array
    all_amps = np.concatenate([np.abs(c) for c in coeffs])

    return s_n, sel_info, all_amps
