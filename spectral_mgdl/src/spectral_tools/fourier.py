"""
basis/fourier.py
T_fourier: spectral input encoding based on inv_L(e_n).

Physical meaning of s_n:
  - Each column = one sine basis function at a selected frequency k
  - Amplitude = log-normalised energy of that frequency in the solution error
  - Sign = direction of correction needed at that frequency

Selected frequencies: Gavish-Donoho threshold on |DST(inv_L(e_n))|.
This is v8's encoding — the network sees "where is the solution error in frequency space".
"""

import math
import numpy as np
import torch
from typing import Tuple, List
from basis_mgdl.operators import HelmholtzOp, dst1, idst1


def gd_threshold(amps: np.ndarray, floor: float = 0.10) -> np.ndarray:
    """
    Gavish-Donoho threshold. Returns 0-indexed positions of significant modes.
    floor: minimum fraction of max amplitude (prevents selecting nothing).
    """
    n = len(amps)
    med = np.median(amps)
    mad = np.median(np.abs(amps - med))
    sigma = mad / 0.6745 + 1e-12
    omega = 0.56 / n**3 - 0.95 / n**2 + 1.82 / n + 1.43
    tau = max(omega * sigma, floor * amps.max())
    idx = np.where(amps > tau)[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmax(amps))])
    return idx


def T_fourier(
    e_n: torch.Tensor,
    op: HelmholtzOp,
    gd_floor: float = 0.10,
) -> Tuple[torch.Tensor, List[int], np.ndarray]:
    """
    Build spectral input matrix s_n from residual e_n.

    Args:
        e_n:      residual L[u_n] - f,  shape (N,)
        op:       HelmholtzOp
        gd_floor: GD threshold floor ratio

    Returns:
        s_n:    (N, m) float32 — network input
        k_sel:  list of selected frequencies (1-indexed)
        amps:   (N,) full amplitude spectrum of delta
    """
    # delta = inv_L(e_n) = u_n - u*  (solution-space error)
    delta = op.inv_L(e_n)                          # (N,)
    d_hat = dst1(delta).numpy()                    # (N,)
    amps = np.abs(d_hat)                           # (N,)

    sel = gd_threshold(amps, floor=gd_floor)       # 0-indexed
    k_sel = (sel + 1).tolist()                     # 1-indexed
    m = len(k_sel)

    a = amps[sel]
    signs = np.sign(d_hat[sel])
    a_max = a.max() + 1e-12
    x = op.x.numpy()                              # (N,)

    cols = []
    for i, ki in enumerate(k_sel):
        log_norm = math.log(1 + a[i]) / math.log(1 + a_max)
        col = log_norm * signs[i] * np.sin(math.pi * ki * x)
        cols.append(col)

    s_n = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
    return s_n, k_sel, amps
