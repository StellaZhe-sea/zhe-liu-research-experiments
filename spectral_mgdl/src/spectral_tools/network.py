"""
network.py
Shared components for all three basis frameworks:
  - GradeNet: W_sym frequency mixing + MLP
  - SpectralLoss: in-band weighted + out-of-band penalty
  - LrAdaptor: loss-slope adaptive learning rate
  - HiddenStateBuffer: cross-grade hidden state storage
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from basis_mgdl.operators import dst1


# ── HiddenStateBuffer ─────────────────────────────────────────────────────────

class HiddenStateBuffer:
    """Stores last-layer hidden state of each grade for cross-grade context."""

    def __init__(self):
        self._store: Dict[int, torch.Tensor] = {}

    def push(self, grade_idx: int, h: torch.Tensor):
        self._store[grade_idx] = h.detach().clone()

    def get(self, grade_idx: int) -> Optional[torch.Tensor]:
        return self._store.get(grade_idx, None)

    def clear(self):
        self._store.clear()


# ── GradeNet ──────────────────────────────────────────────────────────────────

class GradeNet(nn.Module):
    """
    Maps spectral input s_n (N, m) → correction g (N,).

    Architecture:
      - W_sym: learnable symmetric (m×m) frequency mixing matrix
        (self-adjoint prior: basis functions interact symmetrically)
      - Residual connection: h = tanh(s @ W_sym) + s
      - MLP: Linear(m, width) → [Tanh → Linear(width, width)] × (depth-1) → Linear(width, 1)
      - Last hidden layer stored for HiddenStateBuffer

    Works for any basis (Fourier / wavelet / wavelet-packet):
    the meaning of m changes, but the architecture does not.
    """

    def __init__(self, m: int, width: int = 128, depth: int = 3):
        super().__init__()
        self.m = m

        # Symmetric frequency mixing (stored as raw, symmetrised on forward)
        self.W_raw = nn.Parameter(torch.randn(m, m) * 0.02)

        # MLP
        layers = [nn.Linear(m, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

        self._last_hidden: Optional[torch.Tensor] = None

    @property
    def W_sym(self) -> torch.Tensor:
        return 0.5 * (self.W_raw + self.W_raw.T)

    def forward(
        self,
        s: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:   (N, m) spectral input
            ctx: (N, d_ctx) optional cross-grade context (from HiddenStateBuffer)
        Returns:
            g:   (N,) correction
        """
        # Symmetric frequency mixing with residual
        h = torch.tanh(s @ self.W_sym) + s          # (N, m)

        # Optional context injection (additive, gated)
        if ctx is not None:
            # Project ctx to m dims and add as soft bias
            h = h + ctx[:, :self.m] if ctx.shape[1] >= self.m else h

        hidden = self.mlp(h)                         # (N, width)
        self._last_hidden = hidden.detach()
        g = self.head(hidden).squeeze(-1)            # (N,)
        return g

    def get_last_hidden(self) -> Optional[torch.Tensor]:
        return self._last_hidden


# ── SpectralLoss ──────────────────────────────────────────────────────────────

class SpectralLoss(nn.Module):
    """
    Frequency-aware loss function.

    in-band:  weighted MSE on selected frequency coefficients
              weight alpha_k = a_k^2 / sum(a_k^2)  (energy-proportional)
    out-of-band: L2 penalty on network output energy at unselected frequencies
                 (prevents spurious leakage into non-residual modes)

    Total: L = loss_ib + lam_oob * loss_oob
    """

    def __init__(self, lam_oob: float = 0.05):
        super().__init__()
        self.lam_oob = lam_oob

    def forward(
        self,
        g: torch.Tensor,
        delta: torch.Tensor,
        k_sel: List[int],
        amps: np.ndarray,
    ) -> torch.Tensor:
        """
        Args:
            g:      (N,) network output (correction)
            delta:  (N,) target = inv_L(e_n)
            k_sel:  selected frequency indices (1-indexed) — Fourier only
                    for wavelet/WPT, pass list of active column indices
            amps:   (N,) or (M,) amplitude array for weighting
        """
        N = g.shape[0]
        g_hat = dst1(g.double()).float()             # (N,)
        d_hat = dst1(delta.double()).float()         # (N,)

        sel = [k - 1 for k in k_sel]                # 0-indexed
        sel = [s for s in sel if 0 <= s < N]

        if len(sel) == 0:
            return ((g - delta) ** 2).mean()

        a = torch.tensor(
            amps[sel] if len(amps) > max(sel) else amps[:len(sel)],
            dtype=torch.float32, device=g.device
        )
        alpha = a ** 2 / (a ** 2).sum().clamp(min=1e-12)

        # In-band: energy-weighted alignment on selected frequencies
        loss_ib = (alpha * (g_hat[sel] - d_hat[sel]) ** 2).sum()

        # Out-of-band: penalise leakage
        mask = torch.ones(N, dtype=torch.bool, device=g.device)
        mask[sel] = False
        loss_oob = g_hat[mask].pow(2).mean() if mask.any() else g.new_zeros(1).squeeze()

        return loss_ib + self.lam_oob * loss_oob


class WaveletSpectralLoss(nn.Module):
    """
    SpectralLoss variant for wavelet/WPT bases.
    In-band loss is computed in physical space (MSE), since wavelet
    basis vectors are not orthogonal in DST domain.
    Out-of-band: DST leakage penalty still applies.
    """

    def __init__(self, lam_oob: float = 0.05):
        super().__init__()
        self.lam_oob = lam_oob

    def forward(
        self,
        g: torch.Tensor,
        delta: torch.Tensor,
        sel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Physical space MSE (main loss)
        loss_main = ((g - delta) ** 2).mean()

        # DST out-of-band leakage penalty
        g_hat = dst1(g.double()).float()
        if sel_mask is not None and sel_mask.any():
            oob_mask = ~sel_mask
            loss_oob = g_hat[oob_mask].pow(2).mean() if oob_mask.any() else g.new_zeros(1).squeeze()
        else:
            loss_oob = g_hat.pow(2).mean() * 0.0

        return loss_main + self.lam_oob * loss_oob


# ── LrAdaptor ─────────────────────────────────────────────────────────────────

class LrAdaptor:
    """
    Adaptive learning rate based on loss tail slope.
    Increases lr when converging well, decreases when stagnating.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 2e-3,
        lr_min: float = 1e-4,
        lr_max: float = 5e-3,
        window: int = 30,
    ):
        self.opt = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.window = window
        self._losses: List[float] = []

    def step(self, loss_val: float):
        self._losses.append(loss_val)
        if len(self._losses) < self.window * 2:
            return
        tail = self._losses[-self.window:]
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
        ratio = abs(slope) / (abs(tail[0]) + 1e-12)
        lr = self._get_lr()
        if slope < 0 and ratio > 0.01:
            new_lr = lr
        elif ratio > 0.5:
            new_lr = lr * 0.5
        elif ratio < 0.01:
            new_lr = lr * 1.2
        else:
            new_lr = lr * 0.7
        self._set_lr(float(np.clip(new_lr, self.lr_min, self.lr_max)))

    def _get_lr(self) -> float:
        return self.opt.param_groups[0]['lr']

    def _set_lr(self, lr: float):
        for pg in self.opt.param_groups:
            pg['lr'] = lr
