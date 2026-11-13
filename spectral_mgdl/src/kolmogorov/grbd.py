"""
GRBD — GradeNet
Cascade of G grades. Each grade:
  1. compute residual rₖ = u* − uₖ
  2. decompose rₖ via basis
  3. predict per-atom weights via MLP
  4. reconstruct correction δₖ = T⁻¹[w ⊙ C]
  5. update uₖ₊₁ = uₖ + δₖ
"""

import torch
import torch.nn as nn
from basis import BasisSolver


class AtomMLP(nn.Module):
    """
    Shared MLP applied independently to each atom's feature vector.
    Input : (B, A, d_feat)
    Output: (B, A)  scalar weight per atom
    """
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),          # weights in (-1, 1); allows sign flip
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat : (B, A, d)
        return self.net(feat).squeeze(-1)   # (B, A)


class GradeBlock(nn.Module):
    """One grade of GRBD."""

    def __init__(self, basis: BasisSolver, hidden_dim: int):
        super().__init__()
        self.basis = basis
        self.mlp   = AtomMLP(basis.feat_dim, hidden_dim)

    def forward(self, u: torch.Tensor, target: torch.Tensor = None,
                residual: torch.Tensor = None):
        """
        u       : (B, N, N)  current field estimate
        target  : (B, N, N)  ground truth u* (supervised)  OR
        residual: (B, N, N)  precomputed r (physics-informed mode)

        Returns (u_new, delta, r)
        """
        if residual is None:
            r = target - u          # supervised
        else:
            r = residual            # physics-informed

        C    = self.basis.decompose(r)                  # (B, A, 2) or (B, A)
        feat = self.basis.atom_features(C)              # (B, A, d)
        w    = self.mlp(feat)                           # (B, A)

        # Apply weights in coefficient space
        if C.dim() == 3:   # Fourier: (B, A, 2) complex-as-real
            w_exp = w.unsqueeze(-1)                     # (B, A, 1)
            C_weighted = C * w_exp                      # (B, A, 2)
        else:              # Wavelet: (B, A) real
            C_weighted = C * w                          # (B, A)

        delta  = self.basis.reconstruct(C_weighted)     # (B, N, N)
        u_new  = u + delta
        return u_new, delta, r


class GradeNet(nn.Module):
    """
    Full GRBD model: G cascaded GradeBlocks sharing the same basis type.

    Args:
        basis      : BasisSolver instance (FourierBasis or WaveletBasis)
        G          : number of grades
        hidden_dim : MLP hidden dimension (same for all grades)
        share_weights: if True, all grades share one GradeBlock (parameter efficient)
    """

    def __init__(self, basis: BasisSolver, G: int, hidden_dim: int = 128,
                 share_weights: bool = False):
        super().__init__()
        self.G = G
        self.basis = basis

        if share_weights:
            block = GradeBlock(basis, hidden_dim)
            self.grades = nn.ModuleList([block] * G)
        else:
            self.grades = nn.ModuleList(
                [GradeBlock(basis, hidden_dim) for _ in range(G)]
            )

    def forward(self, u0: torch.Tensor, target: torch.Tensor):
        """
        u0     : (B, N, N)  initial guess (e.g. zeros or coarse solution)
        target : (B, N, N)  ground truth

        Returns:
            u_final   : (B, N, N)
            u_history : list of (B, N, N), one per grade (for loss / inspection)
            r_history : list of residuals per grade
        """
        u = u0
        u_history = []
        r_history = []

        for grade in self.grades:
            u, delta, r = grade(u, target=target)
            u_history.append(u)
            r_history.append(r)

        return u, u_history, r_history

    def predict(self, u0: torch.Tensor):
        """Inference without target (returns only final u)."""
        u = u0
        with torch.no_grad():
            # We need a dummy target=0 since residual path not yet implemented
            # In inference: pass u0=coarse, target can be set to 0 and we just
            # propagate — the correction will be applied regardless
            # Better: expose a 'residual_fn' callback. For now, use target=u0
            # which gives r=0 (no-op). User should pass a physics residual instead.
            raise NotImplementedError("Use forward() with target for supervised mode.")


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def graded_loss(u_history, target, weights=None):
    """
    Supervised loss summed over all grades.
    Optionally weight later grades more (geometric schedule).

    u_history : list of G tensors (B, N, N)
    target    : (B, N, N)
    weights   : list of G floats (default: equal)
    """
    G = len(u_history)
    if weights is None:
        weights = [1.0] * G

    total = 0.0
    for u_k, w in zip(u_history, weights):
        diff = u_k - target
        mse  = (diff**2).mean()
        total = total + w * mse
    return total / sum(weights)


def rse(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """Relative Squared Error (scalar, for logging)."""
    num = ((u_pred - u_true)**2).sum()
    den = (u_true**2).sum() + 1e-8
    return (num / den).item()
