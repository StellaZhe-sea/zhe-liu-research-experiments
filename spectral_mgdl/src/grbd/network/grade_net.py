"""
grbd.network.grade_net
======================
GradeNet: basis-agnostic per-grade network.

Input  : s  (N, M)   feature matrix from BasisSolver.atom_features()
Output : g  (N,)     physical-space correction

Architecture:
  W_sym symmetric M×M mixing → proj M→width → depth×(Linear+Tanh) → Linear→1
"""

import torch
import torch.nn as nn


class GradeNet(nn.Module):
    def __init__(self, n_atoms: int, width: int, depth: int):
        super().__init__()
        self.n_atoms = n_atoms
        self._W_raw  = nn.Parameter(torch.zeros(n_atoms, n_atoms))
        self.mix_act = nn.Tanh()
        self.proj    = nn.Linear(n_atoms, width)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self._W_raw, std=0.01)
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)
        for m in list(self.mlp.modules()) + [self.out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    @property
    def W_sym(self):
        return 0.5 * (self._W_raw + self._W_raw.t())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.mix_act(s @ self.W_sym.t()) + s
        return self.out(self.mlp(self.proj(h))).squeeze(-1)
