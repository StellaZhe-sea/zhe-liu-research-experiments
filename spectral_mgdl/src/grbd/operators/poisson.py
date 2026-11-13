"""
grbd.operators.poisson
======================
1-D Poisson  L[u] = -u_xx  on (0,1) with Dirichlet BC.
Special case of Helmholtz with κ=0.
"""

from .helmholtz import HelmholtzOperator


class PoissonOperator(HelmholtzOperator):
    def __init__(self, N: int):
        super().__init__(N, kappa=0.0)
