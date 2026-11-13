"""
GRBD — Graded Residual Basis Decomposition
===========================================
Two-suite modular PDE solver framework.

Suite 1 (Spectral):   DSTBasis | FourierBasis | ChebyshevBasis
Suite 2 (Multiscale): WaveletBasis | WaveletPacketBasis

Quick start:
    from grbd.operators import HelmholtzOperator
    from grbd.basis     import DSTBasis
    from grbd.sampling  import sample_grf
    from grbd.solver    import GradeSolver

    op     = HelmholtzOperator(N=256, kappa=200)
    u_star = sample_grf(op, alpha=2.0, tau=3.0, seed=42)
    solver = GradeSolver(op, DSTBasis(op), op.make_f(u_star), u_star)
    solver.solve(n_grades=8, n_epochs=2000)
"""
from .operators import HelmholtzOperator, PoissonOperator, PeriodicDiffusionOperator
from .basis     import DSTBasis, FourierBasis, ChebyshevBasis, WaveletBasis, WaveletPacketBasis
from .sampling  import sample_grf, sample_grf_batch, sobolev_schedule
from .solver    import GradeSolver
from .viz       import plot_grade, plot_summary, plot_comparison

__version__ = "1.0.0"
__all__ = [
    "HelmholtzOperator","PoissonOperator","PeriodicDiffusionOperator",
    "DSTBasis","FourierBasis","ChebyshevBasis","WaveletBasis","WaveletPacketBasis",
    "sample_grf","sample_grf_batch","sobolev_schedule",
    "GradeSolver",
    "plot_grade","plot_summary","plot_comparison",
]
