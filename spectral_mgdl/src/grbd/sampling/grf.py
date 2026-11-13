"""
grbd.sampling.grf
=================
Gaussian Random Field (GRF) sampling generalised to any PDEOperator.

FNO original (Burgers, periodic):
    u₀ ~ N(0, 625(−Δ + 25I)^{−2})

This module generalises:
    u ~ N(0, σ²(τ²I + |L|)^{−α})

where w_k = σ · (τ² + |λ_k|)^{−α/2}  are the spectral weights,
constructed in the operator's natural eigenbasis → automatically
satisfies correct BC.

Key insight for GRBD:
    The same Sobolev weight w_k used to SAMPLE u* also defines
    per-coefficient importance in the spectral loss — this is the
    FNO logic generalised to all basis types.
"""

import numpy as np
from scipy.fft import idst


def sample_grf(op, alpha: float = 2.0, tau: float = 3.0,
               sigma: float = 1.0, seed=None) -> np.ndarray:
    """
    Sample u ~ N(0, σ²(τ²I + |L|)^{-α}) in the operator's eigenbasis.

    Parameters
    ----------
    op    : PDEOperator  must implement .eigenvalues(), .N
    alpha : float        Sobolev smoothness (higher → smoother)
    tau   : float        regularisation shift
    sigma : float        amplitude scale
    seed  : int | None

    Returns
    -------
    u : (N,) float32  satisfies BC by construction, normalised to max=1
    """
    rng   = np.random.default_rng(seed)
    lam   = np.abs(op.eigenvalues())
    w     = sigma * (tau**2 + lam) ** (-alpha / 2.0)
    xi    = rng.standard_normal(op.N)

    op_cls = type(op).__name__

    if op_cls == "PeriodicDiffusionOperator":
        N     = op.N
        lam_r = op.nu * (2*np.pi * np.fft.rfftfreq(N, d=1.0/N))**2
        w_r   = sigma * (tau**2 + lam_r) ** (-alpha / 2.0)
        xi_c  = (rng.standard_normal(N//2+1) +
                 1j*rng.standard_normal(N//2+1)) / np.sqrt(2)
        u     = np.fft.irfft(w_r * xi_c, n=N).astype(np.float32)
    else:
        # DST eigenbasis (Helmholtz, Poisson, generic Dirichlet)
        u = idst(w * xi, type=1).astype(np.float32)

    u = u / (np.max(np.abs(u)) + 1e-12)
    return u.astype(np.float32)


def sobolev_schedule(op, grade: int, alpha_init: float = 2.0,
                     alpha_decay: float = 0.3, tau: float = 3.0) -> np.ndarray:
    """
    Per-grade Sobolev weights.  As grades increase the residual becomes
    more high-frequency → reduce alpha to focus on finer scales.
        α_n = max(0.5, α_init − n × α_decay)

    Returns w : (N,) float32 normalised
    """
    alpha = max(0.5, alpha_init - grade * alpha_decay)
    lam   = np.abs(op.eigenvalues())
    w     = (tau**2 + lam) ** (-alpha / 2.0)
    return (w / (w.sum() + 1e-12)).astype(np.float32)


def sample_grf_batch(op, n_samples: int, alpha: float = 2.0,
                     tau: float = 3.0, sigma: float = 1.0,
                     seed=None) -> np.ndarray:
    """Batch of GRF samples, shape (n_samples, N) float32."""
    rng = np.random.default_rng(seed)
    return np.stack([
        sample_grf(op, alpha=alpha, tau=tau, sigma=sigma,
                   seed=int(rng.integers(0, 2**31)))
        for _ in range(n_samples)
    ]).astype(np.float32)
