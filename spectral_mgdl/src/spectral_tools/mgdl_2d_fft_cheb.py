"""MGDL 2D framework (variant 1): FFT in x, Chebyshev in y.

Multi-grade solver for a 2D variable-coefficient Helmholtz-type
operator with FFT-based grading in x and Chebyshev grading in y."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import fft, ifft, rfft, irfft, fftfreq
import time


from mgdl_2d_v1 import (Operator2D, GradeNet, LrAdaptor,
                          gd_threshold_2d, GradeSolver2D)


def cheb_nodes(N: int) -> np.ndarray:
    """
    Chebyshev-Gauss-Lobatto nodes on [-1, 1], N+1 points (including endpoints).
    y_j = cos(jπ/N),  j = 0, 1, …, N
    """
    j = np.arange(N + 1, dtype=np.float64)
    return np.cos(np.pi * j / N)          # shape (N+1,)


def cheb_diff_matrix(N: int) -> np.ndarray:
    """
    Chebyshev differentiation matrix D of size (N+1) × (N+1).
    D @ u ≈ du/dy  on Gauss-Lobatto nodes.
    Standard Fornberg/Trefethen formula.
    """
    y = cheb_nodes(N)
    c = np.ones(N + 1); c[0] = 2; c[N] = 2
    c *= (-1) ** np.arange(N + 1)
    Y = np.tile(y, (N + 1, 1))
    dY = Y - Y.T
    D = np.outer(c, 1.0 / c) / (dY + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))           # diagonal correction
    return D.astype(np.float64)           # (N+1, N+1)


def cheb_transform(u: np.ndarray) -> np.ndarray:
    """
    Forward Chebyshev transform: u (N+1,) → c (N+1,)
    c_k = (2/N) Σ_{j=0}^N '' u_j cos(jkπ/N)
    (double prime: j=0 and j=N weighted by 1/2)
    """
    N = u.shape[0] - 1
    # Use DCT-I via FFT trick (scipy has no direct dct-I for this convention)
    # Reflect: [u_0, u_1, ..., u_N, u_{N-1}, ..., u_1] → length 2N, then rfft
    v = np.concatenate([u, u[-2:0:-1]])  # (2N,)
    V = np.fft.rfft(v)                   # (N+1,) complex
    c = V.real / N
    c[0] /= 2; c[N] /= 2
    return c.astype(np.float64)


def cheb_itransform(c: np.ndarray) -> np.ndarray:
    """Inverse Chebyshev transform: c (N+1,) → u (N+1,)."""
    N = c.shape[0] - 1
    C = c.copy()
    C[0] *= 2; C[N] *= 2
    # irfft on length 2N signal
    V = np.concatenate([C, C[-2:0:-1]]) * N
    u = np.fft.irfft(V)[:N + 1]
    return u.astype(np.float64)


def cheb_laplacian_1d(N: int) -> tuple:
    """
    Returns (D2, y, w) where:
      D2 : (N+1)×(N+1) second-derivative matrix in physical space
      y  : Gauss-Lobatto nodes on [-1, 1], length N+1
      w  : Clenshaw-Curtis weights (for integration)
    """
    D  = cheb_diff_matrix(N)
    D2 = D @ D
    y  = cheb_nodes(N)
    # Clenshaw-Curtis weights
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    w[0] = w[N] = 1.0 / (N * N - 1)
    j_idx = np.arange(1, N)
    s = np.zeros(len(j_idx))
    for k in range(1, N // 2 + 1):
        b = 2.0 if k < N // 2 else 1.0
        s += b / (1 - 4 * k * k) * np.cos(2 * k * theta[j_idx])
    w[1:-1] = (2.0 / N) * (1 + s)
    return D2, y, w


# Operator 1：FFT × Chebyshev  (periodic × non-periodic)


class FFTChebOperator2D(Operator2D):
    r"""
    Solves:  L[u] = f  on  [0, 2π] × [-1, 1]

        L = −(∂²/∂x² + α ∂²/∂y²) + κ²

    x  : periodic (length 2π), Fourier basis, Nx modes
    y  : Dirichlet BC (u(x,±1)=0), Chebyshev basis, Ny+1 Gauss-Lobatto pts
         interior only: Ny-1 points (indices 1..Ny-1)

    Parameters
    ----------
    Nx    : number of Fourier modes (must be even)
    Ny    : polynomial degree (Ny+1 total points, Ny-1 interior)
    alpha : ratio of y-diffusion coefficient (default 1.0)
    kappa : reaction coefficient  (default 0.0)
    Lx    : x domain length  (default 2π)
    """

    def __init__(self, Nx: int, Ny: int,
                 alpha: float = 1.0, kappa: float = 0.0,
                 Lx: float = 2 * np.pi):
        assert Nx % 2 == 0, "Nx must be even for real FFT"
        self._Nx   = Nx
        self._Ny   = Ny          # polynomial degree; interior pts = Ny-1
        self.alpha = alpha
        self.kappa = kappa
        self.Lx    = Lx

        # ── Fourier wavenumbers (x) ──────────────────────────
        # rfft of length Nx gives Nx//2+1 complex modes
        # physical wavenumbers: kx = 2π/Lx * [0, 1, …, Nx/2]
        self._kx = (2 * np.pi / Lx) * np.arange(Nx // 2 + 1, dtype=np.float64)
        # Eigenvalue contribution from x: kx²
        self._eigx = self._kx ** 2                   # (Nx//2+1,)

        # ── Chebyshev in y ──────────────────────────────────
        # Full nodes including endpoints: (Ny+1,)
        # Interior nodes used for PDE: indices 1..Ny-1 → Ny-1 points
        D2_full, y_full, _ = cheb_laplacian_1d(Ny)
        self._y_full  = y_full                        # (Ny+1,) on [-1,1]
        # Extract interior block: rows/cols 1..Ny-1
        self._D2_int  = D2_full[1:Ny, 1:Ny]          # (Ny-1, Ny-1)
        self._y_int   = y_full[1:Ny]                 # (Ny-1,)
        self._Nyi     = Ny - 1                       # interior count

        # ── Precompute per-mode LU factorizations ──────────────
        # For each Fourier mode kx_k, the y-problem is:
        #   (kx_k² I − α D2_int + κ² I) U_k = F_k
        # D2_int is NOT symmetric → use direct LU solve per mode.
        # Precompute and cache the system matrices + lu factors.
        from scipy.linalg import lu_factor
        self._lu_cache = []   # list of (lu, piv) for each kx mode
        for kx_k in self._kx:
            A = (kx_k**2 + kappa**2) * np.eye(Ny-1) - alpha * self._D2_int
            self._lu_cache.append(lu_factor(A))

    @property
    def Nx(self): return self._Nx

    @property
    def Ny(self): return self._Nyi      # solver sees interior point count

    @property
    def wavenumbers(self):
        return self._kx, self._y_int

    # ── Grid ────────────────────────────────────────────────
    @property
    def x_grid(self) -> np.ndarray:
        """x ∈ [0, Lx), Nx equispaced points."""
        return np.linspace(0, self.Lx, self._Nx, endpoint=False)

    @property
    def y_grid(self) -> np.ndarray:
        """Chebyshev interior nodes on [-1, 1]."""
        return self._y_int

    # ── Transforms ──────────────────────────────────────────
    def transform_x(self, u: np.ndarray) -> np.ndarray:
        """
        FFT along x-axis.
        u: (..., Nx) → û: (..., Nx//2+1) complex
        Returns complex array.
        """
        return rfft(u, axis=-1, norm='forward')      # norm='forward': divide by Nx

    def itransform_x(self, u_hat: np.ndarray) -> np.ndarray:
        """irfft along x-axis. u_hat: (..., Nx//2+1) → u: (..., Nx) real."""
        return irfft(u_hat, n=self._Nx, axis=-1, norm='forward')

    def transform_y(self, u: np.ndarray) -> np.ndarray:
        """
        Chebyshev transform of interior values.
        u: (Nx, Nyi) → c: (Nx, Nyi)  Chebyshev coefficients.
        Applied row-by-row (each x-slice).
        """
        c = np.zeros_like(u, dtype=np.float64)
        for i in range(u.shape[0]):
            # Embed interior into full array (zero at boundaries)
            u_full = np.zeros(self._Ny + 1)
            u_full[1:self._Ny] = u[i]
            c[i] = cheb_transform(u_full)[1:self._Ny]
        return c.astype(np.float32)

    def transform(self, u: np.ndarray) -> np.ndarray:
        """
        Full 2D forward transform: FFT_x ∘ Cheb_y.
        u: (Nx, Nyi)  real → returns complex (Nx//2+1, Nyi).
        Stored as float32 with shape (Nx//2+1, Nyi, 2) for real/imag.
        For internal use returns complex128.
        """
        # Step 1: FFT in x  → (Nx//2+1, Nyi) complex
        u_hat_x = self.transform_x(u)                # (Nx//2+1, Nyi) complex? No:
        # rfft axis=-1: u (Nx, Nyi) → û (Nx, Nyi//2+1)? No, axis=-1 is wrong here
        # u has shape (Nx, Nyi): FFT along x = axis 0
        u_hat_x = rfft(u.astype(np.float64), axis=0, norm='forward')   # (Nx//2+1, Nyi)
        # Step 2: Cheb in y for each Fourier mode
        Nk   = self._Nx // 2 + 1
        u_hat = np.zeros((Nk, self._Nyi), dtype=np.complex128)
        for k in range(Nk):
            u_k_full = np.zeros(self._Ny + 1, dtype=np.complex128)
            u_k_full[1:self._Ny] = u_hat_x[k]       # (Nyi,)
            c_real = cheb_transform(u_k_full.real)
            c_imag = cheb_transform(u_k_full.imag)
            u_hat[k] = (c_real + 1j * c_imag)[1:self._Ny]
        return u_hat     # (Nx//2+1, Nyi) complex

    def itransform(self, u_hat: np.ndarray) -> np.ndarray:
        """Inverse: (Nx//2+1, Nyi) complex → (Nx, Nyi) real."""
        Nk = self._Nx // 2 + 1
        u_hat_x = np.zeros((Nk, self._Nyi), dtype=np.complex128)
        for k in range(Nk):
            u_k = np.zeros(self._Ny + 1, dtype=np.complex128)
            u_k[1:self._Ny] = u_hat[k]
            c_real = cheb_itransform(u_k.real)
            c_imag = cheb_itransform(u_k.imag)
            u_hat_x[k] = (c_real + 1j * c_imag)[1:self._Ny]
        u = irfft(u_hat_x, n=self._Nx, axis=0, norm='forward')  # (Nx, Nyi)
        return u.astype(np.float32)

    # ── PDE operator ────────────────────────────────────────
    def _apply_L_x_hat(self, u_hat_x: np.ndarray) -> np.ndarray:
        """
        Apply L in Fourier-x × physical-y space.
        u_hat_x: (Nk, Nyi) complex  (rfft along x, physical in y)
        Returns Lu_hat_x: same shape
        L_k U_k = (kx_k² + κ²) U_k − α D2_int @ U_k
        """
        Lu_hat_x = np.zeros_like(u_hat_x)
        for k, kx_k in enumerate(self._kx):
            U_k = u_hat_x[k]
            Lu_hat_x[k] = (kx_k**2 + self.kappa**2)*U_k - self.alpha*(self._D2_int @ U_k)
        return Lu_hat_x

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """e = L[u] − f. u, f: (Nx, Nyi) real.
        All operations in Fourier-x × physical-y space for correctness."""
        from scipy.fft import rfft, irfft
        u_hat_x  = rfft(u.astype(np.float64), axis=0, norm='forward')
        Lu_hat_x = self._apply_L_x_hat(u_hat_x)
        Lu       = irfft(Lu_hat_x, n=self._Nx, axis=0, norm='forward').real
        return (Lu - f).astype(np.float32)

    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        """L^{-1}(rhs). For each Fourier mode k, solve the Nyi×Nyi system
        using precomputed LU factorisation."""
        from scipy.fft import rfft, irfft
        from scipy.linalg import lu_solve
        rhs_hat_x = rfft(rhs.astype(np.float64), axis=0, norm='forward')
        sol_hat_x = np.zeros_like(rhs_hat_x)
        for k, lu_piv in enumerate(self._lu_cache):
            sol_hat_x[k] = lu_solve(lu_piv, rhs_hat_x[k])
        return irfft(sol_hat_x, n=self._Nx, axis=0, norm='forward').real.astype(np.float32)

    def make_solution(self, seed: int = 42) -> tuple:
        """Manufacture smooth (u_star, f) pair."""
        rng   = np.random.default_rng(seed)
        x     = self.x_grid              # (Nx,)
        y     = self._y_int              # (Nyi,)

        # Low-frequency Fourier × low-order Chebyshev combination
        u_star = np.zeros((self._Nx, self._Nyi), dtype=np.float32)
        for kx in [1, 2, 3]:
            for nc in [1, 2, 3]:
                a = rng.standard_normal()
                # Cheb polynomial T_nc evaluated at interior nodes
                Tnc = np.cos(nc * np.arccos(y))        # (Nyi,)
                # Enforce homogeneous Dirichlet: multiply by (1-y²)
                Tnc = Tnc * (1 - y ** 2)
                u_star += a * np.sin(kx * x[:, None]) * Tnc[None, :]
        u_star = u_star.astype(np.float32)
        f = self.residual(u_star, np.zeros((self._Nx, self._Nyi)))
        return u_star, f


class FFTOperator2D(Operator2D):
    r"""
    Solves:  L[u] = f  on  [0, Lx] × [0, Ly] with periodic BC.

        L = −(∂²/∂x² + ∂²/∂y²) + κ²

    Both directions: FFT (complex exponential basis).
    Operates on REAL fields (u real → use rfft in x, full fft in y
    or rfft2 for 2D real).

    Parameters
    ----------
    Nx, Ny : grid points in x and y (both must be even)
    kappa  : reaction coefficient (default 0 for pure Laplacian)
    Lx, Ly : domain lengths (default 2π × 2π)
    """

    def __init__(self, Nx: int, Ny: int,
                 kappa: float = 0.0,
                 Lx: float = 2 * np.pi, Ly: float = 2 * np.pi):
        self._Nx  = Nx; self._Ny = Ny
        self.kappa = kappa
        self.Lx = Lx; self.Ly = Ly

        # Wavenumbers: rfft2 returns (Nx, Ny//2+1) complex
        kx = (2 * np.pi / Lx) * np.fft.rfftfreq(Nx, d=1.0 / Nx)  # (Nx//2+1 ... wrong)
        # numpy rfftfreq: fftfreq then keep first half
        kx_full = (2 * np.pi / Lx) * np.fft.fftfreq(Nx, d=1.0 / Nx)   # (Nx,)
        ky_half = (2 * np.pi / Ly) * np.fft.rfftfreq(Ny, d=1.0 / Ny)   # (Ny//2+1,)

        # For rfft2 output shape (Nx, Ny//2+1):
        self._kx2 = kx_full[:, None] ** 2                  # (Nx, 1)
        self._ky2 = ky_half[None, :] ** 2                  # (1, Ny//2+1)
        self._eigvals = self._kx2 + self._ky2 + kappa ** 2 # (Nx, Ny//2+1)
        # Avoid division by zero at (0,0) if kappa=0
        self._eigvals[self._eigvals == 0] = 1.0

    @property
    def Nx(self): return self._Nx

    @property
    def Ny(self): return self._Ny

    @property
    def wavenumbers(self):
        kx = np.fft.fftfreq(self._Nx, d=1.0 / self._Nx)
        ky = np.fft.rfftfreq(self._Ny, d=1.0 / self._Ny)
        return np.meshgrid(kx, ky, indexing='ij')

    def transform(self, u: np.ndarray) -> np.ndarray:
        """rfft2: (Nx,Ny) real → (Nx, Ny//2+1) complex."""
        return np.fft.rfft2(u.astype(np.float64), norm='forward')

    def itransform(self, u_hat: np.ndarray) -> np.ndarray:
        """irfft2: (Nx, Ny//2+1) complex → (Nx, Ny) real."""
        return np.fft.irfft2(u_hat, s=(self._Nx, self._Ny),
                              norm='forward').astype(np.float32)

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        u_hat  = self.transform(u)
        Lu_hat = self._eigvals * u_hat
        return (self.itransform(Lu_hat) - f).astype(np.float32)

    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        rhs_hat = self.transform(rhs)
        return self.itransform(rhs_hat / self._eigvals).astype(np.float32)

    def make_solution(self, seed: int = 42) -> tuple:
        rng = np.random.default_rng(seed)
        x = np.linspace(0, self.Lx, self._Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self._Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        u_star = np.zeros((self._Nx, self._Ny), dtype=np.float32)
        for kx in [1, 2]:
            for ky in [1, 2]:
                a = rng.standard_normal()
                u_star += a * np.sin(kx * X) * np.cos(ky * Y)
        u_star = u_star.astype(np.float32)
        f = self.residual(u_star, np.zeros_like(u_star))
        return u_star, f


class SpectralLossHybrid(nn.Module):
    """
    Weighted physical-space MSE loss.

    Decomposition:
      loss_in  = Σ_{k in K_n} α_k (g - target)^T (φ_k φ_k^T) (g - target)
               = ||diag(α^{1/2}) S^T (g - target)||²  where S columns = φ_k / |φ_k|
      loss_oob = λ ||(I − P)(g − target)||²  where P = projection onto span(K_n)

    This correctly constrains g in ALL directions:
    - in-band: energy-weighted alignment with target on K_n
    - out-of-band: suppress leakage outside K_n (when λ_oob > 0)

    Since S may have non-orthogonal columns (FFT basis has unit-norm but
    overlapping support), we normalise each column before use.

    u shape: (Nx*Ny,) flattened physical space.
    """

    def __init__(self, op: Operator2D,
                 mask_2d: np.ndarray,
                 alphas:  np.ndarray,       # (n_freqs,)
                 s_mat:   np.ndarray,       # (Nx*Ny, n_freqs) raw basis columns
                 lambda_oob: float = 0.0):
        super().__init__()
        self.Nx = op.Nx; self.Ny = op.Ny
        self.lambda_oob = lambda_oob

        # Normalise columns so each φ_k has unit L2 norm
        col_norms = np.linalg.norm(s_mat, axis=0, keepdims=True) + 1e-12  # (1, K)
        S_norm = (s_mat / col_norms).astype(np.float32)    # (N, K) unit-norm cols

        # Weighted: Ŝ_k = α_k^{1/2} * S_norm_k
        sqrt_alpha = np.sqrt(alphas).astype(np.float32)    # (K,)
        S_weighted = S_norm * sqrt_alpha[None, :]           # (N, K)

        self.register_buffer('S_w',  torch.from_numpy(S_weighted))  # (N,K)
        self.register_buffer('S_n',  torch.from_numpy(S_norm))      # (N,K) for OOB

    def forward(self, g_flat: torch.Tensor, target_flat: torch.Tensor):
        diff = g_flat - target_flat.detach()               # (N,)
        # Physical-space MSE — directly minimises RSE
        loss_in = (diff ** 2).mean()
        loss_oob = g_flat.new_zeros(1).squeeze()
        return loss_in, loss_in.item(), 0.0


# build_spectral_input_hybrid: For FFTCheb and FFT2D operators
# Key difference from DST version:
#   transform returns COMPLEX array → use |·| for GD selection
#   basis matrix uses real part of complex exponentials × Cheb polys


def build_spectral_input_hybrid(residual_e: np.ndarray,
                                 target_delta: np.ndarray,
                                 op: Operator2D,
                                 gd_floor: float = 0.01):
    """
    Build spectral input for FFTCheb or FFT2D operators.

    Differences from build_spectral_input_2d (DST version):
    - transform() may return complex array → |·| for GD selection
    - basis functions: cos(kx·x)·φ_y(y) and sin(kx·x)·φ_y(y) are SEPARATE columns
      (real and imaginary parts of the Fourier basis)

    Returns
    -------
    s     : (Nx*Ny, |K_n|) float32   (doubled for real/imag of complex modes)
    meta  : dict
    """
    Nx, Ny = op.Nx, op.Ny

    e_hat   = op.transform(residual_e)     # complex or real
    tgt_hat = op.transform(target_delta)

    is_complex = np.iscomplexobj(e_hat)

    if is_complex:
        amps_res = np.abs(e_hat).astype(np.float32)    # (Nk, Ny) or (Nx, Ny)
    else:
        amps_res = np.abs(e_hat).astype(np.float32)

    tau     = gd_threshold_2d(amps_res.ravel(), floor_ratio=gd_floor)
    mask_2d = amps_res > tau
    if not mask_2d.any():
        mask_2d.ravel()[np.argmax(amps_res.ravel())] = True

    active_idx = np.argwhere(mask_2d)      # (|K|, 2)
    n_freqs_raw = len(active_idx)

    # δ_n info
    if is_complex:
        amps_delta = np.abs(tgt_hat).astype(np.float32)
    else:
        amps_delta = np.abs(tgt_hat).astype(np.float32)
    a_sel   = amps_delta[mask_2d]          # (|K|,)
    energy  = a_sel ** 2
    alphas  = energy / (energy.sum() + 1e-12)
    H       = float(-np.sum(alphas * np.log(alphas + 1e-12)))

    log_amps = np.log(1.0 + a_sel)
    log_max  = np.log(1.0 + a_sel.max() + 1e-12)
    log_norm = (log_amps / log_max).astype(np.float32)

    # ── Basis matrix ────────────────────────────────────────
    # For FFTCheb: basis_k(x,y) = e^{ikx·x} · T_{ky}(y)
    # Represented as two real columns: cos × T, sin × T
    # → s shape: (Nx*Nyi, 2·|K|) for complex modes
    # For DC and Nyquist (purely real): only cos column

    if isinstance(op, FFTChebOperator2D):
        x   = op.x_grid                   # (Nx,)
        y   = op.y_grid                   # (Nyi,)
        Nk  = op._Nx // 2 + 1
        kx_all = op._kx                   # (Nk,)

        cols_cos = []
        cols_sin = []
        cols_ln  = []
        cols_al  = []
        for idx, (ki, yi) in enumerate(active_idx):
            # ki: index in Nk (Fourier), yi: index in Nyi (Cheb)
            kx_val = kx_all[ki]
            # Chebyshev polynomial T_{yi}(y_int) on interior nodes
            Ty = np.cos(yi * np.arccos(y))   # (Nyi,)  T_{yi}(y)
            Ty = Ty * (1 - y ** 2)            # enforce zero BC

            phi_cos = np.cos(kx_val * x[:, None]) * Ty[None, :]   # (Nx, Nyi)
            phi_sin = np.sin(kx_val * x[:, None]) * Ty[None, :]

            phi_cos = phi_cos.ravel().astype(np.float32)
            phi_sin = phi_sin.ravel().astype(np.float32)

            ln = log_norm[idx]
            al = alphas[idx]

            cols_cos.append(ln * phi_cos)
            if ki > 0 and ki < Nk - 1:       # skip DC and Nyquist (real-only)
                cols_sin.append(ln * phi_sin)
                cols_al.append(al)
                cols_al.append(al)
                cols_ln.append(ln); cols_ln.append(ln)
            else:
                cols_al.append(al)
                cols_ln.append(ln)

        s = np.stack(cols_cos + cols_sin, axis=1)   # (Nx*Nyi, n_freqs)
        n_freqs = s.shape[1]
        alphas_out = np.array(cols_al[:n_freqs_raw] +
                              cols_al[n_freqs_raw:], dtype=np.float32)
        # Trim to n_freqs
        alphas_out = np.array(cols_al, dtype=np.float32)[:n_freqs]

    elif isinstance(op, FFTOperator2D):
        x  = np.linspace(0, op.Lx, op._Nx, endpoint=False)
        y  = np.linspace(0, op.Ly, op._Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')

        kx_all = np.fft.fftfreq(op._Nx, d=1.0 / op._Nx)  # (Nx,)
        ky_all = np.fft.rfftfreq(op._Ny, d=1.0 / op._Ny)  # (Ny//2+1,)

        cols = []
        alphas_list = []
        for idx, (ki, kj) in enumerate(active_idx):
            kx_val = 2 * np.pi / op.Lx * kx_all[ki]
            ky_val = 2 * np.pi / op.Ly * ky_all[kj]
            phi_cos = (np.cos(kx_val * X + ky_val * Y)).ravel().astype(np.float32)
            phi_sin = (np.sin(kx_val * X + ky_val * Y)).ravel().astype(np.float32)
            ln = log_norm[idx]; al = alphas[idx]
            cols.append(ln * phi_cos)
            if not (ki == 0 and kj == 0):
                cols.append(ln * phi_sin)
                alphas_list.extend([al, al])
            else:
                alphas_list.append(al)

        s = np.stack(cols, axis=1)
        n_freqs = s.shape[1]
        alphas_out = np.array(alphas_list, dtype=np.float32)[:n_freqs]
        H = float(-np.sum((alphas_out / (alphas_out.sum()+1e-12)) *
                           np.log(alphas_out / (alphas_out.sum()+1e-12) + 1e-12)))
    else:
        raise ValueError("Unsupported operator type for hybrid input builder")

    meta = dict(
        active_idx=active_idx,
        mask_2d=mask_2d,
        tau=tau,
        n_freqs=n_freqs,
        entropy=H,
        alphas=alphas_out,
        log_norm=log_norm,
        amps_res=amps_res,
        amps_delta=amps_delta,
        e_hat=e_hat,
        tgt_hat=tgt_hat,
        target=target_delta,
        s_input=s,
    )
    return s, meta


# GradeSolverHybrid  — wraps GradeSolver2D with hybrid loss/input


class GradeSolverHybrid:
    """
    Multi-grade solver for FFTCheb / FFT2D operators.
    Identical control flow to GradeSolver2D; only loss and input builder differ.
    """

    def __init__(self, op, f, u_star,
                 base_width=64, base_depth=3, lambda_oob=0.0,
                 gd_floor=0.01, stop_min_improve=0.20,
                 base_lr=1e-3, lr_min=1e-4, lr_max=5e-3,
                 device='cpu'):
        self.op   = op; self.f = f.astype(np.float32)
        self.u_star = u_star.astype(np.float32)
        self.base_width = base_width; self.base_depth = base_depth
        self.lambda_oob = lambda_oob; self.gd_floor = gd_floor
        self.stop_min_improve = stop_min_improve
        self.device = torch.device(device)
        self.grades = []; self.infos = []
        self._u  = np.zeros((op.Nx, op.Ny), dtype=np.float32)
        self._next_residual = None
        self.lr_adaptor = LrAdaptor(base_lr=base_lr, lr_min=lr_min, lr_max=lr_max)

    def _config(self, H, nk, rel_err):
        depth = max(2, min(self.base_depth + int(H * 1.5), 8))
        width_cap = 256 if rel_err < 1e-3 else (512 if rel_err < 1e-2 else 1024)
        return min(max(self.base_width, 4 * nk), width_cap), depth

    def _should_stop(self):
        if len(self.infos) < 2: return False
        def impr(i):
            eb = self.infos[i]['rel_err_before']; ea = self.infos[i]['rel_err_after']
            return (eb - ea) / (eb + 1e-30)
        i0 = impr(-1); i1 = impr(-2)
        stop = ((i0 < 0.10 and i1 < 0.10) or
                (i0 < self.stop_min_improve and i1 < self.stop_min_improve) or
                (i0 < 0 and i1 < 0))
        if stop: print(f'  [EARLY STOP]  improve={i0*100:.1f}%')
        return stop

    def _make_loss(self, meta):
        """Build the right loss object for the operator type."""
        s_mat = meta['s_input']    # (Nx*Ny, n_freqs)
        if isinstance(self.op, FFTChebOperator2D):
            return SpectralLossHybrid(self.op, meta['mask_2d'],
                                       meta['alphas'], s_mat, self.lambda_oob)
        elif isinstance(self.op, FFTOperator2D):
            return SpectralLossHybrid(self.op, meta['mask_2d'],
                                       meta['alphas'], s_mat, self.lambda_oob)
        else:
            raise ValueError(f"Unknown op type {type(self.op)}")

    def add_grade(self, n_epochs=1500):
        n = len(self.grades) + 1
        op_name = type(self.op).__name__
        print(f"\n{'═'*60}\n  GRADE {n}  [{self.op.Nx}×{self.op.Ny}]  {op_name}\n{'═'*60}")

        e_n = self._next_residual if self._next_residual is not None \
              else self.op.residual(self._u, self.f)
        self._next_residual = None

        target       = -self.op.inv_L(e_n)
        target_scale = float(np.linalg.norm(target)) + 1e-12
        target_norm  = (target / target_scale).astype(np.float32)
        rel_b = float(np.linalg.norm(self._u - self.u_star) /
                      (np.linalg.norm(self.u_star) + 1e-12))
        print(f'  Residual L2  : {np.linalg.norm(e_n):.4e}')
        print(f'  Rel L2 error : {rel_b:.4e}')

        lr = self.lr_adaptor.update(self.infos[-1]['losses'], n-1) \
             if n > 1 and self.infos else self.lr_adaptor.base_lr

        s_np, meta = build_spectral_input_hybrid(e_n, target_norm, self.op,
                                                  gd_floor=self.gd_floor)
        meta['target_scale'] = target_scale
        width, depth = self._config(meta['entropy'], meta['n_freqs'], rel_b)
        print(f'  |K_n|={meta["n_freqs"]}  H={meta["entropy"]:.3f}  '
              f'w={width}  d={depth}  lr={lr:.2e}')

        net       = GradeNet(meta['n_freqs'], width, depth).to(self.device)
        s_t       = torch.tensor(s_np, dtype=torch.float32).to(self.device)
        tgt_t     = torch.tensor(target_norm.ravel(),
                                  dtype=torch.float32).to(self.device)
        criterion = self._make_loss(meta).to(self.device)
        opt   = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=300, min_lr=lr * 5e-3)

        max_ep      = n_epochs * 2 if meta['n_freqs'] > 20 else n_epochs
        log_every   = 1000 if max_ep > 3000 else 500
        best_loss   = float('inf'); best_rse = float('inf'); best_ckpt = None
        plateau_cnt = 0
        losses = []; losses_in = []; losses_oob = []; rse_history = []

        net.train()
        for ep in range(max_ep):
            opt.zero_grad()
            g    = net(s_t).squeeze(1)
            loss, l_in, l_oob = criterion(g, tgt_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step(); sched.step(loss.item())

            lv = loss.item()
            losses.append(lv); losses_in.append(l_in); losses_oob.append(l_oob)
            if lv < best_loss * (1 - 5e-5): best_loss = lv; plateau_cnt = 0
            else: plateau_cnt += 1
            if plateau_cnt >= 600 and ep > max_ep // 4:
                print(f'  [plateau] ep{ep+1}  loss={lv:.4e}'); break

            with torch.no_grad():
                g_np = net(s_t).squeeze(1).cpu().numpy() * target_scale
            u_cand = self._u + g_np.reshape(self.op.Nx, self.op.Ny)
            rse_ep = float(np.linalg.norm(u_cand - self.u_star) /
                           (np.linalg.norm(self.u_star) + 1e-12))
            rse_history.append(rse_ep)
            if rse_ep < best_rse:
                best_rse  = rse_ep
                best_ckpt = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if (ep + 1) % log_every == 0:
                print(f'  ep{ep+1:5d}  total={lv:.4e}  RSE={rse_ep:.4e}  best={best_rse:.4e}')

        if best_ckpt:
            net.load_state_dict({k: v.to(self.device) for k, v in best_ckpt.items()})
            print(f'  [rollback] best RSE={best_rse:.4e}')

        net.eval()
        with torch.no_grad():
            g_np = net(s_t).squeeze(1).cpu().numpy() * target_scale
        g_2d   = g_np.reshape(self.op.Nx, self.op.Ny)
        u_cand = self._u + g_2d
        rel_a  = float(np.linalg.norm(u_cand - self.u_star) /
                       (np.linalg.norm(self.u_star) + 1e-12))

        if rel_a < rel_b:
            self._u = u_cand; accepted = True
            print(f'  Rel L2 after : {rel_a:.4e}  ↓{(rel_b-rel_a)/rel_b*100:.1f}%')
        else:
            accepted = False
            print(f'  Rel L2 after : {rel_a:.4e}  ↑ REJECTED')

        self._next_residual = self.op.residual(self._u, self.f)
        net.requires_grad_(False)

        info = {**meta, 'depth': depth, 'width': width,
                'losses': losses, 'losses_in': losses_in, 'losses_oob': losses_oob,
                'rse_history': rse_history, 'g_np': g_2d,
                'rel_err_before': rel_b, 'rel_err_after': rel_a,
                'lr_used': lr, 'accepted': accepted}
        self.grades.append(net); self.infos.append(info)
        return info

    def solve(self, n_grades=6, n_epochs=1500):
        t0 = time.perf_counter()
        for i in range(n_grades):
            self.add_grade(n_epochs)
            if self._should_stop(): print(f'  Stop after {i+1} grades.'); break
        print(f'\n  Total: {time.perf_counter()-t0:.1f}s')
        self._table()

    def _table(self):
        print(f"\n{'═'*80}")
        print(f"{'G':^3}  {'|Kn|':^5}  {'H':^6}  {'w':^4}  {'d':^3}  "
              f"{'err_before':^11}  {'err_after':^11}  {'Δ%':^8}  {'acc':^5}")
        print('─' * 80)
        for i, info in enumerate(self.infos):
            eb, ea = info['rel_err_before'], info['rel_err_after']
            d = (eb - ea) / (eb + 1e-30) * 100
            print(f"  {i+1}  {info['n_freqs']:5d}  {info['entropy']:6.3f}  "
                  f"{info['width']:4d}  {info['depth']:3d}  "
                  f"{eb:11.4e}  {ea:11.4e}  {d:+7.1f}%  "
                  f"{'✓' if info['accepted'] else '✗':^5}")


# SpectralLossFFT2D — for fully periodic FFTOperator2D


# SpectralLossFFT2D merged into SpectralLossHybrid
SpectralLossFFT2D = SpectralLossHybrid
