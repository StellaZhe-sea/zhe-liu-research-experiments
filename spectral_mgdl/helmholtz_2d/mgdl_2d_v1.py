"""
MGDL 2D v1 — Modular 2D extension of the 1D Helmholtz MGDL framework.

Supported equations (via operator plug-in):
  - 2D Helmholtz:  (−Δ + κ²)u = f,  Dirichlet BC  → DST-I ⊗ DST-I
  - 2D NLS (next): periodic BC                      → FFT  ⊗ FFT
  - 2D N-S  (v5):  uses existing NSOperator

Key design decisions vs 1D v9:
  - SpectralBasis2D: abstract class, concrete subclasses per equation type
  - build_spectral_input_2d: returns s (Nx*Ny, |K_n|) — GradeNet unchanged
  - SpectralLoss2D: 2D transform, same in-band / OOB structure
  - GradeNet: UNCHANGED (takes flat spatial dim, arbitrary |K_n|)
  - GradeSolver2D: mirrors HelmholtzGradeSolver, equation-agnostic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import dstn, idstn
from abc import ABC, abstractmethod
import time

# ═══════════════════════════════════════════════════════════════
# 1. Abstract 2D Operator interface
# ═══════════════════════════════════════════════════════════════

class Operator2D(ABC):
    """
    Interface every 2D equation must implement.
    All arrays are numpy float32, shape (Nx, Ny).
    """
    @property
    @abstractmethod
    def Nx(self) -> int: ...

    @property
    @abstractmethod
    def Ny(self) -> int: ...

    @abstractmethod
    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """e = L[u] - f"""

    @abstractmethod
    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        """δ = -L^{-1}(rhs)  (spectral exact)"""

    @abstractmethod
    def transform(self, u: np.ndarray) -> np.ndarray:
        """Forward spectral transform  u → û  (real output for DST/DCT)"""

    @abstractmethod
    def itransform(self, u_hat: np.ndarray) -> np.ndarray:
        """Inverse spectral transform"""

    @property
    @abstractmethod
    def wavenumbers(self):
        """Returns (kx_grid, ky_grid) or equivalent frequency indexing."""


# ═══════════════════════════════════════════════════════════════
# 2. 2D Helmholtz operator  (−Δ + κ²)u = f, Dirichlet BC
#    Uses DST-I ⊗ DST-I  (exact for homogeneous Dirichlet)
# ═══════════════════════════════════════════════════════════════

class Helmholtz2DOperator(Operator2D):
    """
    (−∂²/∂x² − ∂²/∂y² + κ²) u = f
    on [0,1]² with u=0 on ∂Ω.

    Grid: interior points only,  x_i = i/(Nx+1), i=1..Nx
                                  y_j = j/(Ny+1), j=1..Ny

    Spectral basis: φ_{mn}(x,y) = sin(mπx) sin(nπy)
    Eigenvalue:     λ_{mn} = (mπ(Nx+1))² + (nπ(Ny+1))² + κ²
                  (scaled to unit domain after DST normalisation)

    DST-I eigenvalues (scipy convention, N interior pts):
        λ_k^x = (2(Nx+1))² sin²(kπ / 2(Nx+1))  ≈ (kπ/(Nx+1))²  for small k
    We use exact eigenvalues for spectral accuracy.
    """

    def __init__(self, Nx: int, Ny: int, kappa: float = 1.0):
        self._Nx = Nx
        self._Ny = Ny
        self.kappa = kappa
        # Exact DST-I eigenvalues on [0,1]
        # scipy DST-1 of length N acts on interior pts, eigenvalues:
        #   μ_k = 2(N+1) sin(kπ / 2(N+1))   → actual Laplacian eig = μ_k²
        kx = np.arange(1, Nx + 1, dtype=np.float64)
        ky = np.arange(1, Ny + 1, dtype=np.float64)
        # Physical Laplacian eigenvalues (using grid spacing h = 1/(N+1))
        lx = (kx * np.pi) ** 2          # (kπ)² — exact for continuous problem
        ly = (ky * np.pi) ** 2
        # 2D eigenvalue grid: λ_{mn} = lx_m + ly_n + κ²
        self._eigvals = lx[:, None] + ly[None, :] + kappa ** 2   # (Nx, Ny)

    @property
    def Nx(self): return self._Nx

    @property
    def Ny(self): return self._Ny

    @property
    def wavenumbers(self):
        kx = np.arange(1, self._Nx + 1)
        ky = np.arange(1, self._Ny + 1)
        return np.meshgrid(kx, ky, indexing='ij')   # each (Nx, Ny)

    def transform(self, u: np.ndarray) -> np.ndarray:
        """DST-I in both dimensions.  u: (Nx, Ny) → û: (Nx, Ny)  (unnormalised)."""
        return dstn(u.astype(np.float64), type=1, axes=(0, 1)).astype(np.float32)

    def itransform(self, u_hat: np.ndarray) -> np.ndarray:
        """Inverse DST-I: u = DST(û) / (4(Nx+1)(Ny+1))."""
        norm = 4.0 * (self._Nx + 1) * (self._Ny + 1)
        return (dstn(u_hat.astype(np.float64), type=1, axes=(0, 1)) / norm).astype(np.float32)

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        """e = (−Δ + κ²)u − f,  computed spectrally (no finite diff).
        DST convention: û = DST(u)  (unnorm); L̂u = eigvals * û; Lu = IDST(L̂u).
        """
        u_hat  = self.transform(u)                       # û  (unnorm)
        Lu_hat = self._eigvals * u_hat                   # (kπ²+κ²)û
        Lu     = self.itransform(Lu_hat)                 # physical space
        return (Lu - f).astype(np.float32)

    def inv_L(self, rhs: np.ndarray) -> np.ndarray:
        """L^{-1}(rhs).  rhs_hat / eigvals; result = IDST(result_hat).
        Note: target δ_n = −inv_L(e_n) is computed by the caller."""
        rhs_hat    = self.transform(rhs)                 # (unnorm)
        delta_hat  = rhs_hat / self._eigvals             # spectral inversion (+)
        return self.itransform(delta_hat).astype(np.float32)

    def make_solution(self, seed: int = 42) -> tuple:
        """
        Manufacture u_star as smooth sum-of-modes, compute f = L[u_star].
        Returns (u_star, f), both (Nx, Ny).
        """
        rng = np.random.default_rng(seed)
        kx_g, ky_g = self.wavenumbers
        # Random low-frequency coefficients
        K = min(8, self._Nx // 4, self._Ny // 4)
        mask = (kx_g <= K) & (ky_g <= K)
        coeffs = rng.standard_normal((self._Nx, self._Ny)).astype(np.float64) * mask
        # u_star in physical space via iDST
        norm = 4 * (self._Nx + 1) * (self._Ny + 1)
        u_star = self.itransform(coeffs * norm).astype(np.float32)
        # Dirichlet BC: interior only, borders stay 0 (already so by DST)
        f = self.residual(u_star, np.zeros_like(u_star)) + u_star * 0  # L[u_star]
        u_hat_star = self.transform(u_star)
        Lu_hat = self._eigvals * u_hat_star / norm
        f = self.itransform(Lu_hat * norm).astype(np.float32)
        return u_star, f


# ═══════════════════════════════════════════════════════════════
# 3. 2D Spectral Input Builder
#    Mirrors build_spectral_input from 1D v9 exactly,
#    but operates on 2D (Nx, Ny) arrays.
#    Output s: (Nx*Ny, |K_n|)  — GradeNet sees flat spatial dim.
# ═══════════════════════════════════════════════════════════════

def gd_threshold_2d(amps_flat: np.ndarray, floor_ratio: float = 0.01,
                    max_ratio: float = 0.25) -> float:
    """GD threshold on flattened 2D coefficient array."""
    a = amps_flat
    if a.max() == 0:
        return 0.0
    tau = float(a.max()) * floor_ratio
    for _ in range(100):
        if (a > tau).sum() <= max(1, max_ratio * a.size):
            break
        tau *= 1.3
    return float(tau)


def build_spectral_input_2d(residual_e: np.ndarray,
                             target_delta: np.ndarray,
                             op: Operator2D,
                             gd_floor: float = 0.01):
    """
    2D analog of build_spectral_input from 1D v9.

    Parameters
    ----------
    residual_e   : (Nx, Ny) physical-space residual e_n
    target_delta : (Nx, Ny) physical-space target δ_n  (already normalised)
    op           : Operator2D instance
    gd_floor     : GD threshold floor ratio

    Returns
    -------
    s    : (Nx*Ny, |K_n|)  spectral input matrix
    meta : dict with keys matching 1D convention where possible
    """
    Nx, Ny = op.Nx, op.Ny

    # ── Spectral transforms ──────────────────────────────────────
    e_hat   = op.transform(residual_e)          # (Nx, Ny)
    tgt_hat = op.transform(target_delta)         # (Nx, Ny)

    # ── GD sparse selection on |e_hat| ──────────────────────────
    amps_res = np.abs(e_hat).astype(np.float32)
    tau      = gd_threshold_2d(amps_res.ravel(), floor_ratio=gd_floor)
    mask_2d  = amps_res > tau                    # (Nx, Ny) bool

    # ensure at least 1 active frequency
    if not mask_2d.any():
        mask_2d.ravel()[np.argmax(amps_res.ravel())] = True

    # active wavenumber indices: list of (m, n) pairs
    active_idx = np.argwhere(mask_2d)            # (|K_n|, 2)
    n_freqs    = len(active_idx)

    # ── δ_n info on active set ──────────────────────────────────
    amps_delta = np.abs(tgt_hat).astype(np.float32)
    a_sel      = amps_delta[mask_2d]             # (|K_n|,)
    signs      = np.sign(tgt_hat[mask_2d]).astype(np.float32)

    energy  = a_sel ** 2
    alphas  = energy / (energy.sum() + 1e-12)
    H       = float(-np.sum(alphas * np.log(alphas + 1e-12)))

    log_amps = np.log(1.0 + a_sel)
    log_max  = np.log(1.0 + a_sel.max() + 1e-12)
    log_norm = (log_amps / log_max).astype(np.float32)

    # ── Basis matrix s: (Nx*Ny, |K_n|) ─────────────────────────
    # Physical grid (interior points)
    ix = np.arange(1, Nx + 1, dtype=np.float32)   # (Nx,)
    iy = np.arange(1, Ny + 1, dtype=np.float32)   # (Ny,)

    # For DST-I (Dirichlet): φ_{mn}(i,j) = sin(mπi/(Nx+1)) sin(nπj/(Ny+1))
    # active_idx[:,0] = m indices (1-based wavenumber)
    # active_idx[:,1] = n indices (1-based wavenumber)
    m = active_idx[:, 0] + 1   # (|K_n|,)  1-based
    n = active_idx[:, 1] + 1   # (|K_n|,)  1-based

    phi_x = np.sin(np.pi * ix[:, None] * m[None, :] / (Nx + 1))   # (Nx, |K_n|)
    phi_y = np.sin(np.pi * iy[:, None] * n[None, :] / (Ny + 1))   # (Ny, |K_n|)

    # outer product → (Nx, Ny, |K_n|), then flatten spatial dims
    basis = phi_x[:, None, :] * phi_y[None, :, :]      # (Nx, Ny, |K_n|)
    basis = basis.reshape(Nx * Ny, n_freqs)              # (Nx*Ny, |K_n|)

    # Encode with log-norm amplitudes and signs of δ_n
    s = (log_norm[None, :] * signs[None, :] * basis).astype(np.float32)

    meta = dict(
        active_idx=active_idx,      # (|K_n|, 2) wavenumber pairs
        mask_2d=mask_2d,            # (Nx, Ny) bool
        tau=tau,
        n_freqs=n_freqs,
        entropy=H,
        alphas=alphas,
        signs=signs,
        log_norm=log_norm,
        amps_res=amps_res,
        amps_delta=amps_delta,
        e_hat=e_hat,
        tgt_hat=tgt_hat,
        target=target_delta,
        s_input=s,
    )
    return s, meta


# ═══════════════════════════════════════════════════════════════
# 4. SpectralLoss2D
#    Same structure as 1D SpectralLoss:
#    loss = Σ_{K_n} α_k (ĝ_k - δ̂_k)² + λ_oob · mean(ĝ_k² for k ∉ K_n)
#    g and target are in PHYSICAL space (Nx*Ny,).
# ═══════════════════════════════════════════════════════════════

class SpectralLoss2D(nn.Module):
    """
    2D spectral loss using differentiable DST-I via matrix multiply.
    DST-I is separable: û(m,n) = (Mx @ u_2d @ My.T)[m,n]
    This keeps the computation graph intact for backprop.

    loss = Σ_{K_n} α_k (ĝ_k - δ̂_k)² + λ_oob · mean(ĝ_k² for k ∉ K_n)
    """
    def __init__(self, op: Operator2D, mask_2d: np.ndarray,
                 alphas: np.ndarray, lambda_oob: float = 0.0):
        super().__init__()
        self.Nx = op.Nx; self.Ny = op.Ny
        self.lambda_oob = lambda_oob
        norm = 1.0 / (4 * (op.Nx + 1) * (op.Ny + 1))

        # Precompute DST-I matrices (fixed, registered as buffers)
        kx = torch.arange(1, op.Nx+1, dtype=torch.float32)
        jx = torch.arange(1, op.Nx+1, dtype=torch.float32)
        Mx = torch.sin(torch.pi * kx[:, None] * jx[None, :] / (op.Nx+1)) * norm  # (Nx,Nx)

        ky = torch.arange(1, op.Ny+1, dtype=torch.float32)
        jy = torch.arange(1, op.Ny+1, dtype=torch.float32)
        My = torch.sin(torch.pi * ky[:, None] * jy[None, :] / (op.Ny+1))          # (Ny,Ny)

        self.register_buffer('Mx', Mx)
        self.register_buffer('My', My)

        mask_flat = torch.tensor(mask_2d.ravel(), dtype=torch.bool)
        self.register_buffer('mask_in',  mask_flat)
        self.register_buffer('mask_out', ~mask_flat)
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))

        # Precompute target hat (no grad needed)
        self._tgt_hat_flat = None

    def dst2d_flat(self, x_flat):
        """x_flat: (Nx*Ny,) → û_flat: (Nx*Ny,)  differentiable."""
        x2d   = x_flat.reshape(self.Nx, self.Ny)
        u_hat = self.Mx @ x2d @ self.My.T        # (Nx, Ny)
        return u_hat.reshape(-1)                  # (Nx*Ny,)

    def forward(self, g_flat, target_flat):
        """g_flat, target_flat: (Nx*Ny,). Returns: total, l_in, l_oob."""
        g_hat   = self.dst2d_flat(g_flat)
        tgt_hat = self.dst2d_flat(target_flat.detach())  # target has no grad

        diff2   = (g_hat - tgt_hat) ** 2
        loss_in = (self.alphas * diff2[self.mask_in]).sum()
        if self.lambda_oob > 0 and self.mask_out.any():
            loss_oob = self.lambda_oob * (g_hat[self.mask_out] ** 2).mean()
        else:
            loss_oob = g_flat.new_zeros(1).squeeze()
        total = loss_in + loss_oob
        return total, loss_in.item(), loss_oob.item()


# ═══════════════════════════════════════════════════════════════
# 5. GradeNet — UNCHANGED from 1D v9
#    Input: s (Nx*Ny, |K_n|)  Output: g (Nx*Ny,)
# ═══════════════════════════════════════════════════════════════

class GradeNet(nn.Module):
    def __init__(self, idim, width, depth):
        super().__init__()
        self.idim = idim
        self._W_raw = nn.Parameter(torch.zeros(idim, idim))
        self.mix_act = nn.Tanh()
        self.proj = nn.Linear(idim, width)
        self.mlp  = nn.Sequential(*[
            layer
            for _ in range(depth)
            for layer in (nn.Linear(width, width), nn.Tanh())
        ])
        self.out = nn.Linear(width, 1)
        nn.init.normal_(self._W_raw, mean=0.0, std=0.01)
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)
        for m in list(self.mlp.modules()) + [self.out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    @property
    def W_sym(self):
        return 0.5 * (self._W_raw + self._W_raw.t())

    def forward(self, s):
        h_mix = self.mix_act(s @ self.W_sym.t()) + s
        return self.out(self.mlp(self.proj(h_mix)))


# ═══════════════════════════════════════════════════════════════
# 6. LrAdaptor — identical to 1D v9
# ═══════════════════════════════════════════════════════════════

class LrAdaptor:
    def __init__(self, base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                 tail_frac=0.20, slope_eps=-1e-10,
                 decay_bad=0.50, decay_mild=0.70, boost_good=0.70):
        self.base_lr = base_lr; self.lr_min = lr_min; self.lr_max = lr_max
        self.tail_frac = tail_frac; self.slope_eps = slope_eps
        self.decay_bad = decay_bad; self.decay_mild = decay_mild
        self.boost_good = boost_good
        self._current_lr = base_lr

    def update(self, losses, grade_n):
        losses = np.array(losses, dtype=np.float64)
        n = len(losses)
        tail_start = max(0, int(n * (1 - self.tail_frac)))
        tail = losses[tail_start:]; head = losses[:max(1, tail_start)]
        x = np.arange(len(tail), dtype=np.float64)
        slope = np.polyfit(x, tail, 1)[0] / (tail[0] + 1e-30)
        drop  = (head.mean() - tail.mean()) / (head.mean() + 1e-30)
        lr = self._current_lr
        if slope < self.slope_eps:
            reason = f'still descending slope={slope:.2e}'
        elif drop < 0.05:
            lr = min(lr * 1.5, self.lr_max); reason = f'no progress x1.5'
        elif drop > 0.90:
            lr *= self.decay_mild; reason = f'great x{self.decay_mild}'
        else:
            reason = f'moderate hold'
        lr = float(np.clip(lr, self.lr_min, self.lr_max))
        self._current_lr = lr
        print(f'  [LrAdaptor G{grade_n}] {reason}  lr={lr:.2e}')
        return lr

    @property
    def current_lr(self): return self._current_lr


# ═══════════════════════════════════════════════════════════════
# 7. GradeSolver2D  — mirrors HelmholtzGradeSolver exactly
# ═══════════════════════════════════════════════════════════════

class GradeSolver2D:
    """
    Equation-agnostic multi-grade solver for 2D problems.
    Plug in any Operator2D subclass.

    u shape throughout: (Nx, Ny)  numpy float32.
    GradeNet operates on flattened (Nx*Ny,) vectors.
    """

    def __init__(self, op: Operator2D, f: np.ndarray, u_star: np.ndarray,
                 base_width=64, base_depth=3, lambda_oob=0.0,
                 gd_floor=0.01, stop_min_improve=0.20,
                 base_lr=1e-3, lr_min=1e-4, lr_max=5e-3,
                 device='cpu'):
        self.op            = op
        self.f             = f.astype(np.float32)
        self.u_star        = u_star.astype(np.float32)
        self.base_width    = base_width
        self.base_depth    = base_depth
        self.lambda_oob    = lambda_oob
        self.gd_floor      = gd_floor
        self.stop_min_improve = stop_min_improve
        self.device        = torch.device(device)
        self.grades        = []
        self.infos         = []
        self._u            = np.zeros((op.Nx, op.Ny), np.float32)
        self._next_residual = None
        self.lr_adaptor    = LrAdaptor(base_lr=base_lr, lr_min=lr_min, lr_max=lr_max)

    def _config(self, H, nk, rel_err):
        depth = max(2, min(self.base_depth + int(H * 2.0), 10))
        if rel_err < 1e-3:   width_cap = 256
        elif rel_err < 1e-2: width_cap = 512
        else:                width_cap = 1024
        width = min(max(self.base_width, 4 * nk), width_cap)
        return width, depth

    def _should_stop(self):
        if len(self.infos) < 2:
            return False
        eb = self.infos[-1]['rel_err_before']; ea = self.infos[-1]['rel_err_after']
        improve  = (eb - ea) / (eb + 1e-30)
        peb = self.infos[-2]['rel_err_before']; pea = self.infos[-2]['rel_err_after']
        prev_impr = (peb - pea) / (peb + 1e-30)
        cond_A = (improve < 0.10) and (prev_impr < 0.10)
        cond_B = (improve < self.stop_min_improve) and (prev_impr < self.stop_min_improve)
        cond_C = (improve < 0) and (prev_impr < 0)
        stop = cond_A or cond_B or cond_C
        if stop:
            reasons = []
            if cond_A: reasons.append(f'stagnant {improve*100:.1f}% & {prev_impr*100:.1f}%')
            if cond_B: reasons.append(f'low improve {improve*100:.1f}%')
            if cond_C: reasons.append('consecutive divergence')
            print(f'  [EARLY STOP 2D]  ' + '  |  '.join(reasons))
        return stop

    def add_grade(self, n_epochs=1500):
        n = len(self.grades) + 1
        print(f"\n{'═'*60}\n  GRADE {n}  [{self.op.Nx}×{self.op.Ny}]\n{'═'*60}")

        # Residual & target
        if self._next_residual is not None:
            e_n = self._next_residual; self._next_residual = None
        else:
            e_n = self.op.residual(self._u, self.f)

        target       = -self.op.inv_L(e_n)
        target_scale = float(np.linalg.norm(target)) + 1e-12
        target_norm  = (target / target_scale).astype(np.float32)

        rel_b = float(np.linalg.norm(self._u - self.u_star) /
                      (np.linalg.norm(self.u_star) + 1e-12))
        print(f'  Residual L2  : {np.linalg.norm(e_n):.4e}')
        print(f'  Rel L2 error : {rel_b:.4e}')

        # Adaptive lr
        if n > 1 and self.infos:
            lr = self.lr_adaptor.update(self.infos[-1]['losses'], grade_n=n-1)
        else:
            lr = self.lr_adaptor.base_lr

        # Spectral input  s: (Nx*Ny, |K_n|)
        s_np, meta = build_spectral_input_2d(e_n, target_norm, self.op,
                                              gd_floor=self.gd_floor)
        meta['target_scale'] = target_scale
        width, depth = self._config(meta['entropy'], meta['n_freqs'], rel_b)
        print(f'  |K_n|={meta["n_freqs"]}  H={meta["entropy"]:.3f}  '
              f'w={width}  d={depth}  lr={lr:.2e}')

        # Network & loss
        Nflat = self.op.Nx * self.op.Ny
        net   = GradeNet(meta['n_freqs'], width, depth).to(self.device)
        s_t   = torch.tensor(s_np,        dtype=torch.float32).to(self.device)
        tgt_t = torch.tensor(target_norm.ravel(), dtype=torch.float32).to(self.device)
        criterion = SpectralLoss2D(
            op=self.op, mask_2d=meta['mask_2d'],
            alphas=meta['alphas'], lambda_oob=self.lambda_oob
        ).to(self.device)
        opt  = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=300,
            min_lr=lr * 5e-3)

        max_ep  = n_epochs * 2 if meta['n_freqs'] > 20 else n_epochs
        log_every = 1000 if max_ep > 3000 else 500
        best_loss = float('inf'); best_rse = float('inf'); best_ckpt = None
        plateau_cnt = 0
        losses = []; losses_in = []; losses_oob = []; rse_history = []
        t0 = time.perf_counter()

        net.train()
        for ep in range(max_ep):
            opt.zero_grad()
            g = net(s_t).squeeze(1)            # (Nx*Ny,)
            loss, l_in, l_oob = criterion(g, tgt_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            sched.step(loss.item())

            lv = loss.item()
            losses.append(lv); losses_in.append(l_in); losses_oob.append(l_oob)

            if lv < best_loss * (1 - 5e-5):
                best_loss = lv; plateau_cnt = 0
            else:
                plateau_cnt += 1
            if plateau_cnt >= 600 and ep > max_ep // 4:
                print(f'  [plateau] ep{ep+1}  loss={lv:.4e}'); break

            # RSE checkpoint
            with torch.no_grad():
                g_ep = net(s_t).squeeze(1).cpu().numpy() * target_scale
            u_cand = self._u + g_ep.reshape(self.op.Nx, self.op.Ny)
            rse_ep = float(np.linalg.norm(u_cand - self.u_star) /
                           (np.linalg.norm(self.u_star) + 1e-12))
            rse_history.append(rse_ep)
            if rse_ep < best_rse:
                best_rse  = rse_ep
                best_ckpt = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if (ep + 1) % log_every == 0:
                print(f'  ep{ep+1:5d}  total={lv:.4e}  in={l_in:.4e}  '
                      f'RSE={rse_ep:.4e}  bestRSE={best_rse:.4e}')

        # Rollback to best-RSE checkpoint
        if best_ckpt is not None:
            net.load_state_dict({k: v.to(self.device) for k, v in best_ckpt.items()})
            print(f'  [rollback] best RSE={best_rse:.4e}')

        u_before = self._u.copy()
        net.eval()
        with torch.no_grad():
            g_np = net(s_t).squeeze(1).cpu().numpy() * target_scale
        g_2d      = g_np.reshape(self.op.Nx, self.op.Ny)
        u_cand    = self._u + g_2d
        rel_a = float(np.linalg.norm(u_cand - self.u_star) /
                      (np.linalg.norm(self.u_star) + 1e-12))

        if rel_a < rel_b:
            self._u = u_cand; accepted = True
            print(f'  Rel L2 after : {rel_a:.4e}  ↓{abs(rel_a-rel_b)/rel_b*100:.1f}%')
        else:
            accepted = False
            print(f'  Rel L2 after : {rel_a:.4e}  ↑ REJECTED — u unchanged')

        # e_{n+1}
        e_next = self.op.residual(self._u, self.f)
        self._next_residual = e_next

        net.requires_grad_(False)

        info = {
            **meta,
            'depth': depth, 'width': width,
            'losses': losses, 'losses_in': losses_in, 'losses_oob': losses_oob,
            'rse_history': rse_history,
            'g_np': g_2d,
            'rel_err_before': rel_b, 'rel_err_after': rel_a,
            'lr_used': lr, 'accepted': accepted,
        }
        self.grades.append(net)
        self.infos.append(info)
        return info

    def solve(self, n_grades=6, n_epochs=1500):
        t0 = time.perf_counter()
        for i in range(n_grades):
            self.add_grade(n_epochs)
            if self._should_stop():
                print(f'  Stopping after {i+1} grades.'); break
        print(f'\n  Total time: {time.perf_counter()-t0:.1f}s')
        self._table()

    def _table(self):
        print(f"\n{'═'*80}")
        print(f"{'G':^3}  {'|Kn|':^5}  {'H':^6}  {'w':^4}  {'d':^3}  "
              f"{'err_before':^11}  {'err_after':^11}  {'Δ%':^8}  {'acc':^5}")
        print('─' * 80)
        for i, info in enumerate(self.infos):
            eb, ea = info['rel_err_before'], info['rel_err_after']
            delta  = (eb - ea) / (eb + 1e-30) * 100
            print(f"  {i+1:1d}  {info['n_freqs']:5d}  {info['entropy']:6.3f}  "
                  f"{info['width']:4d}  {info['depth']:3d}  "
                  f"{eb:11.4e}  {ea:11.4e}  {delta:+7.1f}%  "
                  f"{'✓' if info['accepted'] else '✗':^5}")
        print('═' * 80)
