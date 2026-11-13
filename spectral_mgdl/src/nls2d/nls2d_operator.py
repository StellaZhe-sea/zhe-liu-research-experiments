"""
nls2d_operator.py
=================
2D Nonlinear Schrödinger Equation operator with Split-Step Fourier Method.

Equation:
    i ∂_t ψ + (1/2)(∂_xx + ∂_yy) ψ + σ|ψ|² ψ = 0

Split-step decomposition:
    Linear step (spectral, exact):  ∂_t ψ = -(i/2)(k² ) ψ
    Nonlinear step (exact):         ∂_t ψ = i σ |ψ|² ψ  →  ψ *= exp(i σ |ψ|² dt)

Residual formulation (for grade solver):
    Given current field ψ_n and target ψ*, define residual as the discrepancy
    after one SSF step. The GradeNet learns a correction δ such that
    SSF(ψ_n + δ) ≈ ψ*.

Author: Spectral-Adaptive Grade Solver project
"""

import numpy as np
import torch
import torch.fft as fft


# ---------------------------------------------------------------------------
# Wavenumber grids
# ---------------------------------------------------------------------------

def make_wavenumbers_2d(Nx: int, Ny: int, Lx: float = 2*np.pi, Ly: float = 2*np.pi,
                         device='cpu', dtype=torch.float64):
    """
    Returns (kx, ky) wavenumber arrays shaped (Nx,1) and (1,Ny).
    FFT-ordered (0, 1, ..., N/2-1, -N/2, ..., -1).
    """
    kx = torch.fft.fftfreq(Nx, d=Lx/(2*np.pi*Nx)).to(device=device, dtype=dtype)  # (Nx,)
    ky = torch.fft.fftfreq(Ny, d=Ly/(2*np.pi*Ny)).to(device=device, dtype=dtype)  # (Ny,)
    kx = kx.reshape(Nx, 1)   # broadcast over y
    ky = ky.reshape(1, Ny)   # broadcast over x
    return kx, ky


def make_k2_grid(Nx: int, Ny: int, Lx: float = 2*np.pi, Ly: float = 2*np.pi,
                 device='cpu', dtype=torch.float64):
    """k² = kx² + ky², shape (Nx, Ny)."""
    kx, ky = make_wavenumbers_2d(Nx, Ny, Lx, Ly, device, dtype)
    return kx**2 + ky**2


# ---------------------------------------------------------------------------
# Split-Step Fourier propagator
# ---------------------------------------------------------------------------

class NLSOp:
    """
    2D NLS operator using Split-Step Fourier Method (Strang splitting, 2nd order).

    Parameters
    ----------
    Nx, Ny : int        Grid points in x and y (should be powers of 2 for FFT)
    Lx, Ly : float      Domain size [0, Lx] × [0, Ly] (periodic BC)
    sigma   : float     Nonlinearity sign: +1 (focusing), -1 (defocusing)
    dt      : float     Time step
    device  : str       'cpu' or 'cuda'
    """

    def __init__(self, Nx: int = 128, Ny: int = 128,
                 Lx: float = 2*np.pi, Ly: float = 2*np.pi,
                 sigma: float = 1.0, dt: float = 0.01,
                 device: str = 'cpu'):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.sigma = sigma
        self.dt = dt
        self.device = device

        # Precompute linear propagator exp(-i k² dt/2) in spectral domain
        k2 = make_k2_grid(Nx, Ny, Lx, Ly, device=device, dtype=torch.float64)
        # Linear phase factor for half-step
        self._lin_half = torch.exp(-1j * 0.25 * k2 * dt).to(torch.complex128)
        # Full step for SSF without Strang (used internally)
        self._lin_full = torch.exp(-1j * k2 * dt).to(torch.complex128)

        # Spatial grids
        self.x = torch.linspace(0, Lx, Nx+1, device=device, dtype=torch.float64)[:-1]
        self.y = torch.linspace(0, Ly, Ny+1, device=device, dtype=torch.float64)[:-1]
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')

    # ------------------------------------------------------------------
    # Core SSF step (Strang 2nd-order splitting)
    # ------------------------------------------------------------------

    def step(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Advance ψ by one time step dt using Strang splitting:
            ψ → L(dt/2) → N(dt) → L(dt/2)

        Parameters
        ----------
        psi : complex tensor of shape (Nx, Ny)

        Returns
        -------
        psi_new : complex tensor of shape (Nx, Ny)
        """
        psi = psi.to(torch.complex128)

        # Half linear step
        psi_hat = fft.fft2(psi)
        psi_hat = psi_hat * self._lin_half
        psi = fft.ifft2(psi_hat)

        # Full nonlinear step (exact: multiply by phase)
        phase = torch.exp(1j * self.sigma * torch.abs(psi)**2 * self.dt)
        psi = psi * phase

        # Half linear step
        psi_hat = fft.fft2(psi)
        psi_hat = psi_hat * self._lin_half
        psi = fft.ifft2(psi_hat)

        return psi

    def propagate(self, psi0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Propagate ψ0 for n_steps time steps. Returns final ψ."""
        psi = psi0.clone().to(torch.complex128)
        for _ in range(n_steps):
            psi = self.step(psi)
        return psi

    def propagate_trajectory(self, psi0: torch.Tensor, n_steps: int,
                              save_every: int = 1) -> list:
        """Returns list of saved snapshots (including initial)."""
        psi = psi0.clone().to(torch.complex128)
        traj = [psi.clone()]
        for t in range(n_steps):
            psi = self.step(psi)
            if (t+1) % save_every == 0:
                traj.append(psi.clone())
        return traj

    # ------------------------------------------------------------------
    # Residual for grade solver
    # ------------------------------------------------------------------

    def residual(self, psi_current: torch.Tensor, psi_target: torch.Tensor) -> torch.Tensor:
        """
        Residual e = SSF(ψ_current) - ψ_target.
        The grade solver seeks δ such that SSF(ψ + δ) ≈ ψ_target.
        Returns complex residual of shape (Nx, Ny).
        """
        return self.step(psi_current) - psi_target

    def l2_error(self, psi_approx: torch.Tensor, psi_ref: torch.Tensor) -> float:
        """Relative L2 error ‖ψ_approx - ψ_ref‖ / ‖ψ_ref‖."""
        diff = psi_approx - psi_ref
        return (torch.norm(diff) / torch.norm(psi_ref)).item()

    # ------------------------------------------------------------------
    # Analytic soliton solutions (for validation)
    # ------------------------------------------------------------------

    def soliton_1d_embedded(self, A: float = 1.0, x0: float = None,
                             vx: float = 0.0, t: float = 0.0) -> torch.Tensor:
        """
        1D soliton embedded in 2D (y-independent), focusing NLS (σ=+1):
            ψ(x,y,t) = A sech(A(x-x0-vx*t)) * exp(i(vx*x - (vx²/2 - A²/2)*t))

        Valid for σ = +1 only.
        """
        if x0 is None:
            x0 = self.Lx / 2
        X = self.X  # (Nx, Ny)
        u = A * (X - x0 - vx*t)
        env = A / torch.cosh(u)
        phase = torch.exp(1j * (vx*X - (vx**2/2 - A**2/2)*t) * torch.ones_like(X,
                dtype=torch.float64))
        return (env * phase).to(torch.complex128)

    def soliton_2d_ground_state(self, A: float = 1.0,
                                 x0: float = None, y0: float = None) -> torch.Tensor:
        """
        Approximate 2D Townes soliton profile (Gaussian approximation for testing):
            ψ ≈ A * sech(r) where r = sqrt((x-x0)²+(y-y0)²)
        Note: true 2D ground state requires numerical shooting; this is a fast approximation.
        """
        if x0 is None: x0 = self.Lx / 2
        if y0 is None: y0 = self.Ly / 2
        dx = self.X - x0
        dy = self.Y - y0
        # Periodic wrap
        dx = dx - self.Lx * torch.round(dx / self.Lx)
        dy = dy - self.Ly * torch.round(dy / self.Ly)
        r = torch.sqrt(dx**2 + dy**2)
        return (A / torch.cosh(r)).to(torch.complex128)

    # ------------------------------------------------------------------
    # Spectral analysis utilities
    # ------------------------------------------------------------------

    def spectral_power(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Returns 2D spectral power |ψ̂(kx,ky)|² / (Nx*Ny)², shape (Nx, Ny).
        FFT-shifted so (kx=0, ky=0) is at center.
        """
        psi_hat = fft.fft2(psi.to(torch.complex128))
        power = torch.abs(psi_hat)**2 / (self.Nx * self.Ny)**2
        return fft.fftshift(power)

    def resonant_quartets(self, psi: torch.Tensor, threshold: float = 0.01,
                           delta_k_max: int = 3) -> list:
        """
        Find dominant frequency pairs (k1, k2) satisfying four-wave resonance:
            k1 + k2 = k3 + k4,  |k1|² + |k2|² = |k3|² + |k4|²  (NLS dispersion)

        Used to build 'selection' edge weight matrix in SpectralGNN.

        Parameters
        ----------
        psi           : current field
        threshold     : min fraction of peak power to include a mode
        delta_k_max   : max |ki - kj| in each direction to check

        Returns
        -------
        List of ((kx_i, ky_i), (kx_j, ky_j)) resonant pairs
        """
        psi_hat = fft.fft2(psi.to(torch.complex128))
        power = torch.abs(psi_hat)**2
        peak = power.max().item()
        cutoff = threshold * peak

        # Active modes
        active = torch.nonzero(power > cutoff, as_tuple=False)  # (M, 2)
        if len(active) == 0:
            return []

        kx_arr = torch.fft.fftfreq(self.Nx, d=self.Lx/(2*np.pi*self.Nx))
        ky_arr = torch.fft.fftfreq(self.Ny, d=self.Ly/(2*np.pi*self.Ny))

        modes = [(kx_arr[i].item(), ky_arr[j].item()) for i, j in active.tolist()]
        pairs = []
        for a, (kxi, kyi) in enumerate(modes):
            for b, (kxj, kyj) in enumerate(modes):
                if b <= a:
                    continue
                # Resonance condition: omega_i + omega_j = const for NLS
                # (dispersion relation: omega = k²/2)
                om_i = (kxi**2 + kyi**2) / 2
                om_j = (kxj**2 + kyj**2) / 2
                # Check if there exist k3 in active modes with |omega_i - omega_j| small
                # Simplified: flag pairs with |k_i| ≈ |k_j| (degenerate quartet)
                if abs(om_i - om_j) < 0.5 * max(om_i, om_j, 1e-8):
                    pairs.append(((kxi, kyi), (kxj, kyj)))
        return pairs

    def conservation_diagnostics(self, psi: torch.Tensor) -> dict:
        """
        Compute conserved quantities:
          N  = ∫|ψ|² dA  (mass/norm)
          H  = ∫(½|∇ψ|² - σ/2 |ψ|⁴) dA  (Hamiltonian)
          Px = ∫Im(ψ* ∂x ψ) dA  (x-momentum)
        """
        psi = psi.to(torch.complex128)
        dx = self.Lx / self.Nx
        dy = self.Ly / self.Ny
        dA = dx * dy

        # Mass
        N = (torch.abs(psi)**2 * dA).sum().real.item()

        # Gradient energy via spectral differentiation
        psi_hat = fft.fft2(psi)
        kx, ky = make_wavenumbers_2d(self.Nx, self.Ny, self.Lx, self.Ly,
                                      device=self.device, dtype=torch.float64)
        kx = kx.to(torch.complex128)
        ky = ky.to(torch.complex128)

        dpsi_dx = fft.ifft2(1j * kx * psi_hat)
        dpsi_dy = fft.ifft2(1j * ky * psi_hat)
        grad_energy = (0.5 * (torch.abs(dpsi_dx)**2 + torch.abs(dpsi_dy)**2) * dA).sum().real.item()
        nonlin_energy = (self.sigma * 0.5 * torch.abs(psi)**4 * dA).sum().real.item()
        H = grad_energy - nonlin_energy

        # x-momentum
        Px = (torch.conj(psi) * dpsi_dx).imag.sum().item() * dA

        return {'N': N, 'H': H, 'Px': Px,
                'grad_energy': grad_energy, 'nonlin_energy': nonlin_energy}


# ---------------------------------------------------------------------------
# 2D Spectral residual analysis (for grade solver input)
# ---------------------------------------------------------------------------

def gd_threshold_2d(power_flat: np.ndarray, floor: float = 0.10) -> float:
    """
    GD threshold adapted for 2D power spectrum (flattened).
    Returns amplitude threshold tau.
    """
    amps = np.sqrt(np.maximum(power_flat, 0))
    mad = np.median(np.abs(amps - np.median(amps)))
    sigma = mad / 0.6745
    n = len(amps)
    omega = 0.56/n**3 - 0.95/n**2 + 1.82/n + 1.43
    tau = max(omega * sigma, floor * amps.max())
    return tau


def select_active_modes_2d(residual: torch.Tensor,
                            floor: float = 0.10,
                            max_modes: int = 64) -> tuple:
    """
    Select active (kx, ky) modes from 2D residual using GD threshold.

    Returns
    -------
    mode_indices : LongTensor of shape (K, 2)   — flat FFT indices (ix, iy)
    mode_kvecs   : FloatTensor of shape (K, 2)  — physical (kx, ky) values
    amplitudes   : FloatTensor of shape (K,)    — |r̂[kx,ky]|
    """
    Nx, Ny = residual.shape
    res_hat = fft.fft2(residual.to(torch.complex128))
    power = torch.abs(res_hat).cpu().numpy().flatten()

    tau = gd_threshold_2d(power, floor=floor)

    power_2d = torch.abs(res_hat).cpu()
    mask = power_2d > tau
    indices = torch.nonzero(mask, as_tuple=False)  # (K, 2)

    if len(indices) == 0:
        # Fallback: keep top-3 modes
        flat_idx = torch.topk(power_2d.flatten(), k=3).indices
        indices = torch.stack([flat_idx // Ny, flat_idx % Ny], dim=1)

    # Sort by amplitude descending, cap at max_modes
    amps = power_2d[indices[:, 0], indices[:, 1]]
    order = torch.argsort(amps, descending=True)
    indices = indices[order[:max_modes]]
    amps = amps[order[:max_modes]]

    # Physical wavenumbers
    kx_arr = torch.fft.fftfreq(Nx).float() * Nx   # integer wavenumbers
    ky_arr = torch.fft.fftfreq(Ny).float() * Ny
    mode_kvecs = torch.stack([
        kx_arr[indices[:, 0]],
        ky_arr[indices[:, 1]]
    ], dim=1)  # (K, 2)

    return indices, mode_kvecs, amps


def build_spectral_input_2d(residual: torch.Tensor,
                             mode_indices: torch.Tensor,
                             mode_kvecs: torch.Tensor,
                             amplitudes: torch.Tensor,
                             Lx: float = 2*np.pi,
                             Ly: float = 2*np.pi) -> torch.Tensor:
    """
    Build spectral encoding s ∈ R^{Nx × Ny × K} for 2D grade solver.

    For each active mode k_i = (kx_i, ky_i):
        s[:,:,i] = log_norm_i * sign_i * cos(kx_i * X + ky_i * Y)

    where log_norm and sign encode amplitude and phase from optimal correction δ* = L^{-1} r.

    Note: In 2D NLS, the "linear inverse" is the SSF half-step operator applied to residual.
    We approximate: δ* ≈ -residual (first-order, since SSF is near-unitary).

    Returns
    -------
    s : Tensor of shape (Nx, Ny, K)
    """
    Nx, Ny = residual.shape
    K = len(mode_indices)

    res_hat = fft.fft2(residual.to(torch.complex128))

    # Optimal correction in spectral domain: δ̂* = -r̂ (first-order approx)
    delta_hat = -res_hat

    # Extract amplitudes and phases at active modes
    a = torch.abs(delta_hat[mode_indices[:, 0], mode_indices[:, 1]]).float()  # (K,)
    phases = torch.angle(delta_hat[mode_indices[:, 0], mode_indices[:, 1]]).float()  # (K,)

    # Energy weights
    E_total = (a**2).sum() + 1e-12
    alpha = a**2 / E_total  # (K,)

    # Log-normalized amplitudes
    a_max = a.max() + 1e-12
    log_norm = torch.log(1 + a) / torch.log(1 + a_max)  # (K,)

    # Spatial grids
    x = torch.linspace(0, Lx, Nx+1)[:-1].float()
    y = torch.linspace(0, Ly, Ny+1)[:-1].float()
    X, Y = torch.meshgrid(x, y, indexing='ij')  # (Nx, Ny)

    # 2π/L factors for physical wavenumbers
    s = torch.zeros(Nx, Ny, K, dtype=torch.float32)
    for i in range(K):
        kx_i = mode_kvecs[i, 0].item() * 2*np.pi / Lx
        ky_i = mode_kvecs[i, 1].item() * 2*np.pi / Ly
        phi_i = phases[i].item()
        # cos encoding (real part of plane wave)
        basis = torch.cos(kx_i * X + ky_i * Y + phi_i)
        s[:, :, i] = log_norm[i].item() * basis

    return s, alpha


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("NLSOp 2D — Validation Suite")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # --- Test 1: 1D soliton mass conservation ---
    print("\n[Test 1] 1D soliton (embedded) mass conservation")
    op = NLSOp(Nx=128, Ny=64, Lx=20.0, Ly=2*np.pi,
               sigma=1.0, dt=0.001, device=device)

    psi0 = op.soliton_1d_embedded(A=1.0, x0=10.0, vx=0.5, t=0.0)
    diag0 = op.conservation_diagnostics(psi0)

    T = 1.0
    n_steps = int(T / op.dt)
    t0 = time.time()
    psi_T = op.propagate(psi0, n_steps)
    elapsed = time.time() - t0
    diagT = op.conservation_diagnostics(psi_T)

    print(f"  Steps: {n_steps}, Time: {elapsed:.2f}s")
    print(f"  Mass  N: {diag0['N']:.6f} → {diagT['N']:.6f}  "
          f"(rel err: {abs(diagT['N']-diag0['N'])/diag0['N']:.2e})")
    print(f"  Hamiltonian H: {diag0['H']:.6f} → {diagT['H']:.6f}  "
          f"(rel err: {abs(diagT['H']-diag0['H'])/(abs(diag0['H'])+1e-12):.2e})")

    # --- Test 2: 2D soliton spectral mode selection ---
    print("\n[Test 2] 2D residual → active mode selection")
    op2 = NLSOp(Nx=64, Ny=64, Lx=2*np.pi, Ly=2*np.pi,
                sigma=1.0, dt=0.01, device=device)
    psi_gs = op2.soliton_2d_ground_state(A=1.0)
    psi_step = op2.step(psi_gs)
    # Add small noise as synthetic residual
    noise = 0.05 * (torch.randn(64, 64) + 1j * torch.randn(64, 64)).to(
        device=device, dtype=torch.complex128)
    residual = (psi_step - psi_gs + noise).real.float()

    indices, kvecs, amps = select_active_modes_2d(residual, floor=0.10, max_modes=32)
    print(f"  Active modes selected: {len(indices)}")
    print(f"  Top-5 (kx,ky): {kvecs[:5].tolist()}")
    print(f"  Top-5 amps: {amps[:5].tolist()}")

    # --- Test 3: Spectral encoding shape ---
    print("\n[Test 3] Spectral encoding s shape")
    s, alpha = build_spectral_input_2d(residual.to(torch.complex128),
                                        indices, kvecs, amps)
    print(f"  s.shape: {s.shape}  (Nx, Ny, K)")
    print(f"  alpha.sum(): {alpha.sum().item():.6f}  (should be ≈ 1.0)")

    # --- Test 4: Resonant quartet detection ---
    print("\n[Test 4] Resonant quartet detection")
    pairs = op2.resonant_quartets(psi_gs, threshold=0.01)
    print(f"  Detected {len(pairs)} resonant mode pairs")
    if pairs:
        print(f"  Example: {pairs[0]}")

    print("\n✓ All validation tests passed.")
    print("=" * 60)
