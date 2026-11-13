"""
Kolmogorov Flow Data Generator
================================
2D incompressible Navier-Stokes in vorticity-stream-function form:

    ∂ω/∂t + u·∇ω = ν∇²ω + f
    −∇²ψ = ω
    u = (∂ψ/∂y, −∂ψ/∂x)

Forcing: f = sin(k_f · y)   (Kolmogorov forcing, standard benchmark)
Domain : [0, 2π]² periodic
Solver : pseudo-spectral (FFT) + RK4 time integration

Usage:
    gen = KolmogorovFlowGenerator(N=64, Re=1000, T=20.0, dt=1e-3, k_f=4)
    omega_traj = gen.generate(n_samples=100, t_start=10.0, t_gap=0.5)
    # omega_traj : (100, N, N)  snapshots after statistical stationarity
"""

import numpy as np
import torch
from typing import Optional


class KolmogorovFlowGenerator:
    """
    Pseudo-spectral Kolmogorov Flow solver.

    Args:
        N    : grid resolution (N×N)
        Re   : Reynolds number ν = 1/Re
        T    : total simulation time
        dt   : time step
        k_f  : forcing wavenumber (sin(k_f * y))
        device: 'cpu' or 'cuda'
    """

    def __init__(self, N: int = 64, Re: float = 1000.0,
                 T: float = 20.0, dt: float = 1e-3,
                 k_f: int = 4, device: str = 'cpu'):
        self.N  = N
        self.Re = Re
        self.nu = 1.0 / Re
        self.T  = T
        self.dt = dt
        self.k_f = k_f
        self.device = device

        # Wavenumber grids
        k = torch.fft.fftfreq(N, d=1.0/N).to(device)   # integers 0..N/2-1, -N/2..-1
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        self.kx = kx
        self.ky = ky
        self.k2 = kx**2 + ky**2                         # |k|²  (N, N)
        # Avoid division by zero at k=0
        self.k2_inv = torch.where(self.k2 == 0,
                                  torch.zeros_like(self.k2),
                                  1.0 / (self.k2 + 1e-12 * (self.k2 == 0).float()))

        # Forcing in spectral space: f = sin(k_f * y)
        # In spectral: F̂[kx, ky] = −i/2 δ(kx=0)(δ(ky=k_f) − δ(ky=−k_f))
        # but simpler: compute in physical space and fft
        x = torch.linspace(0, 2*np.pi, N+1, device=device)[:-1]
        _, Y = torch.meshgrid(x, x, indexing='ij')
        self.forcing_phys = torch.sin(k_f * Y)          # (N, N)
        self.forcing_hat  = torch.fft.fft2(self.forcing_phys)

        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias = ((kx.abs() <= kmax) & (ky.abs() <= kmax)).float()

    # ── Core spectral operators ──────────────────────────────────────────

    def _laplacian_inv(self, omega_hat):
        """ψ̂ = −ω̂ / |k|²  (stream function from vorticity)"""
        return -omega_hat * self.k2_inv

    def _curl(self, psi_hat):
        """u = (∂ψ/∂y, −∂ψ/∂x) in spectral → physical"""
        # ∂ψ/∂y ↔ i·ky·ψ̂,  ∂ψ/∂x ↔ i·kx·ψ̂
        ux_hat = 1j * self.ky * psi_hat
        uy_hat = -1j * self.kx * psi_hat
        ux = torch.fft.ifft2(ux_hat).real
        uy = torch.fft.ifft2(uy_hat).real
        return ux, uy

    def _advection(self, omega_hat):
        """
        Compute u·∇ω in physical space, return spectral.
        Uses 2/3 dealiasing.
        """
        psi_hat = self._laplacian_inv(omega_hat)
        ux, uy  = self._curl(psi_hat)
        omega   = torch.fft.ifft2(omega_hat).real

        # Gradients of ω
        domega_dx = torch.fft.ifft2(1j * self.kx * omega_hat).real
        domega_dy = torch.fft.ifft2(1j * self.ky * omega_hat).real

        adv_phys = ux * domega_dx + uy * domega_dy
        adv_hat  = torch.fft.fft2(adv_phys) * self.dealias
        return adv_hat

    def _rhs(self, omega_hat):
        """
        RHS of vorticity equation:
          dω̂/dt = −(u·∇ω)^ + ν·(−|k|²)·ω̂ + F̂
        """
        adv  = self._advection(omega_hat)
        diff = -self.nu * self.k2 * omega_hat
        return -adv + diff + self.forcing_hat

    def _rk4_step(self, omega_hat, dt):
        k1 = self._rhs(omega_hat)
        k2 = self._rhs(omega_hat + 0.5*dt*k1)
        k3 = self._rhs(omega_hat + 0.5*dt*k2)
        k4 = self._rhs(omega_hat + dt*k3)
        return omega_hat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, omega0: Optional[torch.Tensor] = None,
            n_steps: Optional[int] = None,
            verbose: bool = False) -> torch.Tensor:
        """
        Run the solver from omega0 for n_steps steps.
        Returns final omega (N, N).
        """
        if omega0 is None:
            # Random initial condition (small amplitude)
            rng = torch.randn(self.N, self.N, device=self.device) * 0.1
            omega0 = rng
        if n_steps is None:
            n_steps = int(self.T / self.dt)

        omega_hat = torch.fft.fft2(omega0.to(self.device))
        for i in range(n_steps):
            omega_hat = self._rk4_step(omega_hat, self.dt)
            if verbose and (i % 1000 == 0):
                omega = torch.fft.ifft2(omega_hat).real
                print(f"  step {i}/{n_steps}, |ω|∞ = {omega.abs().max():.4f}")
        return torch.fft.ifft2(omega_hat).real

    def generate(self, n_samples: int, t_start: float = 10.0,
                 t_gap: float = 0.5, seed: int = 42,
                 verbose: bool = True) -> torch.Tensor:
        """
        Generate n_samples statistically stationary snapshots.

        t_start : burn-in time (discard transient)
        t_gap   : time gap between snapshots
        Returns : (n_samples, N, N) tensor of vorticity fields
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Spin up from random IC
        if verbose:
            print(f"[KF] Spinning up for t={t_start}s ...")
        omega = torch.randn(self.N, self.N, device=self.device) * 0.5
        n_burnin = int(t_start / self.dt)
        omega = self.run(omega0=omega, n_steps=n_burnin, verbose=verbose)

        # Collect snapshots
        snapshots = []
        n_gap = int(t_gap / self.dt)
        if verbose:
            print(f"[KF] Collecting {n_samples} snapshots (gap={t_gap}s) ...")
        for i in range(n_samples):
            omega = self.run(omega0=omega, n_steps=n_gap)
            snapshots.append(omega.cpu().clone())
            if verbose and (i % 20 == 0):
                print(f"  snapshot {i+1}/{n_samples}")

        data = torch.stack(snapshots, dim=0)   # (n_samples, N, N)
        if verbose:
            print(f"[KF] Done. shape={data.shape}, "
                  f"|ω| range=[{data.min():.2f}, {data.max():.2f}]")
        return data


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = KolmogorovFlowGenerator(N=64, Re=1000, T=20.0, dt=1e-3, k_f=4)
    data = gen.generate(n_samples=10, t_start=5.0, t_gap=0.5, verbose=True)
    print("Generated:", data.shape)
