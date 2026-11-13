"""
GRBD — Basis Solvers
Implements the unified BasisSolver interface with:
  - FourierBasis  : top-K 2D FFT modes
  - WaveletBasis  : 2D DWT (PyWavelets), multi-level
"""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
import pywt


# ──────────────────────────────────────────────
# Abstract interface
# ──────────────────────────────────────────────

class BasisSolver(ABC):
    """
    Unified interface for all basis decompositions.

    All tensors: shape (B, N, N)  — batch of 2D scalar fields.
    coefficients: shape (B, A)    — A = n_atoms() flattened complex or real coefficients
    features:     shape (B, A, d) — per-atom feature vectors for the network
    """

    @abstractmethod
    def decompose(self, residual: torch.Tensor) -> torch.Tensor:
        """r (B,N,N) → coefficients (B, A)  [complex or real]"""

    @abstractmethod
    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """coefficients (B, A) → field (B, N, N)"""

    @abstractmethod
    def atom_features(self, coefficients: torch.Tensor) -> torch.Tensor:
        """coefficients (B, A) → features (B, A, d_feat)"""

    @abstractmethod
    def n_atoms(self) -> int:
        """Number of atoms (fixed for a given N)."""

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        """Feature dimension d per atom."""


# ──────────────────────────────────────────────
# Fourier Basis
# ──────────────────────────────────────────────

class FourierBasis(BasisSolver):
    """
    Top-K 2D Fourier modes.

    decompose : 2D rfft2 → keep top-K modes by |C|
    reconstruct: irfft2 after placing weighted coefficients back
    features  : [kx_norm, ky_norm, |C|, log(|C|+ε), phase/π]  → d=5
    """

    def __init__(self, N: int, K: int):
        """
        N : spatial resolution (N×N grid)
        K : number of top Fourier modes to retain
        """
        self.N = N
        self.K = K
        # Precompute frequency grids — shape (N, N//2+1)
        fx = torch.fft.fftfreq(N)           # (N,)
        fy = torch.fft.rfftfreq(N)          # (N//2+1,)
        kx, ky = torch.meshgrid(fx, fy, indexing='ij')   # (N, N//2+1)
        self._kx_flat = kx.reshape(-1)      # (N*(N//2+1),)
        self._ky_flat = ky.reshape(-1)
        self._n_full  = N * (N // 2 + 1)   # total rfft2 coefficients

    def _top_k_indices(self, C_flat_abs: torch.Tensor) -> torch.Tensor:
        """
        C_flat_abs : (B, M)  absolute values
        returns    : (B, K)  indices of top-K per batch
        """
        B = C_flat_abs.shape[0]
        k = min(self.K, C_flat_abs.shape[1])
        _, idx = torch.topk(C_flat_abs, k, dim=-1)
        return idx  # (B, K)

    def decompose(self, residual: torch.Tensor) -> torch.Tensor:
        """
        residual : (B, N, N)
        returns  : (B, K, 2)  — stacked [Re, Im] of top-K modes
                   + stores indices in self._last_idx for reconstruct
        """
        B = residual.shape[0]
        C = torch.fft.rfft2(residual)           # (B, N, N//2+1) complex
        C_flat = C.reshape(B, -1)               # (B, M)

        idx = self._top_k_indices(C_flat.abs(), )
        self._last_idx = idx.detach().cpu()     # (B, K) — save for reconstruct

        C_sel = torch.gather(C_flat, 1, idx)    # (B, K) complex
        # Pack as (B, K, 2) real tensor
        coef = torch.stack([C_sel.real, C_sel.imag], dim=-1)
        return coef  # (B, K, 2)

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        coefficients : (B, K, 2)  [Re, Im] of selected modes
        returns      : (B, N, N)
        Uses self._last_idx from last decompose call.
        """
        B = coefficients.shape[0]
        device = coefficients.device
        idx = self._last_idx.to(device)

        C_flat = torch.zeros(B, self._n_full, dtype=torch.complex64, device=device)
        C_sel  = torch.complex(coefficients[..., 0], coefficients[..., 1])  # (B, K)
        C_flat.scatter_(1, idx, C_sel)

        C = C_flat.reshape(B, self.N, self.N // 2 + 1)
        out = torch.fft.irfft2(C, s=(self.N, self.N))  # (B, N, N)
        return out

    def atom_features(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        coefficients : (B, K, 2)
        returns      : (B, K, 5)
          [kx_norm, ky_norm, |C|, log(|C|+1e-8), phase/π]
        """
        B, K, _ = coefficients.shape
        device = coefficients.device

        kx = self._kx_flat[self._last_idx].to(device)  # (B, K)
        ky = self._ky_flat[self._last_idx].to(device)

        re, im = coefficients[..., 0], coefficients[..., 1]
        amp   = (re**2 + im**2).sqrt()                  # (B, K)
        phase = torch.atan2(im, re) / np.pi             # (B, K) ∈ [-1,1]
        log_a = (amp + 1e-8).log()                      # (B, K)

        feat = torch.stack([kx, ky, amp, log_a, phase], dim=-1)  # (B, K, 5)
        return feat

    def n_atoms(self) -> int:
        return self.K

    @property
    def feat_dim(self) -> int:
        return 5


# ──────────────────────────────────────────────
# Wavelet Basis
# ──────────────────────────────────────────────

class WaveletBasis(BasisSolver):
    """
    2D Haar DWT — pure PyTorch, fully differentiable.

    Uses Haar filters (simplest orthogonal wavelet) implemented as
    strided convolutions, so autograd works end-to-end.

    J levels of decomposition. Each level produces 3 detail sub-bands
    (LH, HL, HH) plus the final LL approximation.

    Total atoms A = N²  (Haar DWT is orthogonal, no redundancy).
    Coefficients are real.

    features : [level_norm, subband_id, |c|, log(|c|+ε), energy_frac] → d=5
    """

    def __init__(self, N: int, J: int, wavelet: str = 'haar'):
        self.N = N
        self.J = J
        # wavelet arg kept for API compatibility but only Haar implemented
        if wavelet not in ('haar', 'db1'):
            import warnings
            warnings.warn(f"Pure-torch WaveletBasis only supports Haar; ignoring '{wavelet}'")

        # Haar analysis filters (length-2, orthonormal)
        # Lo = [1,1]/√2,  Hi = [1,−1]/√2
        lo = torch.tensor([1.0,  1.0]) / np.sqrt(2)
        hi = torch.tensor([1.0, -1.0]) / np.sqrt(2)

        # 2D analysis kernels (1,1,2,2): LL, LH, HL, HH
        self._Lo_Lo = (lo.unsqueeze(1) * lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self._Lo_Hi = (hi.unsqueeze(1) * lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self._Hi_Lo = (lo.unsqueeze(1) * hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self._Hi_Hi = (hi.unsqueeze(1) * hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        # Synthesis filters (transpose = same for Haar due to orthogonality)
        self._syn_LL = self._Lo_Lo
        self._syn_LH = self._Lo_Hi
        self._syn_HL = self._Hi_Lo
        self._syn_HH = self._Hi_Hi

        # Precompute sub-band shapes & metadata for feature generation
        meta = []
        size = N
        for j in range(1, J + 1):
            size = size // 2
            for sid, sname in enumerate(['LH', 'HL', 'HH'], 1):
                meta.append({'level': j, 'subband': sid,
                             'size': size, 'n': size * size})
        # Final LL
        meta.append({'level': J, 'subband': 0, 'size': size, 'n': size * size})
        self._meta = meta
        self._total_atoms = N * N   # Haar DWT is a bijection on ℝ^{N×N}

        # Build flat metadata arrays (level, subband) for atom_features
        lvs, sbs = [], []
        for m in meta:
            lvs.extend([m['level'] / J] * m['n'])
            sbs.extend([m['subband'] / 3.0] * m['n'])
        self._lv_arr = torch.tensor(lvs, dtype=torch.float32)
        self._sb_arr = torch.tensor(sbs, dtype=torch.float32)

    def _get_kernels(self, device):
        return (self._Lo_Lo.to(device), self._Lo_Hi.to(device),
                self._Hi_Lo.to(device), self._Hi_Hi.to(device))

    def _dwt2_level(self, x: torch.Tensor):
        """
        One level of 2D Haar DWT.
        x : (B, 1, H, W) → LL, LH, HL, HH each (B, 1, H//2, W//2)
        Uses stride-2 convolution (= sub-sampled convolution).
        """
        device = x.device
        kLL, kLH, kHL, kHH = self._get_kernels(device)
        # Periodic padding by 1 on right/bottom to handle stride-2
        xp = torch.nn.functional.pad(x, (0, 1, 0, 1), mode='circular')
        LL = torch.nn.functional.conv2d(xp, kLL, stride=2)
        LH = torch.nn.functional.conv2d(xp, kLH, stride=2)
        HL = torch.nn.functional.conv2d(xp, kHL, stride=2)
        HH = torch.nn.functional.conv2d(xp, kHH, stride=2)
        return LL, LH, HL, HH

    def _idwt2_level(self, LL, LH, HL, HH):
        """
        One level of 2D Haar IDWT.
        Each input : (B, 1, H, W) → output (B, 1, 2H, 2W)
        Transpose convolution (= upsampling + filtering).
        """
        device = LL.device
        kLL, kLH, kHL, kHH = self._get_kernels(device)
        out  = torch.nn.functional.conv_transpose2d(LL, kLL, stride=2)
        out += torch.nn.functional.conv_transpose2d(LH, kLH, stride=2)
        out += torch.nn.functional.conv_transpose2d(HL, kHL, stride=2)
        out += torch.nn.functional.conv_transpose2d(HH, kHH, stride=2)
        # Circular wrap: the transpose conv adds 1 extra row/col
        # Crop and add the wrap-around contribution
        H2, W2 = LL.shape[2]*2, LL.shape[3]*2
        if out.shape[2] > H2:
            out[:, :, :1, :] += out[:, :, H2:, :]
            out = out[:, :, :H2, :]
        if out.shape[3] > W2:
            out[:, :, :, :1] += out[:, :, :, W2:]
            out = out[:, :, :, :W2]
        return out

    def decompose(self, residual: torch.Tensor) -> torch.Tensor:
        """
        residual : (B, N, N)
        returns  : (B, N²)  flattened coefficients [LH1, HL1, HH1, ..., LLJ]
        """
        B = residual.shape[0]
        x = residual.unsqueeze(1)    # (B, 1, N, N)

        parts = []
        for _ in range(self.J):
            LL, LH, HL, HH = self._dwt2_level(x)
            parts.extend([LH.reshape(B, -1), HL.reshape(B, -1), HH.reshape(B, -1)])
            x = LL
        parts.append(LL.reshape(B, -1))   # final approximation

        return torch.cat(parts, dim=1)    # (B, N²)

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        coefficients : (B, N²)
        returns      : (B, N, N)
        """
        B = coefficients.shape[0]
        device = coefficients.device

        # Unpack coefficients in reverse order
        pos = 0
        subband_list = []  # (LH, HL, HH) per level, finest first
        size = self.N
        shapes = []
        for j in range(self.J):
            size = size // 2
            shapes.append(size)

        # Parse: [LH_j=1, HL_j=1, HH_j=1, ..., LH_j=J, HL_j=J, HH_j=J, LL_J]
        detail_list = []
        sz = self.N
        for j in range(self.J):
            sz = sz // 2
            n = sz * sz
            LH = coefficients[:, pos:pos+n].reshape(B, 1, sz, sz); pos += n
            HL = coefficients[:, pos:pos+n].reshape(B, 1, sz, sz); pos += n
            HH = coefficients[:, pos:pos+n].reshape(B, 1, sz, sz); pos += n
            detail_list.append((LH, HL, HH))
        n = sz * sz
        LL = coefficients[:, pos:pos+n].reshape(B, 1, sz, sz)

        # Reconstruct from coarsest to finest
        x = LL
        for (LH, HL, HH) in reversed(detail_list):
            x = self._idwt2_level(x, LH, HL, HH)

        return x.squeeze(1)    # (B, N, N)

    def atom_features(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        coefficients : (B, A)  real
        returns      : (B, A, 5)
        """
        B, A = coefficients.shape
        device = coefficients.device

        lv = self._lv_arr[:A].to(device).unsqueeze(0).expand(B, -1)
        sb = self._sb_arr[:A].to(device).unsqueeze(0).expand(B, -1)

        amp    = coefficients.abs()
        log_a  = (amp + 1e-8).log()
        energy = amp**2
        e_frac = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)

        return torch.stack([lv, sb, amp, log_a, e_frac], dim=-1)

    def n_atoms(self) -> int:
        return self._total_atoms

    @property
    def feat_dim(self) -> int:
        return 5
