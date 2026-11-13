"""
nls2d_gnn_module.py
===================
SpectralGNN adapted for 2D NLS grade solver.

Key differences from 1D Helmholtz version:
  - Nodes are 2D modes (kx, ky) instead of 1D k
  - 'selection' edge weight encodes four-wave resonance condition
  - Node features live in (Nx, Ny, d) instead of (N, d)
  - Spatial dims are flattened to (Nx*Ny, d) internally for MLP ops

Edge types supported:
  energy    : E_ij = alpha_i * alpha_j  (general coupling via energy)
  selection : E_ij = exp(-|omega_i - omega_j| / gamma)  (NLS resonance)
  distance  : E_ij = exp(-|k_i - k_j| / sigma)  (spectral locality)
  cascade   : E_ij = alpha_i * max(|k_j|-|k_i|, 0)  (forward cascade)
  mlp       : E_ij = MLP(k_i, k_j, alpha_i, alpha_j)  (learnable, upper bound)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Helper: small MLP factory
# ---------------------------------------------------------------------------

def make_mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2,
             activation=nn.SiLU) -> nn.Sequential:
    layers = []
    d_in = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d_in, hidden), activation()]
        d_in = hidden
    layers.append(nn.Linear(d_in, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Eigenvalue embedding for 2D modes
# ---------------------------------------------------------------------------

class EigenvalueEmbedding2D(nn.Module):
    """
    Embeds the 2D NLS dispersion relation omega(k) = (kx² + ky²) / 2.

    Uses Fourier feature embedding of log|omega| for multi-scale representation.
    """

    def __init__(self, d_out: int = 16):
        super().__init__()
        self.d_out = d_out
        self.n_freq = d_out // 2

    def forward(self, kvecs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        kvecs : (K, 2)  physical (kx, ky) integer wavenumbers

        Returns
        -------
        emb : (K, d_out)
        """
        kx, ky = kvecs[:, 0], kvecs[:, 1]
        omega = (kx**2 + ky**2) / 2.0  # (K,)

        # Log-scale embedding
        log_om = torch.sign(omega) * torch.log(torch.abs(omega) + 1.0)  # (K,)

        # Fourier features at octave frequencies
        freqs = 2.0 ** torch.arange(self.n_freq, dtype=kvecs.dtype,
                                     device=kvecs.device)  # (n_freq,)
        args = log_om.unsqueeze(1) * freqs.unsqueeze(0) * np.pi  # (K, n_freq)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (K, d_out)
        return emb


# ---------------------------------------------------------------------------
# Edge weight builder for 2D NLS
# ---------------------------------------------------------------------------

class EdgeBuilder2D(nn.Module):
    """
    Builds edge weight matrix E ∈ R^{K × K} encoding physical coupling.

    Parameters
    ----------
    edge_type : str
        'energy' | 'selection' | 'distance' | 'cascade' | 'mlp'
    """

    def __init__(self, edge_type: str = 'selection', mlp_hidden: int = 32):
        super().__init__()
        self.edge_type = edge_type

        if edge_type == 'mlp':
            # Input: (kx_i, ky_i, kx_j, ky_j, alpha_i, alpha_j, omega_i, omega_j)
            self.edge_mlp = make_mlp(8, mlp_hidden, 1, depth=2)

        # Learnable temperature for selection
        if edge_type == 'selection':
            self.log_gamma = nn.Parameter(torch.tensor(0.0))  # gamma = exp(log_gamma)

        if edge_type == 'distance':
            self.log_sigma = nn.Parameter(torch.tensor(1.0))

    def forward(self, kvecs: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        kvecs : (K, 2)  physical wavenumber vectors (integers)
        alpha : (K,)    energy weights

        Returns
        -------
        E : (K, K)  non-negative edge weights
        """
        K = len(kvecs)
        kx, ky = kvecs[:, 0].float(), kvecs[:, 1].float()
        omega = (kx**2 + ky**2) / 2.0  # (K,)

        if self.edge_type == 'energy':
            # E_ij = alpha_i * alpha_j / Z
            E = torch.outer(alpha, alpha)
            E = E / (E.sum() + 1e-12)

        elif self.edge_type == 'selection':
            # NLS four-wave resonance: omega_i + omega_j ≈ omega_k + omega_l
            # Pairwise resonance mismatch: |omega_i - omega_j|
            # (modes with similar dispersion are resonant partners)
            gamma = torch.exp(self.log_gamma).clamp(min=0.01, max=10.0)
            dom = torch.abs(omega.unsqueeze(0) - omega.unsqueeze(1))  # (K, K)
            E = torch.exp(-dom / gamma)
            # Weight by energy
            energy_weight = torch.outer(alpha.sqrt(), alpha.sqrt())
            E = E * energy_weight
            E = E / (E.max() + 1e-12)

        elif self.edge_type == 'distance':
            # Spectral distance in k-space
            sigma = torch.exp(self.log_sigma).clamp(min=0.1, max=20.0)
            dkx = kx.unsqueeze(0) - kx.unsqueeze(1)  # (K, K)
            dky = ky.unsqueeze(0) - ky.unsqueeze(1)
            dk = torch.sqrt(dkx**2 + dky**2)
            E = torch.exp(-dk / sigma)

        elif self.edge_type == 'cascade':
            # Forward cascade: alpha_i drives modes at higher |k|
            k_mag = torch.sqrt(kx**2 + ky**2)  # (K,)
            dk_forward = (k_mag.unsqueeze(0) - k_mag.unsqueeze(1)).clamp(min=0)  # (K,K): j->i
            # E_ij ~ alpha_j * (k_i - k_j)+ (energy flows from low to high k)
            E = alpha.unsqueeze(0) * dk_forward  # E[i,j] = alpha_j * dk_ij+
            E = E / (E.max() + 1e-12)

        elif self.edge_type == 'mlp':
            # Fully learnable
            kx_i = kx.unsqueeze(1).expand(K, K)
            ky_i = ky.unsqueeze(1).expand(K, K)
            kx_j = kx.unsqueeze(0).expand(K, K)
            ky_j = ky.unsqueeze(0).expand(K, K)
            om_i = omega.unsqueeze(1).expand(K, K)
            om_j = omega.unsqueeze(0).expand(K, K)
            al_i = alpha.unsqueeze(1).expand(K, K)
            al_j = alpha.unsqueeze(0).expand(K, K)
            feat = torch.stack([kx_i, ky_i, kx_j, ky_j,
                                 al_i, al_j, om_i, om_j], dim=-1)  # (K,K,8)
            raw = self.edge_mlp(feat.reshape(-1, 8)).reshape(K, K)
            E = torch.sigmoid(raw)

        else:
            raise ValueError(f"Unknown edge_type: {self.edge_type}")

        return E  # (K, K)


# ---------------------------------------------------------------------------
# PI-GNN Layer (2D spatial)
# ---------------------------------------------------------------------------

class PIGNNLayer2D(nn.Module):
    """
    One message-passing layer for 2D NLS grade solver.

    Node features H: (K, Nx*Ny, d)  [spatial dims flattened]
    Message passing over K frequency nodes, updates per spatial location.

    Dense path (K ≤ K_dense_thresh): exact attention-weighted aggregation
    Mean-field path (K > K_dense_thresh): memory-efficient E-weighted sum
    """

    DENSE_THRESH = 16  # Use dense path if K ≤ 16

    def __init__(self, d: int, d_lam: int = 16, mlp_depth: int = 2):
        super().__init__()
        self.d = d

        # Dense path: message and update MLPs
        self.msg_mlp = make_mlp(2*d + 2*d_lam, d, d, depth=mlp_depth)
        self.upd_mlp = make_mlp(2*d, d, d, depth=mlp_depth)

        # Mean-field path
        self.mf_mlp = make_mlp(2*d + d_lam, d, d, depth=mlp_depth)
        self.mf_upd = make_mlp(2*d, d, d, depth=mlp_depth)

    def forward(self, H: torch.Tensor, E: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H   : (K, S, d)   node features, S = Nx*Ny
        E   : (K, K)      edge weight matrix
        emb : (K, d_lam)  eigenvalue embeddings

        Returns
        -------
        H_new : (K, S, d)
        """
        K, S, d = H.shape

        if K <= self.DENSE_THRESH:
            return self._dense_forward(H, E, emb)
        else:
            return self._meanfield_forward(H, E, emb)

    def _dense_forward(self, H, E, emb):
        K, S, d = H.shape
        d_lam = emb.shape[1]

        # Expand embeddings to spatial dim
        emb_exp = emb.unsqueeze(1).expand(K, S, -1)  # (K, S, d_lam)

        # Build messages for all pairs (i, j)
        H_i = H.unsqueeze(1).expand(K, K, S, d)   # (K, K, S, d): H[i] repeated
        H_j = H.unsqueeze(0).expand(K, K, S, d)   # (K, K, S, d): H[j] repeated
        emb_i = emb_exp.unsqueeze(1).expand(K, K, S, d_lam)
        emb_j = emb_exp.unsqueeze(0).expand(K, K, S, d_lam)

        msg_in = torch.cat([H_i, H_j, emb_i, emb_j], dim=-1)  # (K, K, S, 2d+2d_lam)
        msg = self.msg_mlp(msg_in)  # (K, K, S, d)

        # Attention score: <H_i, msg_ij>_S / sqrt(d)
        score = (H_i * msg).mean(dim=2) / np.sqrt(d)  # (K, K, S... no) → (K, K) pooled over S... hmm
        # Actually per-spatial: (K, K, S)
        score_s = (H_i * msg).sum(dim=-1) / np.sqrt(d)  # (K, K, S)

        # Edge-masked attention
        E_exp = E.unsqueeze(-1)  # (K, K, 1)
        score_s = score_s * E_exp  # (K, K, S)
        attn = F.softmax(score_s, dim=1)  # (K, K, S) softmax over j

        # Aggregate: agg[i] = sum_j attn[i,j,s] * msg[i,j,s,d]
        agg = (attn.unsqueeze(-1) * msg).sum(dim=1)  # (K, S, d)

        H_new = H + self.upd_mlp(torch.cat([H, agg], dim=-1))
        return H_new

    def _meanfield_forward(self, H, E, emb):
        K, S, d = H.shape
        d_lam = emb.shape[1]

        # agg[i] = sum_j E[i,j] * H[j]   (K, S, d)
        agg = torch.einsum('ij,jsd->isd', E, H)  # O(K² S d) but no (K,K,S,d) tensor

        emb_exp = emb.unsqueeze(1).expand(K, S, -1)
        msg = self.mf_mlp(torch.cat([H, agg, emb_exp], dim=-1))  # (K, S, d)
        H_new = H + self.mf_upd(torch.cat([H, msg], dim=-1))
        return H_new


# ---------------------------------------------------------------------------
# Per-mode SubNet (independent per frequency)
# ---------------------------------------------------------------------------

class ModeSubNet2D(nn.Module):
    """
    Per-mode embedding network. Each mode k_i gets its own small MLP
    to project its spatial signal s[:,:,i] → h_i of dim d.
    """

    def __init__(self, d: int, mlp_depth: int = 2, mlp_hidden: int = None):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = d
        # Input: scalar per spatial location (flattened)
        self.net = make_mlp(1, mlp_hidden, d, depth=mlp_depth)

    def forward(self, s_i: torch.Tensor) -> torch.Tensor:
        """s_i: (S,) → h_i: (S, d)"""
        return self.net(s_i.unsqueeze(-1))  # (S, 1) → (S, d)


# ---------------------------------------------------------------------------
# SpectralGNN 2D — full model
# ---------------------------------------------------------------------------

class SpectralGNN2D(nn.Module):
    """
    2D NLS grade solver: learns correction δ(x,y) from spectral residual encoding.

    Architecture:
        ① Per-mode SubNet: s[:,:,i] → H[i] ∈ R^{S × d}
        ② Eigenvalue embedding: omega(k_i) → emb_i ∈ R^{d_lam}
        ③ Edge weight matrix E (encodes NLS physics)
        ④ L × PIGNNLayer2D message passing
        ⑤ Per-mode output head → g_i ∈ R^{S}
        ⑥ Learned frequency aggregation → δ ∈ R^{Nx × Ny}
    """

    def __init__(self, Nx: int, Ny: int,
                 d: int = 64, d_lam: int = 16,
                 gnn_layers: int = 2,
                 edge_type: str = 'selection',
                 mlp_depth: int = 2,
                 subnet_hidden: int = None):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.S = Nx * Ny
        self.d = d

        # Components (K is dynamic, built at forward time)
        self.emb_net = EigenvalueEmbedding2D(d_out=d_lam)
        self.edge_builder = EdgeBuilder2D(edge_type=edge_type)
        self.gnn_layers = nn.ModuleList([
            PIGNNLayer2D(d=d, d_lam=d_lam, mlp_depth=mlp_depth)
            for _ in range(gnn_layers)
        ])

        # Per-mode subnet and head (created dynamically in forward)
        self._subnet_cache = {}
        self._head_cache = {}
        self._subnet_hidden = subnet_hidden or d
        self._mlp_depth = mlp_depth
        self._d = d

        # Aggregation logits (resized dynamically)
        self._agg_logits = None
        self._last_K = 0

        self._d_lam = d_lam

    def _get_subnets(self, K: int, device) -> tuple:
        """Lazily create/cache per-mode subnet and head for current K."""
        if K not in self._subnet_cache:
            subnets = nn.ModuleList([
                ModeSubNet2D(self._d, self._mlp_depth, self._subnet_hidden)
                for _ in range(K)
            ]).to(device)
            heads = nn.ModuleList([
                nn.Linear(self._d, 1) for _ in range(K)
            ]).to(device)
            self._subnet_cache[K] = subnets
            self._head_cache[K] = heads
        return self._subnet_cache[K], self._head_cache[K]

    def _get_agg_logits(self, K: int, device) -> nn.Parameter:
        if K != self._last_K or self._agg_logits is None:
            self._agg_logits = nn.Parameter(torch.zeros(K, device=device))
            self._last_K = K
        return self._agg_logits

    def forward(self, s: torch.Tensor, kvecs: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        s      : (Nx, Ny, K)  spectral encoding
        kvecs  : (K, 2)       physical wavenumber vectors
        alpha  : (K,)         energy weights

        Returns
        -------
        delta : (Nx, Ny)  predicted correction (real)
        """
        K = s.shape[2]
        device = s.device

        # Flatten spatial dims: S = Nx*Ny
        s_flat = s.reshape(self.S, K)  # (S, K)

        # ① Per-mode embedding
        subnets, heads = self._get_subnets(K, device)
        H_list = []
        for i in range(K):
            h_i = subnets[i](s_flat[:, i])  # (S, d)
            H_list.append(h_i)
        H = torch.stack(H_list, dim=0)  # (K, S, d)

        # ② Eigenvalue embedding
        emb = self.emb_net(kvecs.to(device))  # (K, d_lam)

        # ③ Edge weights
        E = self.edge_builder(kvecs.to(device), alpha.to(device))  # (K, K)

        # ④ GNN message passing
        for layer in self.gnn_layers:
            H = layer(H, E, emb)  # (K, S, d)

        # ⑤ Per-mode output
        g_list = []
        for i in range(K):
            g_i = heads[i](H[i]).squeeze(-1)  # (S,)
            g_list.append(g_i)
        G = torch.stack(g_list, dim=1)  # (S, K)

        # ⑥ Learned aggregation
        agg_logits = self._get_agg_logits(K, device)
        w = F.softmax(agg_logits, dim=0)  # (K,)
        delta_flat = (G * w.unsqueeze(0)).sum(dim=1)  # (S,)

        return delta_flat.reshape(self.Nx, self.Ny)  # (Nx, Ny)


# ---------------------------------------------------------------------------
# GradeNet2D — v8 baseline (symmetric frequency mixing, 2D)
# ---------------------------------------------------------------------------

class GradeNet2D(nn.Module):
    """
    Baseline 2D grade network (v8 analog): symmetric frequency mixing + MLP.

    Takes s ∈ R^{Nx × Ny × K}, outputs δ ∈ R^{Nx × Ny}.
    """

    def __init__(self, Nx: int, Ny: int, K_max: int = 64,
                 width: int = 128, depth: int = 3):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.S = Nx * Ny

        # Symmetric frequency mixing (self-adjoint prior)
        self.W_raw = nn.Parameter(torch.randn(K_max, K_max) * 0.1)

        # MLP on mixed features
        layers = []
        d_in = K_max
        for _ in range(depth - 1):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

        self.K_max = K_max
        self.proj = nn.Linear(1, K_max)  # project single-channel s to K_max

    def forward(self, s: torch.Tensor, kvecs=None, alpha=None) -> torch.Tensor:
        """s: (Nx, Ny, K)  →  δ: (Nx, Ny)"""
        K = s.shape[2]
        S = self.S
        s_flat = s.reshape(S, K)  # (S, K)

        # Pad or truncate to K_max
        if K < self.K_max:
            pad = torch.zeros(S, self.K_max - K, device=s.device)
            s_pad = torch.cat([s_flat, pad], dim=1)  # (S, K_max)
        else:
            s_pad = s_flat[:, :self.K_max]

        # Symmetric frequency mixing
        W_sym = 0.5 * (self.W_raw + self.W_raw.T)
        h = torch.tanh(s_pad @ W_sym) + s_pad  # (S, K_max)

        # MLP per spatial location
        delta_flat = self.mlp(h).squeeze(-1)  # (S,)
        return delta_flat.reshape(self.Nx, self.Ny)


# ---------------------------------------------------------------------------
# Spectral loss for 2D grade solver
# ---------------------------------------------------------------------------

class SpectralLoss2D(nn.Module):
    """
    Loss = Σ_{k ∈ K_n} alpha_k |ĝ[k] - δ̂*[k]|²
         + lambda_oob * mean_{k ∉ K_n} |ĝ[k]|²

    In 2D: sums over (kx, ky) mode pairs.
    """

    def __init__(self, lambda_oob: float = 0.1):
        super().__init__()
        self.lambda_oob = lambda_oob

    def forward(self, g: torch.Tensor, delta_star: torch.Tensor,
                mode_indices: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g           : (Nx, Ny)  predicted correction (real)
        delta_star  : (Nx, Ny)  optimal correction (real, -residual approx)
        mode_indices: (K, 2)    active mode (ix, iy) indices
        alpha       : (K,)      energy weights
        """
        Nx, Ny = g.shape

        g_hat = torch.fft.fft2(g.to(torch.complex64))
        ds_hat = torch.fft.fft2(delta_star.to(torch.complex64))

        # In-band loss
        g_at = g_hat[mode_indices[:, 0], mode_indices[:, 1]]        # (K,)
        ds_at = ds_hat[mode_indices[:, 0], mode_indices[:, 1]]       # (K,)
        in_band = (alpha.to(g.device) * torch.abs(g_at - ds_at)**2).sum()

        # Out-of-band penalty
        oob_mask = torch.ones(Nx, Ny, dtype=torch.bool, device=g.device)
        oob_mask[mode_indices[:, 0], mode_indices[:, 1]] = False
        oob_loss = g_hat[oob_mask].abs().pow(2).mean()

        return in_band + self.lambda_oob * oob_loss


# ---------------------------------------------------------------------------
# Factory: build SpectralGNN2D with auto-sizing
# ---------------------------------------------------------------------------

def build_spectral_gnn_2d(Nx: int, Ny: int, K: int, rel_err: float,
                           edge_type: str = 'selection',
                           gnn_layers: int = 2,
                           base_width: int = 64,
                           d_lam: int = 16) -> SpectralGNN2D:
    """
    Auto-size SpectralGNN2D based on problem difficulty and K.

    Parameters
    ----------
    K       : number of active modes
    rel_err : current relative error (larger → need more capacity)
    """
    # Width: function of K and error
    width_cap = 128 if rel_err < 1e-2 else (64 if rel_err < 1e-1 else 32)
    if K <= PIGNNLayer2D.DENSE_THRESH:
        width_cap = min(width_cap, 64)  # dense path memory constraint
    width = max(base_width, min(K * 4, width_cap))

    # MLP depth
    depth = 2 if K <= 8 else 3

    return SpectralGNN2D(
        Nx=Nx, Ny=Ny,
        d=width,
        d_lam=d_lam,
        gnn_layers=gnn_layers,
        edge_type=edge_type,
        mlp_depth=depth,
        subnet_hidden=width,
    )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    Nx, Ny = 32, 32
    K = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    s = torch.randn(Nx, Ny, K).to(device)
    kvecs = torch.randint(-8, 8, (K, 2)).float().to(device)
    alpha = torch.softmax(torch.randn(K), dim=0).to(device)

    # Test GNN
    for et in ['energy', 'selection', 'distance', 'cascade']:
        model = build_spectral_gnn_2d(Nx, Ny, K, rel_err=0.1,
                                       edge_type=et).to(device)
        delta = model(s, kvecs, alpha)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  edge_type={et:12s}  delta.shape={tuple(delta.shape)}  params={n_params:,}")

    # Test baseline
    baseline = GradeNet2D(Nx, Ny, K_max=64, width=128).to(device)
    delta_b = baseline(s)
    print(f"  baseline GradeNet2D: delta.shape={tuple(delta_b.shape)}  "
          f"params={sum(p.numel() for p in baseline.parameters()):,}")

    # Test loss
    delta_star = torch.randn(Nx, Ny).to(device)
    mode_idx = torch.randint(0, Nx, (K, 2)).to(device)
    loss_fn = SpectralLoss2D(lambda_oob=0.1)
    loss = loss_fn(delta_b, delta_star, mode_idx, alpha)
    print(f"  SpectralLoss2D: {loss.item():.4f}")

    print("✓ All module smoke tests passed.")
