"""
Basis-MGDL  —  1D Helmholtz  (single-file, all bases)
======================================================
Three basis frameworks compared on three equations:

  Basis:     Fourier  |  Wavelet (DWT)  |  Wavelet Packet (WPT)
  Equation:  smooth   |  jump-coeff     |  multi-band

Design rules
------------
* Pure spectral method throughout — DST-I via rfft, no finite differences
* Spectral loss + spectral out-of-band penalty, all in torch (no scipy in loss)
* Backprop is standard autograd on network weights ONLY
  — the operator inv_L, DST, residual are all exact spectral, never differentiated
* W_sym symmetric frequency mixing (v8 style)
* LrAdaptor based on loss-tail slope
* HiddenStateBuffer: cross-grade hidden state (Multi-Grade, not Multi-Grid)
* SpectralGNN: optional GNN basis where nodes = selected basis functions
* Every data point recorded in info dict
* Dark theme visualisation matching v8 style
"""

# ── imports ────────────────────────────────────────────────────────────────────
import math, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import dst as scipy_dst, idst as scipy_idst

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── output dir & style ─────────────────────────────────────────────────────────
OUT = Path("./basis_mgdl_output"); OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":        "monospace",
    "axes.spines.top":    False, "axes.spines.right": False,
    "axes.grid":          True,  "grid.alpha":         0.18,
    "figure.facecolor":   "#0d0d0d", "axes.facecolor":  "#181818",
    "axes.labelcolor":    "#bbb",    "axes.titlecolor":  "#fff",
    "xtick.color":        "#888",    "ytick.color":      "#888",
    "grid.color":         "#2a2a2a", "text.color":       "#ccc",
    "legend.framealpha":  0.3,       "legend.facecolor": "#181818",
})
CGRADE  = plt.cm.plasma
BASIS_COLORS = {"fourier": "#4fc3f7", "wavelet": "#aed581", "wpacket": "#ff8a65", "gnn": "#ce93d8"}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SPECTRAL PRIMITIVES  (pure torch, no scipy in forward/loss)
# ══════════════════════════════════════════════════════════════════════════════

def dst1_torch(x: torch.Tensor) -> torch.Tensor:
    """DST Type-I via rfft. Works on last dimension. Exact up to float32 eps."""
    N = x.shape[-1]
    z = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    ext = torch.cat([z, x, z, -x.flip(-1)], dim=-1)
    fc  = torch.fft.rfft(ext, dim=-1)
    return -fc[..., 1:N+1].imag

def idst1_torch(c: torch.Tensor) -> torch.Tensor:
    """DST-I inverse: idst1(x) = dst1(x) / (2*(N+1))."""
    N = c.shape[-1]
    return dst1_torch(c) / (2.0 * (N + 1))

# numpy wrappers (used outside of training loop)
def dst1_np(x: np.ndarray) -> np.ndarray:
    return scipy_dst(x, type=1).astype(np.float64)

def idst1_np(c: np.ndarray) -> np.ndarray:
    return scipy_idst(c, type=1).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HELMHOLTZ OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

class HelmholtzOp:
    """
    -u'' + κ²u = f  on [0,1], Dirichlet BC u(0)=u(1)=0.
    Grid: x_i = i/(N+1),  i = 1..N   (N uniform interior points).
    Eigenvalues: λ_k = (πk)² - κ²

    All operations are exact spectral (DST-I):
        make_f:   u → L[u]
        residual: (u,f) → L[u] - f      (equation-space error)
        inv_L:    r → L⁻¹r = u_n - u*  (solution-space error, exact)
    """
    def __init__(self, N: int = 256, kappa: float = 10.0):
        self.N = N; self.kappa = kappa
        k = np.arange(1, N + 1, dtype=np.float64)
        self.x   = k / (N + 1)
        self.lam = (np.pi * k)**2 - kappa**2
        self._safe = np.where(np.abs(self.lam) > 1e-8, self.lam,
                              np.sign(self.lam + 1e-12) * 1e-8)

    def make_f(self, u: np.ndarray) -> np.ndarray:
        return idst1_np(self.lam * dst1_np(u)).astype(np.float32)

    def residual(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        return idst1_np(self.lam * dst1_np(u) - dst1_np(f)).astype(np.float32)

    def inv_L(self, r: np.ndarray) -> np.ndarray:
        """Exact spectral inverse. inv_L(e_n) = u_n - u*."""
        return idst1_np(dst1_np(r) / self._safe).astype(np.float32)

    def check_inv(self, u_star: np.ndarray) -> float:
        f  = self.make_f(u_star)
        e0 = self.residual(np.zeros(self.N, np.float32), f)
        chk = np.linalg.norm(-self.inv_L(e0) - u_star) / np.linalg.norm(u_star)
        return float(chk)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GD THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def gd_threshold(amps: np.ndarray, floor_ratio: float = 0.10) -> float:
    s   = np.sort(amps)[::-1]
    mad = np.median(np.abs(s - np.median(s)))
    sigma = mad / 0.6745 + 1e-12
    n = len(amps); beta = 1.0 / n
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    return float(max(omega * sigma, floor_ratio * s[0]))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  T_BASIS  — three input encodings
# ══════════════════════════════════════════════════════════════════════════════

def _log_norm_encode(amps_sel: np.ndarray, signs: np.ndarray,
                     basis_cols: np.ndarray) -> np.ndarray:
    """
    Shared encoding: s[:,i] = log_norm_i * sign_i * basis_col_i
    Returns (N, m) float32.
    """
    a_max = amps_sel.max() + 1e-12
    log_norm = np.log(1.0 + amps_sel) / np.log(1.0 + a_max)
    return (log_norm[None, :] * signs[None, :] * basis_cols).astype(np.float32)


def T_fourier(e_n: np.ndarray, op: HelmholtzOp,
              gd_floor: float = 0.10) -> Tuple[np.ndarray, dict]:
    """
    Fourier basis encoding (v8 style).
    Spectral decomposition of delta = inv_L(e_n) = u_n - u*.
    Selected frequencies: GD threshold on |DST(delta)|.
    s_n[:,k] = log_norm_k * sign_k * sin(π k x)
    """
    delta  = op.inv_L(e_n)                          # solution-space error
    d_hat  = dst1_np(delta)
    amps   = np.abs(d_hat)
    tau    = gd_threshold(amps, gd_floor)
    mask   = amps > tau
    if not mask.any(): mask[np.argmax(amps)] = True

    k_sel  = np.where(mask)[0] + 1                 # 1-indexed
    a_sel  = amps[mask]
    signs  = np.sign(d_hat[mask])

    # energy weights for SpectralLoss
    energy = a_sel**2
    alphas = energy / (energy.sum() + 1e-12)
    H      = float(-np.sum(alphas * np.log(alphas + 1e-12)))

    # basis columns: sin(π k x)
    x = op.x
    basis_cols = np.sin(np.pi * k_sel[None, :] * x[:, None])  # (N, m)
    s_n = _log_norm_encode(a_sel, signs, basis_cols)

    meta = dict(
        basis="fourier", k_sel=k_sel, mask=mask, tau=tau,
        k_all=np.arange(1, op.N+1), n_freqs=int(mask.sum()),
        amps_res=amps, e_hat=d_hat, amps_delta=amps,
        alphas=alphas, signs=signs, entropy=H,
        log_norm=np.log(1+a_sel)/np.log(1+a_sel.max()+1e-12),
        target=delta, target_hat=d_hat, s_input=s_n,
        # for SpectralLoss: mask over all N freqs
        mask_in=mask, mask_out=~mask,
    )
    return s_n, meta


def T_wavelet(e_n: np.ndarray, op: HelmholtzOp,
              wavelet: str = "db4", energy_ratio: float = 0.99,
              max_level: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Wavelet DWT encoding.
    Decomposes delta = inv_L(e_n) in Besov-style multi-resolution basis.
    Selected: top coefficients by cumulative energy.
    s_n[:,i] = log_norm_i * sign_i * psi_{j,k}(x)  (reconstructed basis vector)

    Advantage over Fourier: handles local singularities and jump discontinuities
    (Besov space B^s_{p,q} includes discontinuous functions).
    """
    N = op.N
    if max_level is None:
        max_level = int(math.log2(N)) - 2
    max_level = min(max_level, 5)

    delta  = op.inv_L(e_n)
    coeffs = pywt.wavedec(delta, wavelet, mode="periodization", level=max_level)

    # flatten all coefficients with labels
    entries = []
    for lvl, c in enumerate(coeffs):
        for pos, val in enumerate(c):
            entries.append((abs(val)**2, lvl, pos, val))
    total_e = sum(e[0] for e in entries) + 1e-12
    entries.sort(reverse=True)

    # cumulative energy selection
    sel, cum = [], 0.0
    for energy, lvl, pos, val in entries:
        sel.append((lvl, pos, val))
        cum += energy
        if cum / total_e >= energy_ratio:
            break
    if not sel:
        sel = [entries[0][1:]]

    # build basis vectors and s_n
    amps_sel  = np.array([abs(v) for _, _, v in sel])
    signs     = np.array([np.sign(v) for _, _, v in sel])
    energy_sel = amps_sel**2
    alphas    = energy_sel / (energy_sel.sum() + 1e-12)
    H         = float(-np.sum(alphas * np.log(alphas + 1e-12)))

    basis_cols = []
    for lvl, pos, _ in sel:
        zc = [np.zeros_like(c) for c in coeffs]
        zc[lvl][pos] = 1.0
        bv = pywt.waverec(zc, wavelet, mode="periodization")
        bv = bv[:N] if len(bv) > N else np.pad(bv, (0, N - len(bv)))
        basis_cols.append(bv)
    basis_cols = np.stack(basis_cols, axis=1)   # (N, m)
    s_n = _log_norm_encode(amps_sel, signs, basis_cols)

    # For SpectralLoss: compute which DST freqs are covered by these basis vectors
    all_amps_dst = np.abs(dst1_np(delta))
    mask_in  = all_amps_dst > gd_threshold(all_amps_dst, 0.10)
    if not mask_in.any(): mask_in[np.argmax(all_amps_dst)] = True

    sel_info = [(lvl, pos) for lvl, pos, _ in sel]
    log_norm_arr = np.log(1+amps_sel) / np.log(1+amps_sel.max()+1e-12)

    meta = dict(
        basis="wavelet", sel_info=sel_info, n_freqs=len(sel),
        k_all=np.arange(1, N+1), mask=mask_in, mask_in=mask_in, mask_out=~mask_in,
        tau=gd_threshold(all_amps_dst, 0.10),
        amps_res=all_amps_dst, e_hat=dst1_np(delta), amps_delta=all_amps_dst,
        alphas=alphas, signs=signs, entropy=H,
        log_norm=log_norm_arr, target=delta, target_hat=dst1_np(delta),
        s_input=s_n, wavelet=wavelet, max_level=max_level,
    )
    return s_n, meta


def T_wavelet_packet(e_n: np.ndarray, op: HelmholtzOp,
                     wavelet: str = "db4", max_level: int = 4,
                     energy_ratio: float = 0.99) -> Tuple[np.ndarray, dict]:
    """
    Wavelet Packet (WPT) encoding with Coifman-Wickerhauser best-basis.
    Selects the optimal subband partition by Shannon entropy cost.

    Advantage over DWT: decomposes both approximation AND detail subtrees,
    finding the best frequency partition for THIS specific residual.
    Optimal for signals with energy in multiple disjoint frequency bands.
    """
    N = op.N
    max_level = min(max_level, int(math.log2(N)) - 1)
    delta = op.inv_L(e_n)

    wp = pywt.WaveletPacket(data=delta, wavelet=wavelet,
                             mode="periodization", maxlevel=max_level)

    # Shannon entropy cost per node
    def entropy_cost(node_data: np.ndarray) -> float:
        p = node_data**2; s = p.sum() + 1e-12; p = p / s
        p = p[p > 1e-14]
        return float(-np.sum(p * np.log(p)))

    # Best-basis: bottom-up greedy
    def best_nodes(path: str, level: int) -> List[str]:
        if level >= max_level:
            return [path]
        ca, cd = path + "a", path + "d"
        try:
            cost_self     = entropy_cost(wp[path].data)
            cost_children = (entropy_cost(wp[ca].data) +
                             entropy_cost(wp[cd].data))
        except Exception:
            return [path]
        if cost_self <= cost_children:
            return [path]
        return best_nodes(ca, level + 1) + best_nodes(cd, level + 1)

    best = best_nodes("", 0)

    # Within each best-basis node: select coefficients by energy
    all_cols, amps_list, sel_info = [], [], []
    for node_path in best:
        nc = wp[node_path].data
        na = np.abs(nc)
        ne = na**2; net = ne.sum() + 1e-12
        idx_sorted = np.argsort(na)[::-1]
        cum, sel_pos = 0.0, []
        for idx in idx_sorted:
            sel_pos.append(idx)
            cum += ne[idx]
            if cum / net >= energy_ratio: break

        for pos in sel_pos:
            amps_list.append(na[pos])
            zc = np.zeros(len(nc)); zc[pos] = 1.0
            wu = pywt.WaveletPacket(data=None, wavelet=wavelet,
                                    mode="periodization", maxlevel=max_level)
            wu[node_path] = zc
            bv = wu.reconstruct(update=True)
            bv = bv[:N] if len(bv) > N else np.pad(bv, (0, N - len(bv)))
            all_cols.append(bv * np.sign(nc[pos]))
            sel_info.append(f"{node_path}[{pos}]")

    if not all_cols:  # fallback
        return T_wavelet(e_n, op, wavelet)

    amps_sel   = np.array(amps_list)
    signs      = np.ones(len(amps_sel))   # already absorbed into basis vectors
    energy_sel = amps_sel**2
    alphas     = energy_sel / (energy_sel.sum() + 1e-12)
    H          = float(-np.sum(alphas * np.log(alphas + 1e-12)))
    basis_cols = np.stack(all_cols, axis=1)
    s_n        = _log_norm_encode(amps_sel, signs, basis_cols)

    all_amps_dst = np.abs(dst1_np(delta))
    mask_in  = all_amps_dst > gd_threshold(all_amps_dst, 0.10)
    if not mask_in.any(): mask_in[np.argmax(all_amps_dst)] = True
    log_norm_arr = np.log(1+amps_sel) / np.log(1+amps_sel.max()+1e-12)

    meta = dict(
        basis="wpacket", sel_info=sel_info, n_freqs=len(sel_info),
        best_nodes=best, k_all=np.arange(1, N+1),
        mask=mask_in, mask_in=mask_in, mask_out=~mask_in,
        tau=gd_threshold(all_amps_dst, 0.10),
        amps_res=all_amps_dst, e_hat=dst1_np(delta), amps_delta=all_amps_dst,
        alphas=alphas, signs=signs, entropy=H,
        log_norm=log_norm_arr, target=delta, target_hat=dst1_np(delta),
        s_input=s_n, wavelet=wavelet, max_level=max_level,
    )
    return s_n, meta


# ══════════════════════════════════════════════════════════════════════════════
# 5.  NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

class GradeNet(nn.Module):
    """
    v8-style: W_sym symmetric frequency mixing + residual + MLP.
    Works for any basis — m = number of selected basis functions.

    Forward:
        s (N, m) → h_mix = tanh(s @ W_sym) + s → MLP → g (N,)
    """
    def __init__(self, m: int, width: int = 128, depth: int = 3):
        super().__init__()
        self.m = m
        self._W_raw  = nn.Parameter(torch.zeros(m, m))
        self.mix_act = nn.Tanh()
        self.proj    = nn.Linear(m, width)
        self.mlp     = nn.Sequential(*[
            layer for _ in range(depth)
            for layer in (nn.Linear(width, width), nn.Tanh())
        ])
        self.out = nn.Linear(width, 1)

        nn.init.normal_(self._W_raw, 0.0, 0.01)
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)
        nn.init.zeros_(self.proj.bias)
        for mod in list(self.mlp.modules()) + [self.out]:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=0.5)
                nn.init.zeros_(mod.bias)

    @property
    def W_sym(self) -> torch.Tensor:
        return 0.5 * (self._W_raw + self._W_raw.t())

    def forward(self, s: torch.Tensor,
                ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.mix_act(s @ self.W_sym) + s            # (N, m)
        if ctx is not None and ctx.shape[-1] >= self.m:
            h = h + ctx[..., :self.m]
        h = self.proj(h)
        h = self.mlp(h)
        self._last_hidden = h.detach()
        return self.out(h).squeeze(-1)                  # (N,)

    def get_last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


class SpectralGNN(nn.Module):
    """
    GNN where nodes = selected basis functions.
    Node features: (log_norm, sign, mean_basis_val, freq_or_level_idx).
    Edges: fully connected within grade, weight = cosine similarity of basis vectors.
    Message passing aggregates cross-basis information before MLP decode.

    Output: one correction value per spatial point (N,),
            computed by: g(x) = sum_i  h_i * basis_i(x)
    """
    def __init__(self, m: int, node_feat: int = 4,
                 hidden: int = 64, msg_steps: int = 2):
        super().__init__()
        self.m = m
        self.msg_steps = msg_steps

        # Node encoder
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        # Message function (edge: concat src+dst features)
        self.msg_fn = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        # Update (node feature after aggregation)
        self.update_fn = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
        )
        # Decoder: node scalar coefficient
        self.decoder = nn.Linear(hidden, 1)

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=0.5)
                if mod.bias is not None: nn.init.zeros_(mod.bias)

    def forward(self, node_feats: torch.Tensor,
                basis_mat: torch.Tensor,
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        node_feats:  (m, node_feat)
        basis_mat:   (N, m)   — basis vectors in physical space
        edge_weights:(m, m)   — cosine similarity matrix
        Returns g:   (N,)
        """
        h = self.node_enc(node_feats)               # (m, hidden)

        for _ in range(self.msg_steps):
            # message: for each (i,j), concat h_i, h_j, w_ij
            hi = h.unsqueeze(1).expand(-1, self.m, -1)   # (m, m, hidden)
            hj = h.unsqueeze(0).expand(self.m, -1, -1)   # (m, m, hidden)
            ew = edge_weights.unsqueeze(-1)               # (m, m, 1)
            msg_in = torch.cat([hi, hj, ew], dim=-1)     # (m, m, 2*hidden+1)
            msgs   = self.msg_fn(msg_in)                  # (m, m, hidden)
            agg    = msgs.sum(dim=1)                      # (m, hidden)
            h = self.update_fn(torch.cat([h, agg], dim=-1))  # (m, hidden)

        coeff = self.decoder(h).squeeze(-1)          # (m,)  scalar per basis
        g = basis_mat @ coeff                        # (N,)  spatial correction
        self._last_hidden = h.detach()
        return g

    def get_last_hidden(self) -> Optional[torch.Tensor]:
        return getattr(self, "_last_hidden", None)


def build_gnn_inputs(s_n: np.ndarray, meta: dict,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build node features, basis matrix, edge weights for SpectralGNN."""
    m = s_n.shape[1]
    log_norm = meta["log_norm"]                      # (m,)
    signs    = meta["signs"]                          # (m,)

    # Node feature 3: mean absolute value of basis vector in physical space
    mean_abs = np.abs(s_n).mean(axis=0)             # (m,)

    # Node feature 4: normalised index (frequency rank or level index)
    idx_norm = np.arange(m, dtype=np.float32) / max(m - 1, 1)

    node_feats = np.stack([log_norm, signs, mean_abs, idx_norm], axis=1)  # (m, 4)
    basis_mat  = torch.tensor(s_n, dtype=torch.float32, device=device)    # (N, m)

    # Edge weights: cosine similarity between basis columns
    s_t  = basis_mat                                  # (N, m)
    norm = s_t.norm(dim=0, keepdim=True).clamp(min=1e-8)  # (1, m)
    s_n2 = s_t / norm                                 # (N, m) normalised
    edge_weights = (s_n2.T @ s_n2).clamp(-1, 1)      # (m, m)

    return (torch.tensor(node_feats, dtype=torch.float32, device=device),
            basis_mat,
            edge_weights)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SPECTRAL LOSS  (all torch, no scipy)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralLoss(nn.Module):
    """
    In-band:     energy-weighted MSE on selected DST frequencies.
                 mask_in is a boolean array of length N (over all frequencies).
                 alphas must have the same length as mask_in.sum() — i.e. one
                 weight per selected frequency.
    Out-of-band: L2 penalty on network output at unselected frequencies.
    Total: L = L_ib + λ_oob * L_oob

    For wavelet/wpacket bases, mask_in is computed from GD threshold on DST(delta),
    so it still lives in frequency space (length N). alphas is re-computed to match.
    All DST operations use dst1_torch (rfft-based, no scipy).
    """
    def __init__(self, N: int, mask: np.ndarray, alphas: np.ndarray,
                 lambda_oob: float = 0.1):
        super().__init__()
        self.N = N
        self.norm = 2.0 / (N + 1)
        self.lambda_oob = lambda_oob
        # mask_in: (N,) bool — selected DST frequencies
        mask_bool = mask.astype(bool)
        self.register_buffer("mask_in",  torch.tensor(mask_bool,  dtype=torch.bool))
        self.register_buffer("mask_out", torch.tensor(~mask_bool, dtype=torch.bool))
        # alphas must be length = mask_in.sum()
        n_sel = int(mask_bool.sum())
        if len(alphas) == n_sel:
            a = alphas.astype(np.float32)
        else:
            # re-normalise to uniform if size mismatch (wavelet case)
            a = np.ones(n_sel, dtype=np.float32) / max(n_sel, 1)
        self.register_buffer("alphas", torch.tensor(a, dtype=torch.float32))

    def forward(self, g: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        g_hat   = dst1_torch(g)      * self.norm
        tgt_hat = dst1_torch(target) * self.norm

        diff2    = (g_hat - tgt_hat) ** 2
        loss_in  = (self.alphas * diff2[self.mask_in]).sum()
        loss_oob = (g_hat[self.mask_out] ** 2).mean() \
                   if self.mask_out.any() else g.new_zeros(1).squeeze()
        total = loss_in + self.lambda_oob * loss_oob
        return total, loss_in.detach().float().item(), loss_oob.detach().float().item()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  LR ADAPTOR  &  HIDDEN STATE BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class LrAdaptor:
    """Adjust next-grade lr based on tail slope of previous loss curve."""
    def __init__(self, base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                 tail_frac=0.20, decay_bad=0.50, decay_mild=0.70, boost_good=1.20):
        self.base_lr    = base_lr; self.lr_min = lr_min; self.lr_max = lr_max
        self.tail_frac  = tail_frac; self.decay_bad  = decay_bad
        self.decay_mild = decay_mild; self.boost_good = boost_good
        self._lr        = base_lr

    def update(self, losses: list, grade_n: int) -> float:
        if grade_n == 1:
            self._lr = self.base_lr; return self._lr
        losses = np.array(losses, dtype=np.float64)
        n = len(losses); ts = max(0, int(n * (1 - self.tail_frac)))
        tail = losses[ts:]
        slope = np.polyfit(np.arange(len(tail), dtype=float), tail, 1)[0]
        slope /= (abs(tail[0]) + 1e-30)
        ratio = tail.mean() / (losses[:max(1,ts)].mean() + 1e-30)
        lr = self._lr
        if slope < -1e-10:
            reason = f"descending slope={slope:.2e}"
        elif ratio > 0.5:
            lr *= self.decay_bad;  reason = f"poor ratio={ratio:.2f}"
        elif ratio < 0.01:
            lr *= self.boost_good; reason = f"excellent ratio={ratio:.2f}"
        else:
            lr *= self.decay_mild; reason = f"moderate ratio={ratio:.2f}"
        lr = float(np.clip(lr, self.lr_min, self.lr_max))
        self._lr = lr
        print(f"  [LrAdaptor] {reason}  →  lr={lr:.2e}")
        return lr

    @property
    def current_lr(self): return self._lr


class HiddenStateBuffer:
    """Cross-grade hidden state storage (Multi-Grade coupling)."""
    def __init__(self): self._store: Dict[int, torch.Tensor] = {}
    def push(self, grade_idx: int, h: torch.Tensor):
        self._store[grade_idx] = h.detach().clone()
    def get(self, grade_idx: int) -> Optional[torch.Tensor]:
        return self._store.get(grade_idx)
    def clear(self): self._store.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION  (v8 dark style, per-grade + summary)
# ══════════════════════════════════════════════════════════════════════════════

def plot_grade(n: int, op: HelmholtzOp, meta: dict, g_np: np.ndarray,
               u_before: np.ndarray, u_after: np.ndarray, u_star: np.ndarray,
               losses: list, losses_in: list, losses_oob: list,
               lr_used: float, basis_name: str, net_name: str, tag: str):
    color = CGRADE(n / 7.0)
    bc    = BASIS_COLORS.get(basis_name, "#ffffff")
    x     = op.x

    fig = plt.figure(figsize=(16, 24), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Basis-MGDL]  {tag}  Grade {n}  ·  basis={basis_name}  net={net_name}"
        f"  ·  κ={op.kappa:.1f}  ·  |Kn|={meta['n_freqs']}"
        f"  ·  H={meta['entropy']:.3f}  ·  lr={lr_used:.2e}",
        fontsize=12, fontweight="bold", color="white", y=0.99)
    gs = gridspec.GridSpec(6, 2, fig, hspace=0.55, wspace=0.35)

    # Row 0: residual spectrum
    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(meta["k_all"], meta["amps_res"] + 1e-12,
                color="#aaa", lw=0.9, label="|δ̂(k)| spectrum")
    ax.axhline(meta["tau"], color="#ff4455", lw=1.5, ls="--",
               label=f"GD τ={meta['tau']:.4f}")
    ax.scatter(meta["k_all"][meta["mask"]], meta["amps_res"][meta["mask"]],
               color=bc, s=25, zorder=5, label=f"selected  ({meta['n_freqs']})")
    ax.set_xlabel("k"); ax.legend(fontsize=9)
    ax.set_title("Residual/Delta Spectrum  —  GD Threshold")

    # Row 1 left: delta spectrum
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(meta["k_all"], np.abs(meta["amps_delta"]) + 1e-12,
                color="#888", lw=0.9, label="|δ̂(k)|")
    ax.scatter(meta["k_all"][meta["mask"]], np.abs(meta["amps_delta"])[meta["mask"]],
               color="#ff8844", s=20, zorder=5)
    ax.set_title("Delta Spectrum (target space)"); ax.set_xlabel("k"); ax.legend(fontsize=8)

    # Row 1 right: log_norm + alpha weights
    ax  = fig.add_subplot(gs[1, 1])
    xb  = np.arange(len(meta["log_norm"]))
    ax.bar(xb, meta["log_norm"], color=bc, alpha=0.85, width=0.4, label="log_norm")
    ax2 = ax.twinx()
    ax2.bar(xb + 0.4, meta["alphas"], color=color, alpha=0.5, width=0.4, label="α_k")
    ax.set_title("Log-Amp Norm (input) vs α_k (loss weight)")
    ax.set_ylabel("log_norm", color=bc); ax2.set_ylabel("α_k", color=color)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=7)

    # Row 2 left: input s_n columns
    ax = fig.add_subplot(gs[2, 0])
    s  = meta["s_input"]; nshow = min(8, s.shape[1])
    cols = plt.cm.coolwarm(np.linspace(0.05, 0.95, nshow))
    for i in range(nshow):
        lbl = (f"k={meta['k_sel'][i]}" if basis_name == "fourier"
               else f"b{i}")
        ax.plot(x, s[:, i], color=cols[i], lw=0.9, alpha=0.85, label=lbl)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title(f"Input s_n columns  ({basis_name})"); ax.set_xlabel("x")

    # Row 2 right: output vs target
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, meta["target"], color="#888", lw=1.2, ls="--", label="Target δ")
    ax.plot(x, g_np, color=color, lw=1.5, label=f"g_{n}(x)")
    ax.legend(fontsize=8); ax.set_title(f"Grade {n} Output vs Target"); ax.set_xlabel("x")

    # Row 3 left: cumulative solution
    ax = fig.add_subplot(gs[3, 0])
    eb = np.linalg.norm(u_before - u_star) / np.linalg.norm(u_star)
    ea = np.linalg.norm(u_after  - u_star) / np.linalg.norm(u_star)
    ax.plot(x, u_star,   color="white", lw=2,   label="u*", zorder=3)
    ax.plot(x, u_before, color="#666",  lw=1.2, ls="--", alpha=0.7, label=f"u_{n-1}  {eb:.2e}")
    ax.plot(x, u_after,  color=color,   lw=1.5, label=f"u_{n}  {ea:.2e}")
    ax.legend(fontsize=8); ax.set_title("Cumulative Solution"); ax.set_xlabel("x")

    # Row 3 right: pointwise error
    ax = fig.add_subplot(gs[3, 1])
    ax.semilogy(x, np.abs(u_before - u_star) + 1e-12, "--",
                color="#666", lw=1.2, alpha=0.7, label=f"before {eb:.3e}")
    ax.semilogy(x, np.abs(u_after  - u_star) + 1e-12,
                color=color, lw=1.5, label=f"after  {ea:.3e}")
    ax.legend(fontsize=8); ax.set_title("Pointwise Error |u*-u_n|"); ax.set_xlabel("x")

    # Row 4 left: training loss
    ax = fig.add_subplot(gs[4, 0])
    ax.semilogy(losses,     color=color,     lw=1.2, label="Total")
    ax.semilogy(losses_in,  color="#44ff88", lw=0.9, ls="--", alpha=0.7, label="In-band")
    ax.semilogy(losses_oob, color="#ff8844", lw=0.9, ls=":",  alpha=0.7, label="OOB")
    ax.legend(fontsize=7); ax.set_title(f"Training Loss  lr={lr_used:.2e}"); ax.set_xlabel("Epoch")

    # Row 4 right: alpha weights
    ax = fig.add_subplot(gs[4, 1])
    ax.bar(np.arange(len(meta["alphas"])), meta["alphas"], color=color, alpha=0.85)
    ax.set_title("α_k Energy Weights"); ax.set_ylabel("α_k")

    # Row 5: output spectrum
    ax = fig.add_subplot(gs[5, :])
    g_hat   = np.abs(dst1_np(g_np))
    tgt_hat = np.abs(meta["target_hat"])
    ax.semilogy(meta["k_all"], tgt_hat + 1e-12, color="#aaa",
                lw=1.0, ls="--", label="|δ̂(k)| target")
    ax.semilogy(meta["k_all"], g_hat   + 1e-12, color=color,
                lw=1.0, label="|ĝ(k)| output")
    sel_k = meta.get("k_sel", None)
    if sel_k is not None:
        ax.axvline(sel_k[0],  color=bc, lw=0.7, alpha=0.4)
        ax.axvline(sel_k[-1], color=bc, lw=0.7, alpha=0.4, label="K_n range")
        ax.scatter(sel_k, [g_hat[k-1] for k in sel_k], color=bc, s=20, zorder=5)
    ax.set_xlabel("k"); ax.legend(fontsize=8)
    ax.set_title("Output Spectrum vs Target  (out-of-band suppression)")

    fname = OUT / f"{tag}_grade{n:02d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  [saved] {fname.name}")


def plot_summary(infos: list, op: HelmholtzOp, u_star: np.ndarray,
                 u_pred: np.ndarray, tag: str):
    NG = len(infos)
    colors = [CGRADE(i / max(NG - 0.5, 1)) for i in range(NG)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="#0d0d0d")
    fig.suptitle(
        f"[Basis-MGDL]  Summary  {tag}  κ={op.kappa}  {NG} grades"
        f"  final err={infos[-1]['rel_err_after']:.2e}",
        fontsize=13, fontweight="bold", color="white")
    x = op.x

    ax = axes[0, 0]
    ax.plot(x, u_star, color="white", lw=2.5, label="u* (true)", zorder=3)
    ax.plot(x, u_pred, color="#ff6633", lw=1.5, ls="--", label="Predicted")
    ax.legend(); ax.set_title("Final Solution vs True")

    ax = axes[0, 1]
    eb = [i["rel_err_before"] for i in infos]
    ea = [i["rel_err_after"]  for i in infos]
    ax.semilogy(range(1, NG+1), eb, "o--", color="#888",    ms=5, label="Before")
    ax.semilogy(range(1, NG+1), ea, "s-",  color="#ff6633", ms=7, label="After")
    ax.set_title("Rel L2 Error per Grade"); ax.legend(); ax.set_xlabel("Grade")

    ax = axes[0, 2]
    ax.bar(range(1, NG+1), [i["n_freqs"] for i in infos], color=colors, alpha=0.85)
    ax.set_title("|K_n| per Grade"); ax.set_xlabel("Grade")

    ax = axes[1, 0]; ax2 = ax.twinx()
    l1, = ax.plot(range(1, NG+1), [i["entropy"] for i in infos],
                  "^-", color="#44ff88", lw=2, ms=8, label="H_n")
    lrs = [i.get("lr_used", float("nan")) for i in infos]
    l2, = ax2.semilogy(range(1, NG+1), lrs, "D--", color="#ffdd44",
                       lw=1.5, ms=7, label="lr")
    ax.legend(handles=[l1, l2]); ax.set_title("Spectral Entropy & lr")
    ax.set_ylabel("H_n", color="#44ff88"); ax2.set_ylabel("lr", color="#ffdd44")

    ax = axes[1, 1]
    for i, info in enumerate(infos):
        ax.semilogy(info["losses"], color=colors[i], lw=1.0, alpha=0.9,
                    label=f"G{i+1}")
    ax.legend(fontsize=8, ncol=2); ax.set_title("Total Loss per Grade")
    ax.set_xlabel("Epoch")

    ax = axes[1, 2]
    for i, info in enumerate(infos):
        ax.semilogy(np.abs(dst1_np(info["g_np"])) + 1e-12, color=colors[i],
                    lw=0.9, alpha=0.85, label=f"g_{i+1}")
    ax.legend(fontsize=8, ncol=2); ax.set_title("Output DST Spectrum per Grade")
    ax.set_xlabel("k")

    for row in axes:
        for a in row:
            a.set_facecolor("#181818")
            a.tick_params(colors="#888")
            a.grid(True, alpha=0.18, color="#2a2a2a")

    fname = OUT / f"{tag}_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  [saved] {fname.name}")


def plot_comparison(all_infos: dict, op: HelmholtzOp, tag: str):
    """Cross-basis comparison plot: error curves + per-grade reduction rates."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle(f"[Basis-MGDL]  Cross-Basis Comparison  —  {tag}",
                 fontsize=13, fontweight="bold", color="white")

    ax = axes[0]
    for basis, infos in all_infos.items():
        ea = [i["rel_err_after"] for i in infos]
        ax.semilogy(range(1, len(ea)+1), ea, "o-",
                    color=BASIS_COLORS.get(basis, "#fff"), lw=2, ms=7, label=basis)
    ax.set_title("Rel L2 Error (after each grade)"); ax.set_xlabel("Grade")
    ax.set_ylabel("Rel L2"); ax.legend(); ax.set_facecolor("#181818")

    ax = axes[1]
    for basis, infos in all_infos.items():
        rates = []
        for i in infos:
            eb, ea = i["rel_err_before"], i["rel_err_after"]
            rates.append(ea / (eb + 1e-14))
        ax.plot(range(1, len(rates)+1), rates, "s--",
                color=BASIS_COLORS.get(basis, "#fff"), lw=1.5, ms=6, label=basis)
    ax.axhline(1.0, color="#555", ls=":")
    ax.set_title("Per-grade Reduction Rate  (r_n = err_after/err_before)")
    ax.set_xlabel("Grade"); ax.set_ylabel("r_n"); ax.legend(); ax.set_facecolor("#181818")

    ax = axes[2]
    for basis, infos in all_infos.items():
        ax.bar([basis], [infos[-1]["rel_err_after"]],
               color=BASIS_COLORS.get(basis, "#fff"), alpha=0.85)
    ax.set_yscale("log"); ax.set_title("Final Error by Basis")
    ax.set_ylabel("Rel L2"); ax.set_facecolor("#181818")

    for a in axes:
        a.tick_params(colors="#888"); a.grid(True, alpha=0.18, color="#2a2a2a")

    fname = OUT / f"{tag}_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  [saved] {fname.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SOLVER
# ══════════════════════════════════════════════════════════════════════════════

T_BASIS_FNS = {
    "fourier":  T_fourier,
    "wavelet":  T_wavelet,
    "wpacket":  T_wavelet_packet,
}

class MultiGradeSolver:
    """
    Multi-Grade solver. At each grade:
      1. Compute residual e_n = L[u_n] - f  (spectral, exact)
      2. Build T_basis(e_n) → s_n  (Fourier / Wavelet / WPT)
      3. Train GradeNet or SpectralGNN on SpectralLoss
      4. Update u_{n+1} = u_n + g*_n(s_n)  (grade output added, not subtracted,
         because target = -inv_L(e_n) = u* - u_n, matching v8 convention)
      5. Freeze grade n, store hidden state in HiddenStateBuffer
    """
    def __init__(self, op: HelmholtzOp, f: np.ndarray, u_star: np.ndarray,
                 basis: str = "fourier",
                 net_type: str = "mlp",       # "mlp" or "gnn"
                 base_width: int = 64,
                 base_depth: int = 3,
                 lambda_oob: float = 0.1,
                 gd_floor: float = 0.10,
                 stop_delta_H: float = 0.05,
                 stop_min_improve: float = -0.02,
                 base_lr: float = 2e-3,
                 lr_min: float = 1e-4,
                 lr_max: float = 5e-3,
                 device: str = "cpu",
                 tag: str = "exp"):
        assert basis in T_BASIS_FNS, f"basis must be one of {list(T_BASIS_FNS)}"
        self.op   = op; self.f = f; self.u_star = u_star
        self.basis = basis; self.net_type = net_type
        self.base_width = base_width; self.base_depth = base_depth
        self.lambda_oob = lambda_oob; self.gd_floor = gd_floor
        self.stop_delta_H = stop_delta_H; self.stop_min_improve = stop_min_improve
        self.device = torch.device(device); self.tag = tag
        self.grades: list = []; self.infos: list = []
        self._u   = np.zeros(op.N, np.float32)
        self._next_residual: Optional[np.ndarray] = None
        self.lr_adaptor = LrAdaptor(base_lr, lr_min, lr_max)
        self.hsb = HiddenStateBuffer()

    def _config(self, H: float, nk: int, rel_err: float) -> Tuple[int, int]:
        depth = max(2, min(self.base_depth + int(H * 1.5), 8))
        cap   = 128 if rel_err < 1e-3 else (256 if rel_err < 1e-2 else 512)
        width = min(max(self.base_width, 2 * nk), cap)
        return width, depth

    def _should_stop(self) -> bool:
        if len(self.infos) < 2: return False
        h_prev = self.infos[-2]["entropy"]; h_curr = self.infos[-1]["entropy"]
        cond_A = abs(h_curr - h_prev) < self.stop_delta_H
        eb = self.infos[-1]["rel_err_before"]; ea = self.infos[-1]["rel_err_after"]
        imp = (eb - ea) / (eb + 1e-30)
        cond_B = imp < self.stop_min_improve
        cond_C = False
        if len(self.infos) >= 2:
            p_imp = ((self.infos[-2]["rel_err_before"] - self.infos[-2]["rel_err_after"])
                     / (self.infos[-2]["rel_err_before"] + 1e-30))
            cond_C = (imp < 0) and (p_imp < 0)
        stop = cond_A or cond_B or cond_C
        if stop:
            rs = []
            if cond_A: rs.append(f"ΔH<{self.stop_delta_H}")
            if cond_B: rs.append(f"improve={imp*100:.1f}%")
            if cond_C: rs.append("divergence")
            print(f"  [EARLY STOP]  {'  |  '.join(rs)}")
        return stop

    def add_grade(self, n_epochs: int = 3000) -> dict:
        n = len(self.grades) + 1
        print(f"\n{'═'*60}\n  GRADE {n}  [{self.basis}  {self.net_type}]\n{'═'*60}")

        # ── residual ──
        if self._next_residual is not None:
            e_n = self._next_residual; self._next_residual = None
            print(f"  [pre-computed residual from grade {n-1}]")
        else:
            e_n = self.op.residual(self._u, self.f)

        # v8 convention: target = -(u_n - u*) = u* - u_n = -inv_L(e_n)
        target = -self.op.inv_L(e_n)
        rel_b  = float(np.linalg.norm(self._u - self.u_star) /
                       np.linalg.norm(self.u_star))
        print(f"  Residual L2 : {np.linalg.norm(e_n):.4e}")
        print(f"  Rel L2 err  : {rel_b:.4e}")

        # ── lr ──
        lr = (self.lr_adaptor.update(self.infos[-1]["losses"], n-1)
              if n > 1 else self.lr_adaptor.base_lr)
        print(f"  lr          : {lr:.2e}")

        # ── T_basis ──
        t_fn = T_BASIS_FNS[self.basis]
        s_np, meta = t_fn(e_n, self.op, self.gd_floor) \
                     if self.basis == "fourier" else t_fn(e_n, self.op)
        m = s_np.shape[1]
        width, depth = self._config(meta["entropy"], m, rel_b)
        print(f"  |Kn|={m}  H={meta['entropy']:.3f}  w={width}  d={depth}")

        # ── tensors ──
        s_t   = torch.tensor(s_np,   dtype=torch.float32, device=self.device)
        tgt_t = torch.tensor(target, dtype=torch.float32, device=self.device)

        # ── cross-grade context ──
        ctx = None
        if n > 1:
            prev_h = self.hsb.get(n - 1)
            if prev_h is not None:
                ctx = prev_h.to(self.device)   # (N, hidden)

        # ── network ──
        if self.net_type == "gnn":
            nf, bm, ew = build_gnn_inputs(s_np, meta, self.device)
            net = SpectralGNN(m, node_feat=4, hidden=width, msg_steps=2).to(self.device)
        else:
            net = GradeNet(m, width, depth).to(self.device)

        # ── loss & optimiser ──
        criterion = SpectralLoss(
            self.op.N, meta["mask_in"], meta["alphas"], self.lambda_oob
        ).to(self.device)
        opt   = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.005)

        losses: List[float] = []
        losses_in: List[float] = []
        losses_oob: List[float] = []

        # ── training ──
        net.train()
        for ep in range(n_epochs):
            opt.zero_grad()
            if self.net_type == "gnn":
                g = net(nf, bm, ew)
            else:
                g = net(s_t, ctx=ctx)
            loss, l_in, l_oob = criterion(g, tgt_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step(); sched.step()
            losses.append(loss.detach().float().item()); losses_in.append(l_in); losses_oob.append(l_oob)
            if (ep + 1) % 500 == 0:
                print(f"  ep{ep+1:4d}  total={float(loss):.4e}"
                      f"  in={l_in:.4e}  oob={l_oob:.4e}")

        # ── inference ──
        u_before = self._u.copy()
        net.eval()
        with torch.no_grad():
            if self.net_type == "gnn":
                g_np = net(nf, bm, ew).cpu().numpy()
            else:
                g_np = net(s_t, ctx=ctx).cpu().numpy()

        # store hidden state for next grade
        h = net.get_last_hidden()
        if h is not None:
            self.hsb.push(n, h.cpu())

        # v8 update: u_{n+1} = u_n + g (target = u* - u_n, so u_n + g → u*)
        self._u = self._u + g_np
        rel_a = float(np.linalg.norm(self._u - self.u_star) /
                      np.linalg.norm(self.u_star))
        arrow = "↓" if rel_a < rel_b else "↑"
        print(f"  Rel L2 after: {rel_a:.4e}  ({arrow}{abs(rel_a-rel_b)/rel_b*100:.1f}%)")

        # freeze
        net.requires_grad_(False)
        print(f"  Grade {n} FROZEN.")

        # pre-compute next residual
        e_next = self.op.residual(self._u, self.f)
        self._next_residual = e_next
        print(f"  Next residual L2: {np.linalg.norm(e_next):.4e}")

        info = {
            **meta,
            "depth": depth, "width": width,
            "losses": losses, "losses_in": losses_in, "losses_oob": losses_oob,
            "g_np": g_np,
            "rel_err_before": rel_b, "rel_err_after": rel_a,
            "lr_used": lr,
            # record every spatial point error
            "pointwise_err_before": np.abs(u_before - self.u_star),
            "pointwise_err_after":  np.abs(self._u   - self.u_star),
            "u_before": u_before.copy(),
            "u_after":  self._u.copy(),
        }
        self.grades.append(net)
        self.infos.append(info)

        plot_grade(n, self.op, info, g_np, u_before, self._u, self.u_star,
                   losses, losses_in, losses_oob, lr,
                   self.basis, self.net_type, self.tag)
        return info

    def solve(self, n_grades: int = 6, n_epochs: int = 3000) -> dict:
        for i in range(n_grades):
            self.add_grade(n_epochs)
            if self._should_stop():
                print(f"  Stopping after {i+1} grades."); break
        plot_summary(self.infos, self.op, self.u_star, self._u, self.tag)
        self._table()
        return {"infos": self.infos, "u_pred": self._u.copy()}

    def _table(self):
        print(f"\n{'═'*95}")
        print(f"{'G':^3}  {'|Kn|':^5}  {'H':^6}  {'w':^4}  {'d':^3}"
              f"  {'lr':^8}  {'err_before':^11}  {'err_after':^11}  {'Δ%':^8}")
        print("─" * 95)
        for i, info in enumerate(self.infos):
            eb, ea = info["rel_err_before"], info["rel_err_after"]
            print(f" {i+1:^3}  {info['n_freqs']:^5}  {info['entropy']:^6.3f}"
                  f"  {info['width']:^4}  {info['depth']:^3}"
                  f"  {info['lr_used']:^8.2e}"
                  f"  {eb:^11.4e}  {ea:^11.4e}  {(eb-ea)/eb*100:^+7.1f}%")
        print("═" * 95)
        print(f"Final rel L2: {self.infos[-1]['rel_err_after']:.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(eq_name: str, u_star_fn, op: HelmholtzOp,
                   bases: List[str], net_types: List[str],
                   n_grades: int = 6, n_epochs: int = 2000,
                   device: str = "cpu") -> Dict[str, dict]:
    """Run one equation × multiple bases, collect results."""
    u_star = u_star_fn(op.x).astype(np.float32)
    f      = op.make_f(u_star)
    chk    = op.check_inv(u_star)
    print(f"\n{'█'*70}")
    print(f"  EQ: {eq_name}  |  inv_L check err={chk:.2e}")
    print(f"{'█'*70}")

    results = {}
    for basis in bases:
        for net_type in net_types:
            key = f"{basis}_{net_type}"
            tag = f"{eq_name}_{key}"
            print(f"\n{'▶'*3}  {tag}")
            solver = MultiGradeSolver(
                op=op, f=f, u_star=u_star,
                basis=basis, net_type=net_type,
                base_width=64, base_depth=3,
                lambda_oob=0.1, gd_floor=0.10,
                stop_delta_H=0.05, stop_min_improve=-0.02,
                base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                device=device, tag=tag,
            )
            res = solver.solve(n_grades=n_grades, n_epochs=n_epochs)
            results[key] = {"infos": solver.infos, "u_pred": solver._u.copy()}

    # comparison plot (all bases, one net_type)
    for net_type in net_types:
        all_infos = {b: results[f"{b}_{net_type}"]["infos"]
                     for b in bases if f"{b}_{net_type}" in results}
        plot_comparison(all_infos, op, f"{eq_name}_{net_type}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] device = {device}")

    N = 128; kappa = 10.0
    op = HelmholtzOp(N=N, kappa=kappa)

    # ── Equation 1: smooth (Fourier should win) ──
    def u_smooth(x): return np.sin(3*np.pi*x) + 0.5*np.sin(7*np.pi*x) + 0.3*np.sin(11*np.pi*x)

    # ── Equation 2: multi-band (WPT should win) ──
    def u_multiband(x): return np.sin(3*np.pi*x) + np.sin(31*np.pi*x) + np.sin(63*np.pi*x)

    # ── Equation 3: local bump (Wavelet should win) ──
    # Localised solution — energy concentrated spatially, not just in frequency
    def u_local(x): return np.exp(-50*(x - 0.3)**2) + 0.5*np.exp(-80*(x - 0.7)**2)

    BASES     = ["fourier", "wavelet", "wpacket"]
    NET_TYPES = ["mlp", "gnn"]
    N_GRADES  = 5
    N_EPOCHS  = 1500

    all_results = {}

    for eq_name, u_fn in [
        ("eq1_smooth",    u_smooth),
        ("eq2_multiband", u_multiband),
        ("eq3_local",     u_local),
    ]:
        all_results[eq_name] = run_experiment(
            eq_name, u_fn, op,
            bases=BASES, net_types=NET_TYPES,
            n_grades=N_GRADES, n_epochs=N_EPOCHS, device=device,
        )

    print(f"\n[✓] All experiments done. Outputs in {OUT.resolve()}")
