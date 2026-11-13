"""SpectralGNN - physics-informed spectral graph neural network.

Dual-path PI-GNN layer: an exact dense path for low mode counts
and a mean-field path for high mode counts. Drop-in replacement
for the v8 GradeNet constructor."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

M_DENSE_THRESH = 12


class EdgeBuilder(nn.Module):
    def __init__(self, k_sel, alphas, edge_type='energy', sigma=None, selection_tol=2):
        super().__init__()
        self.edge_type = edge_type
        m = len(k_sel); self.m = m
        k = torch.tensor(k_sel, dtype=torch.float32)
        a = torch.tensor(alphas, dtype=torch.float32)

        if edge_type == 'distance':
            sigma = sigma or (m / 3.0)
            diff = (k[:,None] - k[None,:]).abs()
            E = torch.exp(-diff / sigma)
            E = E / (E.sum(-1, keepdim=True) + 1e-8)
            self.register_buffer('E', E)
        elif edge_type == 'energy':
            E = a[:,None] * a[None,:]
            E = E / (E.sum(-1, keepdim=True) + 1e-8)
            self.register_buffer('E', E)
        elif edge_type == 'mlp':
            diff = (k[:,None] - k[None,:]).abs()
            feat = torch.stack([k[:,None].expand(m,m), k[None,:].expand(m,m),
                                a[:,None].expand(m,m), a[None,:].expand(m,m), diff],
                               dim=-1).reshape(m*m, 5)
            feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
            self.register_buffer('edge_feat', feat)
            self.edge_mlp = nn.Sequential(
                nn.Linear(5,16), nn.Tanh(), nn.Linear(16,1), nn.Sigmoid())
            self.register_buffer('E', None)
        elif edge_type == 'selection':
            diff = (k[:,None] - k[None,:]).abs()
            E = torch.exp(-diff/max(selection_tol,1)) * (diff<=selection_tol).float()
            E = E + 0.1*torch.eye(m)
            E = E / (E.sum(-1, keepdim=True) + 1e-8)
            self.register_buffer('E', E)
        elif edge_type == 'cascade':
            k_diff = k[None,:] - k[:,None]
            E = a[:,None].expand(m,m) * torch.clamp(k_diff, min=0)
            E = E + 0.1*torch.eye(m)*a[:,None]
            E = E / (E.sum(-1, keepdim=True) + 1e-8)
            self.register_buffer('E', E)
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")

    def forward(self):
        if self.edge_type == 'mlp':
            e = self.edge_mlp(self.edge_feat)
            E = e.reshape(self.m, self.m)
            return E / (E.sum(-1, keepdim=True) + 1e-8)
        return self.E


class PIGNNLayer(nn.Module):
    """PIGNNLayer."""
    def __init__(self, d, d_lam=16):
        super().__init__()

        self.msg_mlp = nn.Sequential(
            nn.Linear(2*d + 2*d_lam, d), nn.Tanh(), nn.Linear(d, d))

        self.mf_mlp = nn.Sequential(
            nn.Linear(2*d + d_lam, d), nn.Tanh(), nn.Linear(d, d))

        self.upd_mlp = nn.Sequential(
            nn.Linear(2*d, d), nn.Tanh(), nn.Linear(d, d))
        self.scale = d ** 0.5
        for mod in [self.msg_mlp, self.mf_mlp, self.upd_mlp]:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

    def forward(self, H, E, lam_emb):
        m = H.shape[0]
        if m <= M_DENSE_THRESH:
            return self._dense(H, E, lam_emb)
        return self._meanfield(H, E, lam_emb)

    def _dense(self, H, E, lam_emb):
        m, N, d = H.shape; d_lam = lam_emb.shape[-1]
        H_i = H.unsqueeze(1).expand(m, m, N, d)
        H_j = H.unsqueeze(0).expand(m, m, N, d)
        li = lam_emb.unsqueeze(1).unsqueeze(2).expand(m, m, N, d_lam)
        lj = lam_emb.unsqueeze(0).unsqueeze(2).expand(m, m, N, d_lam)
        M = self.msg_mlp(torch.cat([H_i, H_j, li, lj], dim=-1))   # (m,m,N,d)
        score = (H_i * M).sum(-1).mean(-1) / self.scale             # (m,m)
        alpha = F.softmax(score * E, dim=-1)                        # (m,m)
        agg = (alpha.unsqueeze(-1).unsqueeze(-1) * M).sum(1)        # (m,N,d)
        return H + self.upd_mlp(torch.cat([H, agg], dim=-1))

    def _meanfield(self, H, E, lam_emb):

        m, N, d = H.shape; d_lam = lam_emb.shape[-1]
        agg = torch.einsum('ij,jnd->ind', E, H)                     # (m,N,d)
        lam_exp = lam_emb.unsqueeze(1).expand(m, N, d_lam)
        M = self.mf_mlp(torch.cat([H, agg, lam_exp], dim=-1))       # (m,N,d)
        return H + self.upd_mlp(torch.cat([H, M], dim=-1))


class EigenvalueEmbedding(nn.Module):
    def __init__(self, lam_k_sel, d_lam=16):
        super().__init__()
        lam = np.array(lam_k_sel, dtype=np.float32)
        sign = np.sign(lam); log_lam = sign * np.log(np.abs(lam) + 1.0)
        freqs = 2.0 ** np.linspace(0, 4, d_lam // 2)
        emb = np.concatenate([np.sin(log_lam[:,None]*freqs[None,:]),
                               np.cos(log_lam[:,None]*freqs[None,:])], axis=-1).astype(np.float32)
        self.register_buffer('emb', torch.tensor(emb))

    def forward(self): return self.emb


class SpectralGNN(nn.Module):
    """SpectralGNN."""
    def __init__(self, meta, width, depth, lam_k,
                 edge_type='energy', d_lam=16, gnn_layers=2):
        super().__init__()
        self.m = meta['n_freqs']
        k_sel = meta['k_sel']; alphas = meta['alphas']
        lam_k_sel = lam_k[k_sel - 1]

        self.subnet_projs = nn.ModuleList([nn.Linear(1, width) for _ in range(self.m)])
        self.subnet_mlps  = nn.ModuleList([
            nn.Sequential(*[l for _ in range(depth) for l in (nn.Linear(width,width), nn.Tanh())])
            for _ in range(self.m)])
        self.eig_emb     = EigenvalueEmbedding(lam_k_sel, d_lam)
        self.edge_builder = EdgeBuilder(k_sel, alphas, edge_type)
        self.gnn          = nn.ModuleList([PIGNNLayer(width, d_lam) for _ in range(gnn_layers)])
        self.out_heads    = nn.ModuleList([nn.Linear(width, 1) for _ in range(self.m)])
        self.agg_logits   = nn.Parameter(torch.zeros(self.m))
        self._init_weights()

    def _init_weights(self):
        for p in self.subnet_projs:
            nn.init.xavier_normal_(p.weight); nn.init.zeros_(p.bias)
        for mlp in self.subnet_mlps:
            for l in mlp:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_normal_(l.weight, gain=0.5); nn.init.zeros_(l.bias)
        for h in self.out_heads:
            nn.init.xavier_normal_(h.weight, gain=0.3); nn.init.zeros_(h.bias)

    def forward(self, s):
        N, m = s.shape
        H = torch.stack([self.subnet_mlps[i](self.subnet_projs[i](s[:,i:i+1]))
                         for i in range(m)], dim=0)      # (m,N,d)
        E = self.edge_builder()
        lam_emb = self.eig_emb()
        for layer in self.gnn:
            H = layer(H, E, lam_emb)
        G = torch.stack([self.out_heads[i](H[i]).squeeze(1) for i in range(m)], dim=0)  # (m,N)
        w = F.softmax(self.agg_logits, dim=0)
        return (w[:,None] * G).sum(0).unsqueeze(1)        # (N,1)


def build_spectral_gnn(meta, rel_err, op,
                       base_width=64, base_depth=3,
                       edge_type='energy', gnn_layers=2, d_lam=16):
    """Build spectral gnn."""
    H_  = meta['entropy']; nk = meta['n_freqs']
    depth = max(2, min(base_depth + int(H_*1.5), 8))
    if rel_err < 1e-3:   width_cap = 128
    elif rel_err < 1e-2: width_cap = 256
    else:                width_cap = 512

    if nk <= M_DENSE_THRESH:
        width_cap = min(width_cap, 64)
    width = min(max(base_width, min(nk*4, base_width*2)), width_cap)
    net = SpectralGNN(meta=meta, width=width, depth=depth,
                      lam_k=op.lam_k, edge_type=edge_type,
                      d_lam=d_lam, gnn_layers=gnn_layers)
    return net, width, depth
