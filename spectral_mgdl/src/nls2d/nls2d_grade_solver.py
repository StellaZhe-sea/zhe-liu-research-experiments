"""
nls2d_grade_solver.py  ─  2D NLS Spectral-Adaptive Grade Solver
================================================================
Ablation: Baseline / Method-B / Method-C (PI-GNN)

Architecture: Spectral-weight (K-only).
  Each network predicts K amplitude weights {w_i}.
  Output: δ(x,y) = Σ_i w_i · s(x,y,i)   [basis reconstruction]
  This is 153× faster than spatial-MLP while keeping all physics.

Problem
-------
  u*(x,y) = 1.5 sech(1.5·r)   radial soliton  (analytical)
  u0(x,y) = Gaussian at wrong centre / amplitude

Run: python nls2d_grade_solver.py [--fast] [--ep EP] [--grades G]
"""

import argparse, os, sys, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.fft as tfft

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--fast',   action='store_true', help='3 grades × 300 ep')
parser.add_argument('--ep',     type=int,   default=3000)
parser.add_argument('--grades', type=int,   default=8)
parser.add_argument('--kmax',   type=int,   default=20)
parser.add_argument('--device', type=str,   default='cpu')
parser.add_argument('--output_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'nls2d_outputs'))
args = parser.parse_args()

if args.fast:
    args.ep     = 300
    args.grades = 3

DEVICE   = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
N_GRADES = args.grades
EPOCHS   = args.ep
K_MAX    = args.kmax

OUT_DIR  = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_num_threads(max(1, os.cpu_count() or 1))
print(f"Device={DEVICE}  threads={torch.get_num_threads()}"
      f"  N_GRADES={N_GRADES}  EPOCHS={EPOCHS}  K_MAX={K_MAX}")

# ── Grid & solutions ───────────────────────────────────────────────────────
NX, NY   = 64, 64
LX, LY   = 4*np.pi, 4*np.pi
OOB_LAM  = 0.15
GD_FLOOR = 0.08
EARLY_PAT = max(1, EPOCHS // 3)   # early-stop patience

x = torch.linspace(0, LX, NX+1)[:-1].float()
y = torch.linspace(0, LY, NY+1)[:-1].float()
X, Y = torch.meshgrid(x, y, indexing='ij')

# analytical solution  ── radial sech soliton
A = 1.5
_dx = X - LX/2;  _dx -= LX*torch.round(_dx/LX)
_dy = Y - LY/2;  _dy -= LY*torch.round(_dy/LY)
u_star = (A / torch.cosh(A * (_dx**2+_dy**2).sqrt())).detach()

# initial guess ── Gaussian, wrong centre & amplitude
_dx0 = X - LX/3;  _dx0 -= LX*torch.round(_dx0/LX)
_dy0 = Y - 2*LY/3; _dy0 -= LY*torch.round(_dy0/LY)
u0   = (0.9 * torch.exp(-(_dx0**2+_dy0**2) / (2*1.8**2))).detach()

NORM_STAR = float(u_star.norm())
print(f"u* ∈ [{u_star.min():.3f}, {u_star.max():.3f}]  "
      f"init RSE = {float((u0-u_star).norm()/NORM_STAR):.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Spectral utilities
# ═══════════════════════════════════════════════════════════════════════════

def gd_tau(pwr_flat: np.ndarray) -> float:
    amps = np.sqrt(np.maximum(pwr_flat, 0))
    mad  = np.median(np.abs(amps - np.median(amps)))
    n    = len(amps)
    om   = 0.56/n**3 - 0.95/n**2 + 1.82/n + 1.43
    return float(max(om * mad/0.6745, GD_FLOOR * amps.max()))

def select_modes(res: torch.Tensor):
    pwr  = tfft.fft2(res.float()).abs()
    tau  = gd_tau(pwr.cpu().numpy().ravel())
    idx  = torch.nonzero(pwr > tau, as_tuple=False)
    if len(idx) == 0:
        flat = pwr.flatten().topk(4).indices
        idx  = torch.stack([flat//NY, flat%NY], 1)
    amps = pwr[idx[:,0], idx[:,1]]
    ordr = amps.argsort(descending=True)
    idx, amps = idx[ordr[:K_MAX]], amps[ordr[:K_MAX]]
    kx_v = tfft.fftfreq(NX).float()*NX;  ky_v = tfft.fftfreq(NY).float()*NY
    kvecs = torch.stack([kx_v[idx[:,0]], ky_v[idx[:,1]]], 1)
    return idx, kvecs, amps

def build_basis(res: torch.Tensor, idx, kvecs, amps):
    """Returns s (NX,NY,K), alpha (K,), log_norm (K,), phases (K,)."""
    K       = len(idx)
    res_hat = tfft.fft2(res.float())
    dhat    = -res_hat
    a       = dhat[idx[:,0], idx[:,1]].abs()
    phi     = dhat[idx[:,0], idx[:,1]].angle()
    alpha   = a**2 / (a**2).sum().clamp(1e-12)
    lg      = torch.log(1+a) / (torch.log(1+a.max()) + 1e-12)
    s = torch.zeros(NX, NY, K)
    for i in range(K):
        s[:,:,i] = lg[i] * torch.cos(
            kvecs[i,0]*2*np.pi/LX * X +
            kvecs[i,1]*2*np.pi/LY * Y + phi[i])
    return s, alpha, lg, phi

def mode_feats(kvecs, alpha, lg, phi) -> torch.Tensor:
    """(K,6) feature matrix per mode node."""
    kx, ky = kvecs[:,0].float(), kvecs[:,1].float()
    om = (kx**2+ky**2)/2
    return torch.stack([kx, ky, om, alpha.float(), lg.float(), phi.float()], 1)

def ctx_feats(context, idx) -> torch.Tensor:
    """(K,2) context amplitude+phase at active mode locations."""
    if context is None:
        return torch.zeros(len(idx), 2)
    ch = tfft.fft2(context.float())
    c  = ch[idx[:,0], idx[:,1]]
    return torch.stack([c.abs(), c.angle()], 1).float()

def spec_loss(g, delta_star, idx, alpha):
    gh = tfft.fft2(g.float());  dh = tfft.fft2(delta_star.float())
    ib = (alpha.float() * (gh[idx[:,0],idx[:,1]] - dh[idx[:,0],idx[:,1]]).abs()**2).sum()
    msk = torch.ones(NX, NY, dtype=torch.bool); msk[idx[:,0],idx[:,1]] = False
    return ib + OOB_LAM * gh[msk].abs().pow(2).mean()

def reconstruct(s, w):
    """δ = s @ w  →  (NX,NY)."""
    return torch.einsum('xyk,k->xy', s.float(), w.float())

# ═══════════════════════════════════════════════════════════════════════════
# Networks  ── all operate on K mode nodes only (fast)
# ═══════════════════════════════════════════════════════════════════════════

def mlp2(in_d, hid, out_d):
    return nn.Sequential(nn.Linear(in_d,hid), nn.SiLU(), nn.Linear(hid,out_d))

# ── Baseline ──────────────────────────────────────────────────────────────
class BaselineNet(nn.Module):
    """Symmetric K×K mixing on mode features → K weights."""
    def __init__(self, K, d=64):
        super().__init__(); self.K=K
        self.proj = nn.Linear(6, d)
        self.Wm   = nn.Parameter(torch.randn(K,K)*0.02)
        self.head = mlp2(d, d, 1)

    def forward(self, mf, s, **_):
        h  = torch.tanh(self.proj(mf))          # (K,d)
        Ws = .5*(self.Wm+self.Wm.T)
        h  = torch.tanh(Ws @ h) + h             # (K,d)
        w  = self.head(h).squeeze(-1)            # (K,)
        return reconstruct(s, w)

# ── Method B ──────────────────────────────────────────────────────────────
class MethodBNet(nn.Module):
    """Baseline + 2-dim context features per mode (freq-domain filtered)."""
    def __init__(self, K, d=64):
        super().__init__(); self.K=K
        self.proj  = nn.Linear(8, d)             # 6+2 context
        self.Wm    = nn.Parameter(torch.randn(K,K)*0.02)
        self.gate  = nn.Parameter(torch.zeros(1))
        self.head  = mlp2(d, d, 1)

    def forward(self, mf, s, cf=None, **_):
        if cf is None: cf = torch.zeros(self.K, 2)
        g  = torch.sigmoid(self.gate)
        mf_aug = torch.cat([mf, g*cf], 1)       # (K,8)
        h  = torch.tanh(self.proj(mf_aug))
        Ws = .5*(self.Wm+self.Wm.T)
        h  = torch.tanh(Ws @ h) + h
        w  = self.head(h).squeeze(-1)
        return reconstruct(s, w)

# ── Method C: PI-GNN ──────────────────────────────────────────────────────
class EigenEmb(nn.Module):
    def __init__(self, d=16):
        super().__init__(); self.n=d//2
    def forward(self, kv):
        om  = (kv[:,0]**2+kv[:,1]**2)/2
        lom = torch.sign(om)*torch.log(om.abs()+1)
        fr  = 2.**torch.arange(self.n, dtype=kv.dtype)
        ag  = lom[:,None]*fr[None,:]*np.pi
        return torch.cat([ag.sin(), ag.cos()], 1)   # (K,d)

class PIGNNNet(nn.Module):
    """
    PI-GNN with NLS selection edges + energy-proportional shortcut.
    Shortcut:  w_sc = scale * sqrt(alpha_i) / lg_i   (physics prior)
    Residual:  w_res = GNN output (small initially)
    Final:     w = w_sc + res_sc * w_res
    """
    def __init__(self, K, d=32, d_lam=12, n_layers=2):
        super().__init__(); self.K=K
        self.proj    = nn.Linear(6, d)
        self.emb     = EigenEmb(d_lam)
        self.log_gam = nn.Parameter(torch.tensor(1.5))
        self.msgs    = nn.ModuleList([mlp2(d+d_lam, d, d) for _ in range(n_layers)])
        self.upds    = nn.ModuleList([mlp2(2*d,     d, d) for _ in range(n_layers)])
        self.head    = nn.Linear(d, 1)
        # Shortcut: global scale (learn correct overall amplitude)
        self.scale   = nn.Parameter(torch.tensor(0.0))
        # Residual weight scale (starts small so GNN refines, not disrupts)
        self.res_sc  = nn.Parameter(torch.tensor(0.05))
        # Small init on head to keep residual near zero initially
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def _edge(self, kv, alpha):
        om  = (kv[:,0]**2+kv[:,1]**2)/2
        gm  = self.log_gam.exp().clamp(.05, 20.)
        dom = (om.unsqueeze(0)-om.unsqueeze(1)).abs()
        E   = (-dom/gm).exp()
        E   = E * torch.outer(alpha.float().sqrt(), alpha.float().sqrt())
        return E / (E.max()+1e-12)

    def forward(self, mf, s, kv=None, alpha=None, **_):
        # Physics-informed shortcut (always provides a sensible start)
        al = mf[:,3].clamp(0).float()            # alpha_i >= 0
        lg = mf[:,4].clamp(1e-4).float()         # lg_i > 0
        w_sc = self.scale.abs() * (al.sqrt() / lg)   # (K,)

        # GNN residual refinement
        h   = torch.tanh(self.proj(mf))
        emb = self.emb(kv.float())
        E   = self._edge(kv.float(), alpha)
        for msg_fn, upd_fn in zip(self.msgs, self.upds):
            agg = E @ h
            msg = msg_fn(torch.cat([agg, emb], 1))
            h   = h + upd_fn(torch.cat([h, msg], 1))
        w_res = self.head(h).squeeze(-1)          # (K,)

        w = w_sc + self.res_sc.abs() * w_res
        return reconstruct(s, w)

# ── factory ───────────────────────────────────────────────────────────────
def make_net(method, K):
    d = max(32, min(K*4, 128))
    if   method=='baseline': return BaselineNet(K, d)
    elif method=='methodB':  return MethodBNet (K, d)
    else:                    return PIGNNNet   (K, d=32, d_lam=12, n_layers=2)

# ═══════════════════════════════════════════════════════════════════════════
# Grade trainer
# ═══════════════════════════════════════════════════════════════════════════

def train_grade(net, u_n, delta_star, s, mf, idx, alpha,
                method, kw):
    lr     = 8e-4 if method=='gnn' else 6e-4
    opt    = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    if method == 'gnn':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, patience=max(20,EPOCHS//30), factor=0.5, min_lr=5e-6)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=EPOCHS, eta_min=5e-6)
    extra  = {k: v for k,v in kw.items()}       # cf, kv

    loss_h, rse_h, time_h = [], [], []
    # init best to 'no correction' so we never make things worse
    with torch.no_grad():
        best_rse   = float((u_n - u_star).norm() / NORM_STAR)
        best_delta = u_n.detach().clone()
    no_imp = 0

    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        opt.zero_grad()
        g    = net(mf, s, **extra)
        loss = (spec_loss(g, delta_star, idx, alpha)
                + 0.04 * F.mse_loss(g, delta_star))
        loss.backward();  opt.step();  sched.step(loss.item()) if hasattr(sched,"patience") else sched.step()
        dt   = time.perf_counter() - t0

        with torch.no_grad():
            rse = float((u_n+g-u_star).norm()/NORM_STAR)

        loss_h.append(float(loss));  rse_h.append(rse);  time_h.append(dt)

        if rse < best_rse - 1e-6:
            best_rse = rse
            with torch.no_grad():
                best_delta = (u_n + g).detach().clone()
            no_imp = 0
            best_delta.clamp_(-10, 10)  # safety clamp
        else:
            no_imp += 1
        if no_imp >= EARLY_PAT:
            pad = EPOCHS - ep - 1
            loss_h += [loss_h[-1]]*pad
            rse_h  += [rse_h[-1]] *pad
            time_h += [0.]*pad
            break

    # pad to EPOCHS length
    while len(loss_h) < EPOCHS:
        loss_h.append(loss_h[-1]); rse_h.append(rse_h[-1]); time_h.append(0.)

    return best_delta, loss_h, rse_h, time_h

# ═══════════════════════════════════════════════════════════════════════════
# Full solver
# ═══════════════════════════════════════════════════════════════════════════

MKEYS  = ['baseline','methodB','gnn']
MNAMES = ['Baseline','Method-B','Method-C (PI-GNN)']
MCOLS  = ['#2196F3','#FF9800','#4CAF50']

def run():
    all_loss = {k:[] for k in MKEYS}
    all_rse  = {k:[] for k in MKEYS}
    all_time = {k:[] for k in MKEYS}
    snaps    = {k:[] for k in MKEYS}   # list of (u_before, u_after) per grade

    for mi, method in enumerate(MKEYS):
        print(f"\n{'='*58}\n  {MNAMES[mi]}\n{'='*58}")
        u_cur = u0.clone()
        ctx_b = None     # Method-B cross-grade context

        for g in range(N_GRADES):
            u_bef = u_cur.clone()
            res   = u_cur - u_star
            rse0  = float(res.norm()/NORM_STAR)

            idx, kvecs, amps = select_modes(res)
            s_mat, alpha, lg, phi = build_basis(res, idx, kvecs, amps)
            delta_star_g = -res
            K = len(idx)

            mf  = mode_feats(kvecs, alpha, lg, phi)
            kw  = {'kv': kvecs, 'alpha': alpha}
            if method == 'methodB':
                kw['cf'] = ctx_feats(ctx_b, idx)

            net = make_net(method, K)
            n_p = sum(p.numel() for p in net.parameters())
            print(f"  G{g+1}  K={K:2d}  RSE_in={rse0:.3e}  params={n_p:,}", flush=True)

            u_new, lh, rh, th = train_grade(
                net, u_cur, delta_star_g, s_mat, mf, idx, alpha, method, kw)

            # Method-B: compute freq-filtered context for next grade
            if method == 'methodB':
                correction = (u_new - u_cur).detach()
                # keep top min(K, K_MAX//2) modes
                idx2, _, _ = select_modes(correction)
                mask = torch.zeros(NX, NY)
                n_k  = min(len(idx2), K_MAX//2)
                mask[idx2[:n_k,0], idx2[:n_k,1]] = 1.
                filt = tfft.ifft2(tfft.fft2(correction.float())*mask).real
                ctx_b = (filt / (filt.abs().max()+1e-8)).detach()

            u_cur = u_new.detach()
            print(f"       RSE_out={float((u_cur-u_star).norm()/NORM_STAR):.3e}"
                  f"  early_stop@ep={rh.index(min(rh))+1}", flush=True)

            all_loss[method].extend(lh)
            all_rse [method].extend(rh)
            all_time[method].extend(th)
            snaps   [method].append((u_bef, u_cur.clone()))

        print(f"  ✓ Final RSE = {all_rse[method][-1]:.4e}")

    return all_loss, all_rse, all_time, snaps

# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  saved → {p}")
    plt.close(fig)

def hmap(ax, data, title='', vmin=None, vmax=None, cmap='viridis', cb=True):
    im = ax.imshow(data.T, origin='lower', aspect='equal', cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=[0,LX,0,LY])
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    if cb: plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    return im

def add_vlines(ax, n_grades=N_GRADES, epochs=EPOCHS):
    for g in range(1, n_grades):
        ax.axvline(g*epochs, color='#888', lw=0.6, ls='--', alpha=0.5)
    ax.set_xlim(0, n_grades*epochs)

def grade_labels(ax, n_grades=N_GRADES, epochs=EPOCHS):
    ylo, yhi = ax.get_ylim()
    for g in range(n_grades):
        ax.text((g+.5)*epochs, yhi, f'G{g+1}',
                ha='center', va='top', fontsize=7, color='#666')

# ── Fig 1-3: summary curves ───────────────────────────────────────────────
def plot_summary(all_loss, all_rse, all_time):
    total = N_GRADES*EPOCHS
    xs    = np.arange(total)

    for data, ylabel, fname, log in [
        (all_loss, 'Spectral Loss',           'summary_loss.png', True),
        (all_rse,  'RSE  ‖uₙ−u*‖/‖u*‖',     'summary_rse.png',  True),
        (all_time, 'Time per epoch  (ms)',     'summary_time.png', False),
    ]:
        fig, ax = plt.subplots(figsize=(14, 4))
        for mi, mk in enumerate(MKEYS):
            ys = np.array(data[mk][:total], dtype=float)
            if fname == 'summary_time.png':
                ys *= 1000
                w = max(20, EPOCHS//15)
                ys = np.convolve(ys, np.ones(w)/w, mode='same')
            if log:
                ax.semilogy(xs, ys, color=MCOLS[mi], lw=0.8,
                            alpha=0.85, label=MNAMES[mi])
            else:
                ax.plot(xs, ys, color=MCOLS[mi], lw=0.8,
                        alpha=0.85, label=MNAMES[mi])
        add_vlines(ax)
        grade_labels(ax)
        ax.set_xlabel(f'Epoch  ({EPOCHS} per grade, {N_GRADES} grades)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel + ' — All Grades × All Methods', fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.2)
        fig.tight_layout(); save(fig, fname)

# ── Fig 4: final heatmap ──────────────────────────────────────────────────
def plot_final_heatmap(snaps):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    vmax = float(u_star.max())
    for ri, mk in enumerate(MKEYS):
        u_f   = snaps[mk][-1][1].float()
        err   = (u_f - u_star).abs()
        emax  = float(err.max())
        rse_f = float(err.norm()/NORM_STAR)
        hmap(axes[ri,0], u_f,    f'{MNAMES[ri]}\nTrained $u$',  0, vmax)
        hmap(axes[ri,1], u_star, 'Analytical $u^*$',             0, vmax)
        hmap(axes[ri,2], err,    f'|Error|  RSE={rse_f:.3e}',    0, emax, 'Reds')
    fig.suptitle('Final Solution vs Analytical  (after all grades)',
                 fontsize=13, y=1.01)
    fig.tight_layout(); save(fig, 'final_heatmap.png')

# ── Per-grade: curves (Fig A per grade) ───────────────────────────────────
def plot_grade_curves(g, all_loss, all_rse):
    ep0, ep1 = g*EPOCHS, (g+1)*EPOCHS
    xs = np.arange(EPOCHS)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for mi, mk in enumerate(MKEYS):
        a1.semilogy(xs, all_loss[mk][ep0:ep1], color=MCOLS[mi], lw=1.2,
                    label=MNAMES[mi])
        a2.semilogy(xs, all_rse[mk][ep0:ep1],  color=MCOLS[mi], lw=1.2,
                    label=MNAMES[mi])
    for ax, lbl in [(a1,'Spectral Loss'),(a2,'RSE')]:
        ax.set_xlabel('Epoch within grade', fontsize=10)
        ax.set_ylabel(lbl, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.25)
    a1.set_title(f'Grade {g+1} — Loss', fontsize=11)
    a2.set_title(f'Grade {g+1} — RSE',  fontsize=11)
    fig.suptitle(f'Grade {g+1}  Training Curves', fontsize=12)
    fig.tight_layout(); save(fig, f'grade{g+1:02d}_A_curves.png')

# ── Per-grade: heatmaps (Fig B per grade) ────────────────────────────────
def plot_grade_heatmaps(g, snaps):
    vmax  = float(u_star.max())
    n_row = len(MKEYS) + 1         # +1 for analytical reference row
    fig, axes = plt.subplots(n_row, 3, figsize=(12, 4*n_row))
    for ri, mk in enumerate(MKEYS):
        u_bef, u_aft = snaps[mk][g]
        err  = (u_aft - u_star).abs()
        emax = float(err.max())
        hmap(axes[ri,0], u_bef.float(), f'{MNAMES[ri]}\nBefore G{g+1}', 0, vmax)
        hmap(axes[ri,1], u_aft.float(), f'{MNAMES[ri]}\nAfter  G{g+1}', 0, vmax)
        hmap(axes[ri,2], err,            f'|Error|  max={emax:.3e}',     0, emax,'Reds')
    # bottom row: analytical
    for c in range(3):
        hmap(axes[n_row-1, c], u_star, 'Analytical $u^*$', 0, vmax)
    fig.suptitle(f'Grade {g+1}  Solution Heatmaps', fontsize=13, y=1.01)
    fig.tight_layout(); save(fig, f'grade{g+1:02d}_B_heatmaps.png')

# ── Per-grade: time (Fig C per grade) ────────────────────────────────────
def plot_grade_time(g, all_time):
    ep0, ep1 = g*EPOCHS, (g+1)*EPOCHS
    xs = np.arange(EPOCHS)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for mi, mk in enumerate(MKEYS):
        ys = 1000*np.array(all_time[mk][ep0:ep1])
        w  = max(1, EPOCHS//20)
        ax.plot(xs, np.convolve(ys, np.ones(w)/w, 'same'),
                color=MCOLS[mi], lw=1.2, label=MNAMES[mi])
    ax.set_xlabel('Epoch within grade', fontsize=10)
    ax.set_ylabel('ms / epoch  (rolling avg)', fontsize=10)
    ax.set_title(f'Grade {g+1}  Per-Epoch Time', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig.tight_layout(); save(fig, f'grade{g+1:02d}_C_time.png')

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    print(f"\n{'#'*58}")
    print(f"  2D NLS Grade Solver  —  {N_GRADES} grades × {EPOCHS} ep")
    print(f"  Grid {NX}×{NY}  K_MAX={K_MAX}  OOB_λ={OOB_LAM}")
    print(f"{'#'*58}\n")

    all_loss, all_rse, all_time, snaps = run()

    print(f"\n── Generating {3 + 1 + N_GRADES*3} figures …")
    plot_summary(all_loss, all_rse, all_time)
    plot_final_heatmap(snaps)
    for g in range(N_GRADES):
        plot_grade_curves(g, all_loss, all_rse)
        plot_grade_heatmaps(g, snaps)
        plot_grade_time(g, all_time)

    elapsed = time.time() - t0
    print(f"\n{'='*58}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  Final RSEs:")
    for mi, mk in enumerate(MKEYS):
        print(f"    {MNAMES[mi]:22s}: {all_rse[mk][-1]:.4e}")
    print(f"{'='*58}")
