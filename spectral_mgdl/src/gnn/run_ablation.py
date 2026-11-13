#!/usr/bin/env python3
"""(see code)"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.fft import dst, idst
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from spectral_gnn_module import build_spectral_gnn, M_DENSE_THRESH

OUT = Path('./ablation_output'); OUT.mkdir(exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HelmholtzOp:
    def __init__(self, N, kappa):
        self.N = N; self.kappa = kappa
        self.x = np.arange(1, N+1) / (N+1)
        k = np.arange(1, N+1)
        self.k_all = k; self.lam_k = (np.pi*k)**2 - kappa**2
        self._safe = np.where(np.abs(self.lam_k)>1e-8, self.lam_k,
                              np.sign(self.lam_k+1e-12)*1e-8)

    def make_f(self, u):
        return idst(self.lam_k * dst(u, type=1), type=1).astype(np.float32)

    def residual(self, u, f):
        return idst(self.lam_k*dst(u,type=1).astype(np.float32)
                    - dst(f,type=1).astype(np.float32), type=1).astype(np.float32)

    def inv_L(self, r):
        return idst(dst(r,type=1).astype(np.float32)/self._safe, type=1).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gd_threshold(amps, floor=0.10):
    s = np.sort(amps)[::-1]
    mad = np.median(np.abs(s - np.median(s)))
    sigma = mad / 0.6745 + 1e-12
    n = len(amps); beta = 1/n
    omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
    return float(max(omega*sigma, floor*s[0]))


def build_spectral_input(e, target, op, gd_floor=0.10):
    N = op.N; k_all = np.arange(1, N+1)
    e_hat = dst(e, type=1).astype(np.float32); amps = np.abs(e_hat)
    tau = gd_threshold(amps, gd_floor); mask = amps > tau
    if not mask.any(): mask[np.argmax(amps)] = True
    k_sel = k_all[mask]; t_hat = dst(target, type=1).astype(np.float32)
    a_sel = np.abs(t_hat[mask]); signs = np.sign(t_hat[mask])
    energy = a_sel**2; alphas = energy / (energy.sum() + 1e-12)
    H_ent = float(-np.sum(alphas * np.log(alphas + 1e-12)))
    log_norm = np.log(1+a_sel) / (np.log(1+a_sel.max()+1e-12) + 1e-12)
    j_ = np.arange(1, N+1)[:, None]
    s = (log_norm[None,:] * signs[None,:] * np.sin(np.pi*k_sel[None,:]*j_/(N+1))).astype(np.float32)
    return s, dict(k_sel=k_sel, mask=mask, tau=tau, k_all=k_all,
                   n_freqs=int(mask.sum()), entropy=H_ent,
                   alphas=alphas, amps_res=amps, e_hat=e_hat,
                   amps_delta=np.abs(t_hat), target_hat=t_hat,
                   signs=signs, log_norm=log_norm, target=target, s_input=s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v8 baseline GradeNet (W_sym)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GradeNet(nn.Module):
    def __init__(self, idim, width, depth):
        super().__init__()
        self._W = nn.Parameter(torch.zeros(idim, idim))
        self.act = nn.Tanh()
        self.proj = nn.Linear(idim, width)
        self.mlp = nn.Sequential(*[
            l for _ in range(depth) for l in (nn.Linear(width,width), nn.Tanh())])
        self.out = nn.Linear(width, 1)
        nn.init.normal_(self._W, std=0.01)
        nn.init.xavier_normal_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        for m in list(self.mlp.modules()) + [self.out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5); nn.init.zeros_(m.bias)

    @property
    def W_sym(self): return 0.5 * (self._W + self._W.t())

    def forward(self, s):
        return self.out(self.mlp(self.proj(self.act(s @ self.W_sym.t()) + s)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SpectralLoss(nn.Module):
    def __init__(self, N, mask, alphas, lambda_oob=0.1):
        super().__init__()
        self.register_buffer('mask_in',  torch.tensor(mask,  dtype=torch.bool))
        self.register_buffer('mask_out', torch.tensor(~mask, dtype=torch.bool))
        self.register_buffer('alphas',   torch.tensor(alphas, dtype=torch.float32))
        self.lambda_oob = lambda_oob; self.norm = 2.0/(N+1)

    def dst1(self, x):
        try: return torch.fft.dst(x, type=1, dim=-1)
        except:
            N = x.shape[-1]
            k = torch.arange(1,N+1, dtype=x.dtype, device=x.device)
            M = torch.sin(torch.pi * k[:,None] * k[None,:] / (N+1))
            return x @ M.T

    def forward(self, g, target):
        gh = self.dst1(g)*self.norm; th = self.dst1(target)*self.norm
        d2 = (gh - th)**2
        li = (self.alphas * d2[self.mask_in]).sum()
        lo = self.lambda_oob * (gh[self.mask_out]**2).mean()
        return li + lo, li.item(), lo.item()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class LrAdaptor:
    def __init__(self, base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                 tail_frac=0.20, decay_bad=0.5, decay_mild=0.7, boost_good=1.2):
        self.base_lr=base_lr; self.lr_min=lr_min; self.lr_max=lr_max
        self.tail_frac=tail_frac; self.decay_bad=decay_bad
        self.decay_mild=decay_mild; self.boost_good=boost_good
        self._lr = base_lr

    def update(self, losses, n):
        # n is the grade number just COMPLETED; we're setting lr for grade n+1
        if n <= 1 or len(losses) == 0:
            self._lr = self.base_lr; return self._lr
        losses = np.array(losses, dtype=np.float64)
        tail_start = max(0, int(len(losses)*(1-self.tail_frac)))
        tail = losses[tail_start:]
        if len(tail) < 2 or tail[0] == 0:
            self._lr = float(np.clip(self._lr * self.decay_mild, self.lr_min, self.lr_max))
            return self._lr
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0] / (tail[0]+1e-30)
        ratio = tail.mean() / (losses[:max(1,tail_start)].mean() + 1e-30)
        if slope < -1e-10:       pass   # still descending
        elif ratio > 0.5:        self._lr *= self.decay_bad
        elif ratio < 0.01:       self._lr *= self.boost_good
        else:                    self._lr *= self.decay_mild
        self._lr = float(np.clip(self._lr, self.lr_min, self.lr_max))
        return self._lr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def solve(op, f, u_star, use_gnn, edge_type='energy', gnn_layers=2,
          d_lam=16, base_width=64, base_depth=3, gd_floor=0.10,
          lambda_oob=0.1, n_grades=8, n_epochs=2000,
          stop_delta_H=0.05, stop_min_improve=0.02,
          device='cpu', label=''):
    """Solve."""
    dev = torch.device(device)
    lra = LrAdaptor(base_lr=2e-3, lr_min=1e-4, lr_max=5e-3)
    u = np.zeros(op.N, np.float32); next_res = None; infos = []
    print(f'\n{"═"*68}\n  {label}\n{"═"*68}')

    for n in range(1, n_grades+1):
        e = next_res if next_res is not None else op.residual(u, f)
        target = -op.inv_L(e)
        rel_b = np.linalg.norm(u - u_star) / np.linalg.norm(u_star)

        lr = lra.update(infos[-1]['losses'] if infos else [], n-1)

        s_np, meta = build_spectral_input(e, target, op, gd_floor)
        nk = meta['n_freqs']; H_ent = meta['entropy']


        if use_gnn:
            net, width, depth = build_spectral_gnn(
                meta, rel_b, op,
                base_width=base_width, base_depth=base_depth,
                edge_type=edge_type, gnn_layers=gnn_layers, d_lam=d_lam)
            mode = f'GNN({edge_type},L={gnn_layers},{"dense" if nk<=M_DENSE_THRESH else "mfield"})'
        else:
            depth = max(2, min(base_depth + int(H_ent*1.5), 8))
            wc = 128 if rel_b<1e-3 else (256 if rel_b<1e-2 else 512)
            width = min(max(base_width, 2*nk), wc)
            net = GradeNet(nk, width, depth)
            mode = 'GradeNet(W_sym)'

        n_params = sum(p.numel() for p in net.parameters())
        net = net.to(dev)
        s_t   = torch.tensor(s_np).to(dev)
        tgt_t = torch.tensor(target).to(dev)
        crit  = SpectralLoss(op.N, meta['mask'], meta['alphas'], lambda_oob).to(dev)
        opt   = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.005)

        losses = []; losses_in = []; losses_oob = []
        t0 = time.time(); net.train()
        for ep in range(n_epochs):
            opt.zero_grad()
            g = net(s_t).squeeze(1)
            loss, l_in, l_oob = crit(g, tgt_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step(); sched.step()
            losses.append(loss.item()); losses_in.append(l_in); losses_oob.append(l_oob)
            if (ep+1) % 500 == 0:
                print(f'    ep{ep+1:5d}  loss={loss.item():.4e}  in={l_in:.4e}  oob={l_oob:.4e}')
        t1 = time.time()

        net.eval()
        with torch.no_grad(): g_np = net(s_t).squeeze(1).cpu().numpy()
        u = u + g_np
        rel_a = np.linalg.norm(u - u_star) / np.linalg.norm(u_star)
        net.requires_grad_(False)
        next_res = op.residual(u, f)

        improve = (rel_b - rel_a) / (rel_b + 1e-30)
        arrow = '↓' if rel_a < rel_b else '↑'
        print(f'  G{n:02d}  |K|={nk:3d}  [{mode}]  '
              f'w={width}  d={depth}  lr={lr:.1e}  #p={n_params:,}')
        print(f'        err: {rel_b:.4e} {arrow} {rel_a:.4e}  '
              f'({improve*100:+.1f}%)  t={t1-t0:.1f}s')

        infos.append(dict(
            n=n, nk=nk, entropy=H_ent, width=width, depth=depth,
            lr=lr, n_params=n_params, t=t1-t0,
            eb=rel_b, ea=rel_a, improve=improve,
            losses=losses, losses_in=losses_in, losses_oob=losses_oob,
            g_np=g_np, mode=mode,
        ))


        if len(infos) >= 2:
            dH = abs(infos[-1]['entropy'] - infos[-2]['entropy'])
            cA = dH < stop_delta_H
            cB = improve < stop_min_improve
            cC = (len(infos) >= 2 and improve < 0
                  and infos[-2]['improve'] < 0)
            if cA or cB or cC:
                reasons = ([f'ΔH={dH:.3f}'] if cA else []) + \
                          ([f'improve={improve*100:.1f}%'] if cB else []) + \
                          (['diverge×2'] if cC else [])
                print(f'  [EARLY STOP] {" | ".join(reasons)}')
                break

    return infos


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_comparison(all_results, n_epochs):
    labels = list(all_results.keys())
    print(f'\n{"═"*90}')
    print(f'  ABLATION TABLE  |  N=256 κ=200  seed=42  n_ep={n_epochs}')
    print(f'{"═"*90}')
    print(f'  {"Variant":<32}  {"Grades":^6}  {"Final err":^12}  {"vs baseline":^12}  {"Total t":^8}')
    print(f'  {"-"*78}')
    baseline_err = all_results['baseline'][-1]['ea'] if 'baseline' in all_results else None
    for label, infos in all_results.items():
        final_err = infos[-1]['ea']
        n_grades  = len(infos)
        total_t   = sum(i['t'] for i in infos)
        if baseline_err and label != 'baseline':
            ratio = final_err / baseline_err
            vs = f'{ratio:.3f}x  {"✓" if ratio<1 else "✗"}'
        else:
            vs = '  (ref)  '
        print(f'  {label:<32}  {n_grades:^6}  {final_err:^12.4e}  {vs:^12}  {total_t:^8.1f}s')
    print(f'{"═"*90}')


    print(f'\n  GRADE-BY-GRADE  (rel L2 error after each grade)')
    all_labels = list(all_results.keys())
    header = f'  {"G":^3}  ' + '  '.join(f'{l[:18]:^18}' for l in all_labels)
    print(header); print('  ' + '-'*(len(header)-2))
    max_g = max(len(v) for v in all_results.values())
    for g in range(1, max_g+1):
        row = f'  {g:^3}  '
        for label in all_labels:
            infos = all_results[label]
            if g <= len(infos): row += f'  {infos[g-1]["ea"]:^18.4e}'
            else:                row += f'  {"—":^18}'
        print(row)
    print()


def plot_comparison(all_results, n_epochs, op, u_star, u_preds):
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0d0d0d')
    for ax in axes: ax.set_facecolor('#181818')
    fig.suptitle(f'Ablation: 1D Helmholtz  N={op.N} κ={op.kappa}  n_ep={n_epochs}',
                 color='white', fontsize=13)


    ax = axes[0]
    for (label, infos), c in zip(all_results.items(), colors):
        ea = [i['ea'] for i in infos]
        ax.semilogy(range(1, len(ea)+1), ea, 'o-', color=c, lw=2, ms=7, label=label)
    ax.set_xlabel('Grade', color='#bbb'); ax.set_ylabel('Rel L2 Error', color='#bbb')
    ax.set_title('Convergence per Grade', color='white')
    ax.legend(fontsize=8); ax.tick_params(colors='#888')


    ax = axes[1]
    for (label, infos), c in zip(all_results.items(), colors):
        ax.semilogy(infos[-1]['losses'], color=c, lw=1.2, alpha=0.9, label=label)
    ax.set_xlabel('Epoch', color='#bbb'); ax.set_ylabel('SpectralLoss', color='#bbb')
    ax.set_title('Training Loss (last grade)', color='white')
    ax.legend(fontsize=8); ax.tick_params(colors='#888')


    ax = axes[2]; x = op.x
    ax.plot(x, u_star, color='white', lw=2.5, label='$u^*$', zorder=3)
    for (label, _), c, u_p in zip(all_results.items(), colors, u_preds.values()):
        ax.plot(x, u_p, lw=1.5, color=c, ls='--', alpha=0.85, label=label)
    ax.set_xlabel('x', color='#bbb'); ax.set_title('Final Solution', color='white')
    ax.legend(fontsize=7); ax.tick_params(colors='#888')

    plt.tight_layout()
    fname = OUT / f'ablation_comparison_ep{n_epochs}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'\n  [saved] {fname}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ep',     type=int, default=2000, help='Epochs per grade')
    parser.add_argument('--device', type=str, default='auto', help='cuda/cpu/auto')
    parser.add_argument('--fast',   action='store_true', help='Only run baseline + energy (quick check)')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    print(f'[*] device={device}  n_ep={args.ep}')


    N = 256; kappa = 200.0
    op = HelmholtzOp(N, kappa); x = op.x
    u_star = (np.sin(3*np.pi*x) + 0.5*np.sin(7*np.pi*x)
              + 0.3*np.sin(11*np.pi*x)).astype(np.float32)
    f = op.make_f(u_star)
    chk = np.linalg.norm(-op.inv_L(op.residual(np.zeros(N,np.float32),f))-u_star)/np.linalg.norm(u_star)
    print(f'[check] inv_L err={chk:.2e}  (should be <1e-5)')

    COMMON = dict(op=op, f=f, u_star=u_star,
                  gd_floor=0.10, lambda_oob=0.1, n_grades=8,
                  n_epochs=args.ep, stop_delta_H=0.05, stop_min_improve=0.02,
                  base_width=64, base_depth=3, device=device)


    variants = [
        ('baseline',            dict(use_gnn=False)),
        ('GNN-energy (L=2)',    dict(use_gnn=True, edge_type='energy',    gnn_layers=2, d_lam=16)),
    ]
    if not args.fast:
        variants += [
            ('GNN-distance (L=2)',  dict(use_gnn=True, edge_type='distance',  gnn_layers=2, d_lam=16)),
            ('GNN-mlp (L=2)',       dict(use_gnn=True, edge_type='mlp',       gnn_layers=2, d_lam=16)),
            ('GNN-selection (L=2)', dict(use_gnn=True, edge_type='selection', gnn_layers=2, d_lam=16)),
            ('GNN-cascade (L=2)',   dict(use_gnn=True, edge_type='cascade',   gnn_layers=2, d_lam=16)),
            ('GNN-energy (L=0)',    dict(use_gnn=True, edge_type='energy',    gnn_layers=0, d_lam=16)),
        ]

    all_results = {}; u_preds = {}
    for label, variant_kw in variants:
        torch.manual_seed(42); np.random.seed(42)
        infos = solve(label=label, **COMMON, **variant_kw)
        all_results[label] = infos

        u_pred = np.zeros(N, np.float32)
        for info in infos:
            u_pred += info['g_np']
        u_preds[label] = u_pred

    print_comparison(all_results, args.ep)
    plot_comparison(all_results, args.ep, op, u_star, u_preds)


    import json
    summary = {}
    for label, infos in all_results.items():
        summary[label] = [dict(n=i['n'], nk=i['nk'], eb=i['eb'], ea=i['ea'],
                               improve=i['improve'], t=i['t'], n_params=i['n_params'])
                          for i in infos]
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (float, int)): return o
            try: return o.item()
            except: return super().default(o)
    with open(OUT/'results.json', 'w') as fh:
        json.dump(summary, fh, indent=2, cls=NpEnc)
    print(f'  [saved] {OUT}/results.json')


if __name__ == '__main__':
    main()
