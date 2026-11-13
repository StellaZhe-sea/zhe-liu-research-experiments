"""
GRBD Training & Ablation
=========================
Trains Fourier-GRBD vs Wavelet-GRBD on 2D Kolmogorov Flow.

Task: given a coarse/noisy initial field u0, predict the true vorticity field u*.
      (We use: u0 = low-pass filtered version of u*, so u* is the reconstruction target)

Usage:
    python train.py --N 64 --G 6 --epochs 500 --basis fourier wavelet
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from basis import FourierBasis, WaveletBasis
from grbd  import GradeNet, graded_loss, rse
from ns_data import KolmogorovFlowGenerator


# ── Dataset helpers ──────────────────────────────────────────────────────

def make_coarse_input(data: torch.Tensor, cutoff: float = 0.25) -> torch.Tensor:
    """
    Create low-frequency (coarse) input by zeroing high-frequency modes.
    data   : (B, N, N)
    cutoff : fraction of modes to keep (e.g. 0.25 = keep lowest 25%)
    """
    N = data.shape[-1]
    C = torch.fft.rfft2(data)
    kmax = int(N * cutoff / 2)
    mask = torch.zeros_like(C.real)
    mask[:, :kmax, :kmax] = 1.0
    C_low = C * mask
    return torch.fft.irfft2(C_low, s=(N, N))


def split_data(data, train_frac=0.8):
    n = data.shape[0]
    n_train = int(n * train_frac)
    return data[:n_train], data[n_train:]


# ── Training loop ────────────────────────────────────────────────────────

def train_one_model(basis_name: str, data_train, data_val,
                    G: int, hidden_dim: int, epochs: int,
                    lr: float, batch_size: int, device: str,
                    N: int, K_fourier: int, J_wavelet: int):

    print(f"\n{'='*55}")
    print(f"  Training  basis={basis_name.upper()}  G={G}  epochs={epochs}")
    print(f"{'='*55}")

    # Build basis
    if basis_name == 'fourier':
        basis = FourierBasis(N=N, K=K_fourier)
    elif basis_name == 'wavelet':
        basis = WaveletBasis(N=N, J=J_wavelet, wavelet='db4')
    else:
        raise ValueError(f"Unknown basis: {basis_name}")

    print(f"  Basis atoms: {basis.n_atoms()},  feat_dim: {basis.feat_dim}")

    # Build model
    model = GradeNet(basis=basis, G=G, hidden_dim=hidden_dim,
                     share_weights=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Geometric grade weights: later grades count more
    grade_w = [2**i for i in range(G)]
    grade_w_sum = sum(grade_w)
    grade_w = [w / grade_w_sum for w in grade_w]

    # Data loaders
    u0_train = make_coarse_input(data_train).to(device)
    u0_val   = make_coarse_input(data_val).to(device)
    u_train  = data_train.to(device)
    u_val    = data_val.to(device)

    ds_train = TensorDataset(u0_train, u_train)
    loader   = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    # History
    train_losses, val_rses = [], []
    best_val_rse = float('inf')
    best_state   = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for u0_b, u_b in loader:
            optimizer.zero_grad()
            _, u_hist, _ = model(u0_b, u_b)
            loss = graded_loss(u_hist, u_b, weights=grade_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        ep_loss /= len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            u_pred, _, _ = model(u0_val, u_val)
            val_r = rse(u_pred, u_val)

        train_losses.append(ep_loss)
        val_rses.append(val_r)

        if val_r < best_val_rse:
            best_val_rse = val_r
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % max(1, epochs//10) == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"  ep {ep:4d}/{epochs} | loss={ep_loss:.6f} | "
                  f"val_RSE={val_r:.6f} | best={best_val_rse:.6f} | "
                  f"t={elapsed:.1f}s")

    print(f"\n  ✓ Best val RSE ({basis_name.upper()}): {best_val_rse:.6f}")
    return {
        'basis': basis_name,
        'best_val_rse': best_val_rse,
        'train_losses': train_losses,
        'val_rses': val_rses,
        'best_state': best_state,
        'model': model,
        'basis_obj': basis,
    }


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_results(results_list, save_dir: str, data_val, device):
    os.makedirs(save_dir, exist_ok=True)

    # ① Convergence curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'fourier': '#2563eb', 'wavelet': '#dc2626'}
    for res in results_list:
        name = res['basis']
        c    = colors.get(name, 'gray')
        axes[0].semilogy(res['train_losses'], label=f"{name.upper()} train loss", color=c)
        axes[1].semilogy(res['val_rses'],     label=f"{name.upper()} val RSE",   color=c)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[0].set_title("Training Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RSE"); axes[1].legend()
    axes[1].set_title("Validation RSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence.png'), dpi=150)
    plt.close()
    print(f"  [plot] convergence.png saved")

    # ② Field comparison (first validation sample)
    u0_vis  = make_coarse_input(data_val[:1]).to(device)
    u_true  = data_val[:1].to(device)

    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(5*(n_models+2), 4))

    vmin = u_true[0].cpu().min().item()
    vmax = u_true[0].cpu().max().item()

    axes[0].imshow(u0_vis[0].cpu(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title("Input (coarse)")
    axes[0].axis('off')

    axes[1].imshow(u_true[0].cpu(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    for i, res in enumerate(results_list):
        model = res['model']
        model.load_state_dict({k: v.to(device) for k, v in res['best_state'].items()})
        model.eval()
        with torch.no_grad():
            u_pred, _, _ = model(u0_vis, u_true)
        r = rse(u_pred, u_true)
        axes[i+2].imshow(u_pred[0].cpu(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i+2].set_title(f"{res['basis'].upper()}\nRSE={r:.4f}")
        axes[i+2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'field_comparison.png'), dpi=150)
    plt.close()
    print(f"  [plot] field_comparison.png saved")

    # ③ Per-grade residual energy plot (for the best model)
    best_res = min(results_list, key=lambda x: x['best_val_rse'])
    model = best_res['model']
    model.load_state_dict({k: v.to(device) for k, v in best_res['best_state'].items()})
    model.eval()

    with torch.no_grad():
        _, u_hist, _ = model(u0_vis, u_true)

    energies = []
    for u_k in u_hist:
        err = ((u_k - u_true)**2).mean().item()
        energies.append(err)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(energies)+1), energies, 'o-',
                color=colors.get(best_res['basis'], 'purple'))
    ax.set_xlabel("Grade k"); ax.set_ylabel("MSE(uₖ, u*)")
    ax.set_title(f"Per-Grade Residual Decay ({best_res['basis'].upper()})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grade_decay.png'), dpi=150)
    plt.close()
    print(f"  [plot] grade_decay.png saved")


# ── Summary table ─────────────────────────────────────────────────────────

def print_summary(results_list):
    print("\n" + "="*55)
    print("  ABLATION SUMMARY")
    print("="*55)
    print(f"  {'Basis':<12} {'Best Val RSE':>14} {'#Atoms':>8}")
    print(f"  {'-'*38}")
    for res in sorted(results_list, key=lambda x: x['best_val_rse']):
        print(f"  {res['basis'].upper():<12} {res['best_val_rse']:>14.6f} "
              f"{res['basis_obj'].n_atoms():>8}")
    print("="*55)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',          type=int,   default=64)
    parser.add_argument('--G',          type=int,   default=6,
                        help='Number of grades')
    parser.add_argument('--K',          type=int,   default=64,
                        help='Fourier: top-K modes')
    parser.add_argument('--J',          type=int,   default=4,
                        help='Wavelet: decomposition levels')
    parser.add_argument('--hidden',     type=int,   default=128)
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch',      type=int,   default=16)
    parser.add_argument('--n_samples',  type=int,   default=200,
                        help='Total dataset size')
    parser.add_argument('--Re',         type=float, default=1000.0)
    parser.add_argument('--t_start',    type=float, default=8.0,
                        help='Burn-in time for N-S solver')
    parser.add_argument('--t_gap',      type=float, default=0.5)
    parser.add_argument('--basis',      nargs='+',  default=['fourier', 'wavelet'])
    parser.add_argument('--save_dir',   type=str,   default='./results')
    parser.add_argument('--data_cache', type=str,   default='./ns_data_cache.pt')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"\n[GRBD] Device: {args.device}")
    print(f"[GRBD] Bases to compare: {args.basis}")

    # ── Data ──────────────────────────────────────────────────────────────
    if os.path.exists(args.data_cache):
        print(f"[Data] Loading cached data from {args.data_cache}")
        data = torch.load(args.data_cache)
    else:
        print(f"[Data] Generating Kolmogorov Flow data (N={args.N}, Re={args.Re}) ...")
        gen = KolmogorovFlowGenerator(
            N=args.N, Re=args.Re, T=args.t_start + args.n_samples*args.t_gap,
            dt=1e-3, k_f=4,
            device=args.device if args.device == 'cpu' else 'cpu'
        )
        data = gen.generate(n_samples=args.n_samples,
                            t_start=args.t_start,
                            t_gap=args.t_gap, verbose=True)
        torch.save(data, args.data_cache)
        print(f"[Data] Cached to {args.data_cache}")

    data_train, data_val = split_data(data, train_frac=0.8)
    print(f"[Data] Train: {data_train.shape}, Val: {data_val.shape}")

    # ── Training ──────────────────────────────────────────────────────────
    results_list = []
    for bname in args.basis:
        res = train_one_model(
            basis_name  = bname,
            data_train  = data_train,
            data_val    = data_val,
            G           = args.G,
            hidden_dim  = args.hidden,
            epochs      = args.epochs,
            lr          = args.lr,
            batch_size  = args.batch,
            device      = args.device,
            N           = args.N,
            K_fourier   = args.K,
            J_wavelet   = args.J,
        )
        results_list.append(res)

    # ── Results ───────────────────────────────────────────────────────────
    print_summary(results_list)
    plot_results(results_list, args.save_dir, data_val, args.device)
    print(f"\n[GRBD] Plots saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
