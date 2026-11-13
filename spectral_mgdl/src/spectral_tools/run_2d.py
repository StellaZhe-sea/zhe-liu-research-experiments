"""run_2d.py - direct entry point.

Depends on mgdl_2d_v1.py and mgdl_2d_fft_cheb.py in the same
directory. Usage: python run_2d.py --op fftcheb ..."""

import argparse, numpy as np
from mgdl_2d_fft_cheb import (FFTChebOperator2D, FFTOperator2D,
                                GradeSolverHybrid)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',   default='fftcheb',
                   choices=['fftcheb','fft2d'], help='operator type')
    p.add_argument('--Nx',     type=int, default=32)
    p.add_argument('--Ny',     type=int, default=16)
    p.add_argument('--alpha',  type=float, default=1.0,
                   help='diffusion coefficient in y (fftcheb only)')
    p.add_argument('--kappa',  type=float, default=1.0,
                   help='reaction coefficient')
    p.add_argument('--grades', type=int, default=8)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--width',  type=int, default=64)
    p.add_argument('--depth',  type=int, default=3)
    p.add_argument('--gd_floor', type=float, default=0.02)
    p.add_argument('--lr',     type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    if args.mode == 'fftcheb':
        op = FFTChebOperator2D(Nx=args.Nx, Ny=args.Ny,
                               alpha=args.alpha, kappa=args.kappa)
        print(f'FFT×Chebyshev  Nx={args.Nx}  Ny={args.Ny}(interior={op.Ny})'
              f'  α={args.alpha}  κ={args.kappa}')
    else:
        op = FFTOperator2D(Nx=args.Nx, Ny=args.Ny, kappa=args.kappa)
        print(f'FFT×FFT (periodic)  Nx={args.Nx}  Ny={args.Ny}  κ={args.kappa}')

    u_star, f = op.make_solution(seed=args.seed)
    print(f'u_star shape={u_star.shape}  L2={np.linalg.norm(u_star):.3e}')

    solver = GradeSolverHybrid(
        op, f, u_star,
        base_width=args.width,
        base_depth=args.depth,
        gd_floor=args.gd_floor,
        base_lr=args.lr,
        device=args.device,
    )
    solver.solve(n_grades=args.grades, n_epochs=args.epochs)

if __name__ == '__main__':
    main()
