"""AS-MGDL v2 for the 1D Helmholtz equation.

Adaptive-structured multi-grade deep learning solver. Each grade
targets a set of active frequencies; a Snake-activated network maps
filtered hidden features to spectral coefficients, and a
Gavish-Donoho style analysis selects the next grade's frequencies
from the residual spectrum."""

import torch
import torch.nn as nn
import numpy as np
import time, gc
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)


class Snake(nn.Module):
    """Snake."""
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = max(float(a), 1e-3)

    def forward(self, x):
        return x + (1.0 / self.a) * torch.sin(self.a * x) ** 2

    def __repr__(self):
        return f'Snake(a={self.a:.3f})'


class SpectralTools:
    """SpectralTools."""
    def __init__(self, N: int, k: float, device):
        self.N = N; self.k = float(k); self.device = device

        j = torch.arange(1, N+1, dtype=torch.float64, device=device)
        m = torch.arange(1, N+1, dtype=torch.float64, device=device)
        self.x_grid  = j / (N+1)                              # (N,)
        self.S       = (2.0/(N+1)) * torch.sin(
                        torch.outer(m, j) * np.pi / (N+1))
        self.omega_m = m * np.pi                               # mπ
        self.D_m     = k**2 - self.omega_m**2


    def dst(self, v: torch.Tensor) -> torch.Tensor:
        """Dst."""
        return self.S @ v.to(torch.float64)

    def dst_cols(self, M: torch.Tensor) -> torch.Tensor:
        """Dst cols."""
        return self.S @ M.to(torch.float64)

    def idst_sparse(self, g_hat: torch.Tensor,
                    K: torch.Tensor) -> torch.Tensor:
        """Idst sparse."""
        full = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        full[K] = g_hat.to(torch.float64)
        return (self.S.T @ full)


    def residual_hat(self, u_pred: torch.Tensor,
                     alpha: float, beta: float,
                     f_grid: torch.Tensor) -> tuple:
        """
        R̂_m = D_m·v̂_m - f̃_m
        v = u - lift,   f̃ = f - k²·lift
        """
        x      = self.x_grid
        lift   = alpha*(1-x) + beta*x
        v      = u_pred.to(torch.float64) - lift
        f_til  = f_grid.to(torch.float64) - self.k**2 * lift
        v_hat  = self.S @ v
        f_hat  = self.S @ f_til
        R_hat  = self.D_m * v_hat - f_hat
        return R_hat, v_hat                                    # (N,), (N,)


    def find_active_modes(self, e_hat: torch.Tensor,
                          beta_gd: float = 1.0) -> tuple:
        """Find active modes."""
        # Step2: G-D
        w   = 0.56*beta_gd**3 - 0.95*beta_gd**2 + 1.82*beta_gd + 1.43
        tau = float(w * torch.median(e_hat.abs()))
        K   = torch.where(e_hat.abs() > tau)[0]               # 0-indexed

        if len(K) == 0:
            return K, tau, None, None, 0.0


        m_active  = (K + 1).to(torch.float64)                 # 1-indexed
        f_angular = m_active * np.pi
        phi_l     = torch.zeros_like(f_angular)
        f_mean    = float(f_angular.mean())

        return K, tau, f_angular, phi_l, f_mean


class Grade1Net(nn.Module):
    """Grade1Net."""
    def __init__(self, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor,
                alpha: float, beta: float) -> tuple:
        """Forward."""
        h    = self.backbone(x)                               # (N, hidden_dim)
        out  = self.out_proj(h)                               # (N, 1)
        lift = alpha*(1-x) + beta*x                           # (N, 1)
        u    = lift + x*(1-x)*out
        return u.squeeze(1), h                                # (N,), (N, d)


class GradeNNet(nn.Module):
    """GradeNNet."""
    def __init__(self, r_n: int, f_mean: float, depth: int = 3):
        super().__init__()
        self.r_n   = r_n
        self.width = 2 * r_n                                  # d_n = 2r_n (Option B)

        act = Snake(a=f_mean)
        layers = [nn.Linear(self.width, self.width), act]
        for _ in range(depth - 1):
            layers += [nn.Linear(self.width, self.width), act]
        layers.append(nn.Linear(self.width, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(list(self.net.modules())[-1].weight)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, Phi_hat_Kn: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.net(Phi_hat_Kn).squeeze(-1)               # (r_n,)


class PhiBuilder:
    """PhiBuilder."""

    def __init__(self, st: SpectralTools):
        self.st = st

    def build_Bn(self, x_grid: torch.Tensor,
                 f_angular: torch.Tensor,
                 phi_l: torch.Tensor) -> tuple:
        """Build Bn."""
        x = x_grid.to(torch.float64).unsqueeze(1)             # (N,1)
        f = f_angular.to(torch.float64).unsqueeze(0)          # (1,r)
        p = phi_l.to(torch.float64).unsqueeze(0)              # (1,r)


        phase  = x * f + p                                     # (N, r)
        B_full = torch.cat([torch.cos(phase),
                             torch.sin(phase)], dim=1)         # (N, 2r)


        try:
            U, S, Vh = torch.linalg.svd(B_full, full_matrices=False)
            energy   = (S**2).cumsum(0) / (S**2).sum()
            r_keep   = int((energy < 0.99).sum()) + 1
            r_keep   = min(r_keep, B_full.shape[1])
            B_n = (U[:, :r_keep] * S[:r_keep]).float()
        except Exception:
            B_n = B_full.float()
            r_keep = B_full.shape[1]

        return B_n, r_keep

    def bandpass_filter_hidden(self, h: torch.Tensor,
                               K_active: torch.Tensor) -> torch.Tensor:
        """Bandpass filter hidden."""
        h64       = h.to(torch.float64)                       # (N, d)
        h_hat     = self.st.S @ h64
        mask      = torch.zeros(self.st.N, dtype=torch.float64,
                                device=self.st.device)
        mask[K_active] = 1.0
        h_hat_f   = h_hat * mask.unsqueeze(1)
        h_filt    = self.st.S @ h_hat_f

        norms     = h_filt.norm(dim=0, keepdim=True) + 1e-8
        return (h_filt / norms).float()                       # (N, d)

    def compute_phi(self, B_n: torch.Tensor,
                    h_filt: torch.Tensor,
                    Psi_a: torch.Tensor,
                    Psi_b: torch.Tensor) -> torch.Tensor:
        """Compute phi."""
        r_n = B_n.shape[1]
        h_a = h_filt[:, :r_n]                                 # (N, r_n)
        h_b = h_filt[:, r_n:2*r_n]                           # (N, r_n)


        arg_a = B_n * h_a + Psi_a.unsqueeze(0)               # (N, r_n)
        arg_b = B_n * h_b + Psi_b.unsqueeze(0)               # (N, r_n)

        Phi = torch.cat([torch.cos(arg_a),
                         torch.sin(arg_b)], dim=1)            # (N, 2r_n)
        return Phi

    def phi_to_spectral_features(self, Phi: torch.Tensor,
                                  K_n: torch.Tensor) -> torch.Tensor:
        """Phi to spectral features."""
        Phi_hat = self.st.dst_cols(Phi.to(torch.float64))    # (N, 2r_n)
        return Phi_hat[K_n, :].float()                        # (r_n, 2r_n)


# §6  PCGrad


def pcgrad_update(losses: list, params: list, opt: torch.optim.Optimizer):
    """Pcgrad update."""
    n = len(losses)
    grads = []


    for loss in losses:
        opt.zero_grad()
        loss.backward(retain_graph=True)
        g = torch.cat([
            p.grad.clone().flatten() if p.grad is not None
            else torch.zeros(p.numel(), device=p.device)
            for p in params
        ])
        grads.append(g)


    grads_pc = [g.clone() for g in grads]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = torch.dot(grads_pc[i], grads[j])
            if dot < 0:
                norm2 = grads[j].dot(grads[j]) + 1e-12
                grads_pc[i] = grads_pc[i] - (dot / norm2) * grads[j]


    opt.zero_grad()
    g_sum = sum(grads_pc)
    offset = 0
    for p in params:
        size   = p.numel()
        p.grad = g_sum[offset:offset+size].reshape(p.shape).clone()
        offset += size
    opt.step()


def setup_helmholtz(k: float, N: int, device):
    """Setup helmholtz."""
    j      = torch.arange(1, N+1, dtype=torch.float64, device=device)
    x_grid = j / (N+1)
    alpha  = 1.0
    beta   = float(np.cos(k) + np.sin(k))
    u_ex   = (torch.cos(torch.tensor(k, dtype=torch.float64) * x_grid)
             + torch.sin(torch.tensor(k, dtype=torch.float64) * x_grid)
             + x_grid * torch.sin(100.0 * np.pi * x_grid))
    f_grid = ((k**2 - (100.0*np.pi)**2) * x_grid * torch.sin(100.0*np.pi*x_grid)
             + 200.0*np.pi * torch.cos(100.0*np.pi*x_grid))
    return x_grid, f_grid, u_ex, alpha, beta


def _train_loop(net, all_params, opt, sch,
                loss_fn, epochs, delta, plateau_tol, eps_n,
                log_fn=None):
    """Train loop."""
    prev = float('inf'); plateau = 0; history = []

    for ep in range(epochs+1):
        total, per_task = loss_fn()

        if per_task is not None:
            pcgrad_update(per_task, all_params, opt)
        else:
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
        sch.step()

        lv = float(total)
        history.append(lv)

        rel = abs(lv-prev)/(abs(prev)+1e-30)
        plateau = plateau+1 if rel < delta else 0
        prev = lv

        if plateau >= plateau_tol:
            print("(see code)")
            break
        if lv < eps_n:
            print("(see code)")
            break

        if log_fn and ep % 200 == 0:
            log_fn(ep, lv)

    return history


class ASMGDL:
    """ASMGDL."""
    def __init__(self, k: float, N: int = 512,
                 hidden_dim: int = 128, device=device):
        self.k = float(k); self.N = N
        self.hidden_dim = hidden_dim; self.device = device

        self.st      = SpectralTools(N, k, device)
        self.phi_bld = PhiBuilder(self.st)

        x_grid, f_grid, u_exact, alpha, beta = setup_helmholtz(k, N, device)
        self.x_grid   = x_grid                    # (N,) float64
        self.f_grid   = f_grid                    # (N,) float64
        self.u_exact  = u_exact                   # (N,) float64
        self.alpha    = alpha
        self.beta     = beta
        self.x_in     = x_grid.float().unsqueeze(1)
        self.f_in     = f_grid.float()                # (N,)


        self.grade1      : Optional[Grade1Net]  = None
        self.grade1_h    : Optional[torch.Tensor] = None
        self.grade_n_list: list = []   # {net, B_n, Psi_a, Psi_b, K_prev, h_filt}
        self.K_active    : Optional[torch.Tensor] = None
        self.e_hat       : Optional[torch.Tensor] = None
        self.u_total     : Optional[torch.Tensor] = None

        self.history = {'rse': [], 'E_star': [], 'loss_per_grade': []}


    def _rse(self, u):
        return float(torch.norm(u.float()-self.u_exact.float())
                     / torch.norm(self.u_exact.float()))


    def _post_grade_analysis(self, u_total: torch.Tensor, grade_idx: int):
        """Post grade analysis."""
        with torch.no_grad():
            R_hat, _ = self.st.residual_hat(
                u_total, self.alpha, self.beta, self.f_in)

        e_hat = R_hat.detach()
        E_star = float((e_hat**2).sum())
        rse    = self._rse(u_total)

        K, tau, f_ang, phi_l, f_mean = self.st.find_active_modes(e_hat)

        self.K_active = K
        self.e_hat    = e_hat
        self.u_total  = u_total

        self.history['E_star'].append(E_star)
        self.history['rse'].append(rse)

        print("(see code)")
        print(f"    RSE={rse:.4e} | E*={E_star:.4e}")
        if K is not None and len(K) > 0:
            print(f"    G-D: τ={tau:.3e} | |K|={len(K)} | f̄={f_mean:.2f}")
        else:
            print("(see code)")

        return K, f_ang, phi_l, f_mean


    def train_grade1(self, epochs=3000, lr_start=1e-3, lr_end=1e-5,
                     eps0=5e-3, delta=1e-5, plateau_tol=100) -> None:
        print(f"\n{'='*60}")
        print(f"Grade 1 | Tanh | width={self.hidden_dim}")
        print(f"{'='*60}")

        net   = Grade1Net(self.hidden_dim).to(self.device)
        gamma = (lr_end/lr_start)**(1.0/epochs)
        opt   = torch.optim.Adam(net.parameters(), lr=lr_start)
        sch   = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)

        def loss_fn():
            u, _   = net(self.x_in, self.alpha, self.beta)
            R, _   = self.st.residual_hat(u, self.alpha, self.beta, self.f_in)
            total  = (R**2).sum()
            return total, None

        def log_fn(ep, lv):
            with torch.no_grad():
                u, _ = net(self.x_in, self.alpha, self.beta)
            print(f"  Epoch {ep:5d} | Loss:{lv:.3e} | RSE:{self._rse(u):.3e}")

        hist = _train_loop(net, list(net.parameters()), opt, sch,
                           loss_fn, epochs, delta, plateau_tol,
                           eps_n=eps0, log_fn=log_fn)
        self.history['loss_per_grade'].append(hist)


        net.freeze()
        self.grade1 = net
        with torch.no_grad():
            u_g1, h = net(self.x_in, self.alpha, self.beta)
        self.grade1_h = h.detach()                            # (N, hidden_dim)

        K, f_ang, phi_l, f_mean = self._post_grade_analysis(u_g1.detach(), 1)
        return K, f_ang, phi_l, f_mean


    def train_grade_n(self, grade_idx: int,
                      K_prev: torch.Tensor,
                      f_angular: torch.Tensor,
                      phi_l: torch.Tensor,
                      f_mean: float,
                      epochs=3000, lr_start=1e-3, lr_end=1e-5,
                      eps0=5e-3, delta=1e-5, plateau_tol=100,
                      use_pcgrad=True) -> tuple:
        r_n = len(K_prev)
        print(f"\n{'='*60}")
        print(f"Grade {grade_idx} | Snake(a={f_mean:.2f}) | "
              f"r_n={r_n} | width={2*r_n}")
        print(f"{'='*60}")


        h_raw = self.grade1_h                                  # (N, hidden_dim)


        B_n, r_svd = self.phi_bld.build_Bn(
            self.x_grid.float(), f_angular.float(), phi_l.float())
        # B_n: (N, r_svd)


        h_filt = self.phi_bld.bandpass_filter_hidden(h_raw, K_prev)

        d_need = 2 * r_svd
        d_have = h_filt.shape[1]
        if d_have < d_need:
            pad    = torch.zeros(self.N, d_need-d_have, device=self.device)
            h_filt = torch.cat([h_filt, pad], dim=1)
        elif d_have > d_need:
            h_filt = h_filt[:, :d_need]
        # h_filt: (N, 2*r_svd)


        Psi_a = nn.Parameter(torch.zeros(r_svd, device=self.device))
        Psi_b = nn.Parameter(torch.zeros(r_svd, device=self.device))


        net = GradeNNet(r_svd, f_mean).to(self.device)
        nn.init.zeros_(list(net.net.modules())[-1].weight)

        all_params = list(net.parameters()) + [Psi_a, Psi_b]
        gamma = (lr_end/lr_start)**(1.0/epochs)
        opt   = torch.optim.Adam(all_params, lr=lr_start)
        sch   = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)


        u_prev = self.u_total.float().detach()

        E_cur  = float((self.e_hat**2).sum())
        E_ref  = float(self.history['E_star'][0]) + 1e-30
        eps_n  = eps0 * (E_cur / E_ref)

        def loss_fn():

            Phi       = self.phi_bld.compute_phi(
                            B_n.float(), h_filt.float(),
                            Psi_a.float(), Psi_b.float())       # (N, 2r_svd)
            Phi_hat_K = self.phi_bld.phi_to_spectral_features(
                            Phi.to(torch.float64), K_prev)      # (r_n, 2r_svd)


            g_hat_K = net(Phi_hat_K.float())                   # (r_n,)


            g_phys  = self.st.idst_sparse(
                          g_hat_K.to(torch.float64), K_prev).float()  # (N,)


            u_total = u_prev + g_phys


            R_hat, _ = self.st.residual_hat(
                u_total, self.alpha, self.beta, self.f_in)

            total = (R_hat[K_prev]**2).sum()

            if use_pcgrad:

                per_task = [(R_hat[K_prev[i]]**2) for i in range(len(K_prev))]
            else:
                per_task = None

            return total, per_task

        def log_fn(ep, lv):
            with torch.no_grad():
                Phi   = self.phi_bld.compute_phi(
                            B_n.float(), h_filt.float(),
                            Psi_a.float(), Psi_b.float())
                PhiK  = self.phi_bld.phi_to_spectral_features(
                            Phi.to(torch.float64), K_prev)
                gh    = net(PhiK.float())
                gp    = self.st.idst_sparse(gh.to(torch.float64), K_prev).float()
                ut    = u_prev + gp
            print(f"  Epoch {ep:5d} | Loss:{lv:.3e} | RSE:{self._rse(ut):.3e}")

        hist = _train_loop(net, all_params, opt, sch,
                           loss_fn, epochs, delta, plateau_tol,
                           eps_n=eps_n, log_fn=log_fn)
        self.history['loss_per_grade'].append(hist)


        net.freeze()
        with torch.no_grad():
            Phi_f  = self.phi_bld.compute_phi(
                         B_n.float(), h_filt.float(),
                         Psi_a.float(), Psi_b.float())
            PhiK_f = self.phi_bld.phi_to_spectral_features(
                         Phi_f.to(torch.float64), K_prev)
            gh_f   = net(PhiK_f.float())
            gp_f   = self.st.idst_sparse(gh_f.to(torch.float64), K_prev).float()
            u_new  = (u_prev + gp_f).detach()

        self.grade_n_list.append({
            'net': net, 'B_n': B_n, 'h_filt': h_filt,
            'Psi_a': Psi_a.detach(), 'Psi_b': Psi_b.detach(),
            'K_prev': K_prev
        })

        K_new, f_ang_new, phi_new, f_mean_new = self._post_grade_analysis(
            u_new, grade_idx)
        return K_new, f_ang_new, phi_new, f_mean_new


    def solve(self, num_grades: int = 4,
              epochs: int = 3000,
              lr_start: float = 1e-3,
              lr_end: float = 1e-5,
              eps_global: float = 1e-7,
              **kw):

        kwargs = dict(epochs=epochs, lr_start=lr_start,
                      lr_end=lr_end, **kw)
        start  = time.time()

        grade1_kwargs = kwargs.copy()
        grade1_kwargs.pop("use_pcgrad", None)
        K, f_ang, phi_l, f_mean = self.train_grade1(**grade1_kwargs)

        for g in range(2, num_grades+1):
            E = self.history['E_star'][-1]
            if E < eps_global:
                print("(see code)")
                break
            if K is None or len(K) == 0:
                print("(see code)")
                break

            K, f_ang, phi_l, f_mean = self.train_grade_n(
                g, K, f_ang, phi_l, f_mean, **kwargs)

        print(f"\n{'='*60}")
        print("(see code)")
        print("(see code)")
        return self.u_total


if __name__ == '__main__':
    solver = ASMGDL(k=20.0, N=512, hidden_dim=128)
    u_pred = solver.solve(
        num_grades   = 4,
        epochs       = 3000,
        lr_start     = 1e-3,
        lr_end       = 1e-5,
        plateau_tol  = 100,
        use_pcgrad   = True,
    )
