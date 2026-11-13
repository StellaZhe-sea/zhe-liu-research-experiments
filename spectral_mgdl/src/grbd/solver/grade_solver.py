"""grbd.solver.grade_solver — GradeSolver, fully basis-agnostic."""
import numpy as np, torch
from .checkpoint             import LrAdaptor, train_grade
from ..network.grade_net     import GradeNet
from ..network.spectral_loss import make_loss


class GradeSolver:
    def __init__(self, op, basis, f, u_star,
                 base_width=64, base_depth=3, lambda_oob=0.1,
                 stop_min_improve=0.02, stop_delta_H=0.05,
                 base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                 max_retry=1, device="cpu"):
        self.op=op; self.basis=basis; self.f=f; self.u_star=u_star
        self.base_width=base_width; self.base_depth=base_depth
        self.lambda_oob=lambda_oob; self.stop_min_improve=stop_min_improve
        self.stop_delta_H=stop_delta_H; self.max_retry=max_retry
        self.device=torch.device(device)
        self.lr_adaptor=LrAdaptor(base_lr=base_lr,lr_min=lr_min,lr_max=lr_max)
        self._u=np.zeros(op.N,np.float32); self._e_next=None
        self.grades=[]; self.infos=[]

    def _config(self, H, n_atoms, rel_err):
        depth=max(2,min(self.base_depth+int(H*1.5),8))
        cap=128 if rel_err<1e-3 else (256 if rel_err<1e-2 else 512)
        return min(max(self.base_width,2*n_atoms),cap), depth

    def _should_stop(self):
        if not self.infos: return False
        info=self.infos[-1]; eb,ea=info["rel_err_before"],info["rel_err_after"]
        improve=(eb-ea)/(eb+1e-30)
        if ea>=eb:
            print(f"  [EARLY STOP]  RSE flat/rose {eb:.4e}→{ea:.4e}"); return True
        if improve<self.stop_min_improve:
            print(f"  [EARLY STOP]  improve={improve*100:.1f}%<{self.stop_min_improve*100:.0f}%"); return True
        if len(self.infos)>=2:
            dH=abs(self.infos[-1]["entropy"]-self.infos[-2]["entropy"])
            if dH<self.stop_delta_H:
                print(f"  [EARLY STOP]  ΔH={dH:.4f}<{self.stop_delta_H}"); return True
        return False

    def add_grade(self, n_epochs=2000):
        n=len(self.grades)+1
        print(f"\n{'═'*62}\n  GRADE {n}  [{type(self.basis).__name__}]\n{'═'*62}")
        e_n = self._e_next if self._e_next is not None else self.op.residual(self._u,self.f)
        self._e_next=None
        target=-self.op.inv_L(e_n)
        rel_b=float(np.linalg.norm(self._u-self.u_star)/(np.linalg.norm(self.u_star)+1e-30))
        print(f"  ‖e‖={np.linalg.norm(e_n):.4e}  RSE_before={rel_b:.4e}")
        lr=self.lr_adaptor.update(self.infos[-1]["losses"] if self.infos else [],grade_n=n)
        print(f"  lr={lr:.2e}")

        _,meta=self.basis.decompose(e_n)
        s,meta=self.basis.atom_features(e_n,target,meta)
        width,depth=self._config(meta["entropy"],meta["n_atoms"],rel_b)
        print(f"  |K_n|={meta['n_atoms']}  H={meta['entropy']:.3f}  w={width}  d={depth}")

        s_t  =torch.tensor(s,     dtype=torch.float32).to(self.device)
        tgt_t=torch.tensor(target,dtype=torch.float32).to(self.device)
        u_before=self._u.copy(); retry_lr=lr

        for attempt in range(self.max_retry+1):
            if attempt>0:
                retry_lr*=0.5; print(f"  [RETRY {attempt}] lr={retry_lr:.2e}")
            crit=make_loss(self.basis,meta,self.lambda_oob).to(self.device)
            net=GradeNet(meta["n_atoms"],width,depth).to(self.device)
            g_np,losses,losses_in,losses_oob=train_grade(net,s_t,tgt_t,crit,retry_lr,n_epochs,self.device)
            u_cand=u_before+g_np
            rel_a=float(np.linalg.norm(u_cand-self.u_star)/(np.linalg.norm(self.u_star)+1e-30))
            if rel_a<rel_b:
                self._u=u_cand; break
            else:
                print(f"  [WARNING] attempt {attempt+1}: {rel_b:.4e}→{rel_a:.4e}")
                if attempt==self.max_retry:
                    print(f"  [REJECTED] grade {n} discarded."); rel_a=rel_b

        print(f"  RSE_after={rel_a:.4e}  {'↓' if rel_a<rel_b else '→'}")
        net.requires_grad_(False)
        self._e_next=self.op.residual(self._u,self.f)

        info=dict(**meta, width=width, depth=depth, losses=losses,
                  losses_in=losses_in, losses_oob=losses_oob, g_np=g_np,
                  rel_err_before=rel_b, rel_err_after=rel_a, lr_used=retry_lr,
                  basis_name=type(self.basis).__name__)
        self.grades.append(net); self.infos.append(info)
        return info

    def solve(self, n_grades=8, n_epochs=2000):
        for i in range(n_grades):
            self.add_grade(n_epochs)
            if self._should_stop():
                print(f"  Stopped after {i+1} grades."); break
        self.print_table()
        return self._u.copy()

    @property
    def u(self): return self._u.copy()

    @property
    def final_rse(self): return self.infos[-1]["rel_err_after"] if self.infos else float("inf")

    def print_table(self):
        print(f"\n{'═'*96}")
        hdr="G    basis                |K|    H      w    d    lr        before      after       Δ%     ✓"
        print(hdr); print("─"*96)
        for i,info in enumerate(self.infos):
            eb,ea=info["rel_err_before"],info["rel_err_after"]
            ok="✓" if not any(p.requires_grad for p in self.grades[i].parameters()) else "✗"
            print(f" {i+1:^3}  {info['basis_name']:^20}  {info['n_atoms']:^5}"
                  f"  {info['entropy']:^6.3f}  {info['width']:^4}  {info['depth']:^3}"
                  f"  {info['lr_used']:^8.2e}  {eb:^10.3e}  {ea:^10.3e}"
                  f"  {(eb-ea)/eb*100:^+7.1f}%  {ok}")
        print("═"*96)
        print(f"  Final RSE: {self.final_rse:.4e}")
