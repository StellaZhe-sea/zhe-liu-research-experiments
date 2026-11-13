"""grbd.solver.checkpoint — LrAdaptor + train_grade (best-checkpoint)."""
import copy, numpy as np, torch, torch.optim as optim


class LrAdaptor:
    def __init__(self, base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                 tail_frac=0.20, decay_bad=0.50, decay_mild=0.70, boost_good=1.20):
        self.base_lr=base_lr; self.lr_min=lr_min; self.lr_max=lr_max
        self.tail_frac=tail_frac; self.decay_bad=decay_bad
        self.decay_mild=decay_mild; self.boost_good=boost_good
        self._lr=base_lr

    def reset(self): self._lr = self.base_lr

    def update(self, losses, grade_n):
        if grade_n <= 1: self._lr = self.base_lr; return self._lr
        losses = np.array(losses, dtype=np.float64)
        n=len(losses); ts=max(0,int(n*(1-self.tail_frac)))
        tail=losses[ts:]; head=losses[:max(1,ts)]
        ratio = tail.mean()/(head.mean()+1e-30)
        lr = self._lr
        if   ratio > 0.5:  lr*=self.decay_bad;  tag=f"poor({ratio:.2f})×{self.decay_bad}"
        elif ratio < 0.01: lr*=self.boost_good; tag=f"great({ratio:.2f})×{self.boost_good}"
        else:              lr*=self.decay_mild; tag=f"mid({ratio:.2f})×{self.decay_mild}"
        self._lr = float(np.clip(lr, self.lr_min, self.lr_max))
        print(f"    [LrAdaptor] {tag}  lr={self._lr:.2e}")
        return self._lr

    @property
    def current_lr(self): return self._lr


def train_grade(net, s_t, tgt_t, criterion, lr, n_epochs, device):
    """Train one GradeNet; return (best_g_np, losses, losses_in, losses_oob)."""
    opt   = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.005)
    losses, losses_in, losses_oob = [], [], []
    best_loss=float("inf"); best_state=None
    net.train()
    for ep in range(n_epochs):
        opt.zero_grad()
        g = net(s_t)
        total, l_in, l_oob = criterion(g, tgt_t)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step(); sched.step()
        lv = total.item()
        losses.append(lv); losses_in.append(l_in); losses_oob.append(l_oob)
        if lv < best_loss:
            best_loss=lv; best_state=copy.deepcopy(net.state_dict())
        if (ep+1)%500==0:
            print(f"    ep{ep+1:5d}  tot={lv:.3e}  in={l_in:.3e}  oob={l_oob:.3e}  best={best_loss:.3e}")
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        best_g = net(s_t).cpu().numpy().astype(np.float32)
    return best_g, losses, losses_in, losses_oob
