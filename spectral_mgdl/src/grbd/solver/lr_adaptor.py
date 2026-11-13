"""
grbd.solver.lr_adaptor
======================
Adaptive learning rate scheduler across grades.

After each grade completes, analyses the loss curve tail and
adjusts the base lr for the next grade's optimiser.
"""

import numpy as np


class LrAdaptor:
    """
    Parameters
    ----------
    base_lr     : float   initial lr
    lr_min      : float   hard floor
    lr_max      : float   hard ceiling
    tail_frac   : float   fraction of loss history considered "tail"
    slope_eps   : float   tail slope threshold for "still descending"
    decay_bad   : float   lr multiplier when convergence is poor
    decay_mild  : float   lr multiplier for moderate convergence
    boost_good  : float   lr multiplier for excellent convergence
    """

    def __init__(self, base_lr=2e-3, lr_min=1e-4, lr_max=5e-3,
                 tail_frac=0.20, slope_eps=-1e-10,
                 decay_bad=0.50, decay_mild=0.70, boost_good=1.20):
        self.base_lr    = base_lr
        self.lr_min     = lr_min
        self.lr_max     = lr_max
        self.tail_frac  = tail_frac
        self.slope_eps  = slope_eps
        self.decay_bad  = decay_bad
        self.decay_mild = decay_mild
        self.boost_good = boost_good
        self._current   = base_lr

    @property
    def current_lr(self) -> float:
        return self._current

    def reset(self):
        self._current = self.base_lr

    def update(self, losses: list, grade_n: int) -> float:
        if grade_n <= 1:
            self._current = self.base_lr
            return self._current

        arr  = np.array(losses, dtype=np.float64)
        n    = len(arr)
        ts   = max(0, int(n * (1 - self.tail_frac)))
        tail = arr[ts:]

        x     = np.arange(len(tail), dtype=np.float64)
        slope = np.polyfit(x, tail, 1)[0] / (tail[0] + 1e-30)
        ratio = tail.mean() / (arr[:max(1, ts)].mean() + 1e-30)

        lr = self._current
        if slope < self.slope_eps:
            reason = f"still descending (slope={slope:.2e})"
        elif ratio > 0.5:
            lr     = lr * self.decay_bad
            reason = f"poor convergence (ratio={ratio:.2f}) → ×{self.decay_bad}"
        elif ratio < 0.01:
            lr     = lr * self.boost_good
            reason = f"excellent (ratio={ratio:.2f}) → ×{self.boost_good}"
        else:
            lr     = lr * self.decay_mild
            reason = f"moderate (ratio={ratio:.2f}) → ×{self.decay_mild}"

        lr            = float(np.clip(lr, self.lr_min, self.lr_max))
        self._current = lr
        print(f"  [LrAdaptor]  {reason}  →  lr={lr:.2e}")
        return lr
