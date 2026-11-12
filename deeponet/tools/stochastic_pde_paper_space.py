"""Apply the main paper Problem 6 sigma=0.1 to official KL modes."""
from sde import GRFs


class PaperSPDEGRFs(GRFs):
    def eval_KL_bases(self, ls, sensors, M):
        return 0.1 * super().eval_KL_bases(ls, sensors, M)
