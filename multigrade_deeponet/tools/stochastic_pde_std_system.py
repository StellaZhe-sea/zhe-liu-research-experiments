"""Official SPDE reference solver with its documented standard-deviation target."""
import numpy as np
from sde import SPDESystem
from utils import trapz


class StandardDeviationSPDE(SPDESystem):
    def eval_s(self, sensor_value):
        h = self.T / (self.Nx - 1)
        sensor_value = sensor_value.reshape((self.M, self.Nx))
        tmp = 1 / sensor_value
        v1 = trapz(tmp, h)
        tmp *= self.f * np.linspace(0, self.T, num=self.Nx)
        v2 = trapz(tmp, h)
        C = v2[:, -1:] / v1[:, -1:]
        v = C * v1 - v2
        return np.std(v, axis=0)
