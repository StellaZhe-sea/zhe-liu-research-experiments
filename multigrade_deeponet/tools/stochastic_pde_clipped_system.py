"""Official SPDE pathwise construction with xi clipped before solving."""
import numpy as np
from sde import SPDESystem, ProcessPool, config, make_triple


class ClippedSPDESystem(SPDESystem):
    @staticmethod
    def draw_rv(num, modes):
        return np.clip(np.random.randn(num, modes), -3.1, 3.1)

    def gen_operator_data_path(self, space, Nx, M, num):
        print('Generating operator data (xi clipped to [-3.1, 3.1])...', flush=True)
        features = space.random(num)
        x = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, x, M)
        rv = self.draw_rv(num, M)
        pool = ProcessPool(nodes=config.processes)
        try:
            s_values = np.array(pool.map(self.eval_s_path, bases, rv))
        finally:
            pool.close(); pool.join(); pool.clear()
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_values = np.hstack((space.eval_KL_bases(features, sensors, M), rv))
        res = np.vstack([make_triple(sensor_values[i], x, s_values[i], self.npoints_output) for i in range(num)])
        m = Nx * M
        assert np.max(np.abs(res[:, m:m+M])) <= 3.1
        return [res[:, :m], res[:, m:-1]], res[:, -1:]

    def gen_example_data_path(self, space, l, Nx, M):
        features = np.full((1, 1), l)
        x = np.linspace(0, self.T, num=self.Nx)[:, None]
        bases = space.eval_KL_bases(features, x, M)
        rv = self.draw_rv(1, M)
        values = self.eval_s_path(bases[0], rv[0])
        sensors = np.linspace(0, self.T, num=Nx)[:, None]
        sensor_values = space.eval_KL_bases(features, sensors, M)
        return [np.tile(sensor_values, (self.Nx, 1)), np.hstack((np.tile(rv, (self.Nx, 1)), x))], values[:, None]
