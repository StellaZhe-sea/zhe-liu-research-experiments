"""Isolated repository-configuration SPDE standard-deviation experiment."""
import argparse
import contextlib
import ctypes
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
from run_legendre_paper import Tee


class LoggingTee(Tee):
    def close(self):
        # absl may close its stream during shutdown; never close the terminal.
        if not self.logfile.closed:
            self.logfile.flush()


def awake_seconds():
    # Windows unbiased interrupt time excludes sleep/hibernation; not CPU time.
    value = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
        raise ctypes.WinError()
    return value.value / 10000000.0


def execute(out, seed, check):
    import numpy as np
    import tensorflow as tf
    import deepxde as dde
    from sde import GRFs
    from stochastic_pde_std_system import StandardDeviationSPDE
    from utils import mean_squared_error_outlier
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    count, mc, epochs = (2, 8, 2) if check else (1000, 20000, 20000)
    config = dict(target='pointwise population standard deviation', seed=seed,
                  seed_note='Local seed, not a documented paper seed',
                  source='Repository statistical configuration; Appendix equivalence not yet verified',
                  train_functions=count, test_functions=count, monte_carlo_realizations=mc,
                  spatial_points=100, points_per_function=10, kernel='RBF',
                  length_scale_range=[0.2, 2], sigma=0.1, KL_modes=8, sensors_per_mode=30,
                  branch=[240, 100, 100], trunk=[1, 100, 100], activation='relu',
                  initializer='Glorot normal', learning_rate=0.001, epochs=epochs,
                  standard_deviation_ddof=0, interpreter=sys.executable)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    system = StandardDeviationSPDE(1, 10, 100, mc, 10)
    # Constant coefficients yield known parabolas; verify the target operation.
    coeffs = np.linspace(1, 2, mc)
    x = np.linspace(0, 1, 100)
    expected = np.std(5*x[None, :]*(1-x[None, :])/coeffs[:, None], axis=0)
    actual = system.eval_s(np.repeat(coeffs[:, None], 100, axis=1))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    start_wall, start_awake = time.perf_counter(), awake_seconds()
    space = GRFs(1, 'RBF', 0.2, 2, N=100, interp='linear')
    print('STAGE 1: Generating standard-deviation training and test data...', flush=True)
    tx, ty = system.gen_operator_data(space, 30, 8, count, 'KL')
    np.savez_compressed(str(out / 'train.npz'), X_train0=tx[0], X_train1=tx[1], y_train=ty)
    vx, vy = system.gen_operator_data(space, 30, 8, count, 'KL')
    np.savez_compressed(str(out / 'test.npz'), X_test0=vx[0], X_test1=vx[1], y_test=vy)
    assert tx[0].shape == (count*10, 240) and tx[1].shape == (count*10, 1)
    assert np.isfinite(ty).all() and np.all(ty >= 0)
    data_seconds = awake_seconds() - start_awake
    net = dde.maps.OpNN([240, 100, 100], [1, 100, 100], 'relu', 'Glorot normal', use_bias=True, stacked=False)
    model = dde.Model(dde.data.OpDataSet(tx, ty, vx, vy), net)
    model.compile('adam', lr=0.001, metrics=[mean_squared_error_outlier])
    model.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=2)
    checker = dde.callbacks.ModelCheckpoint(str(out / 'model' / 'best'), save_better_only=True, period=1 if check else 1000)
    print('STAGE 2: Training...', flush=True)
    train_wall, train_awake = time.perf_counter(), awake_seconds()
    history, state = model.train(epochs=epochs, callbacks=[checker])
    training_active = awake_seconds() - train_awake
    training_wall = time.perf_counter() - train_wall
    dde.saveplot(history, state, issave=True, isplot=False, output_dir=str(out))
    model.restore(str(out / 'model' / ('best-' + str(state.best_step))), verbose=1)
    pred = model.predict(vx)
    np.savez_compressed(str(out / 'test_predictions.npz'), reference=vy, prediction=pred)
    metrics = dict(best_step=int(state.best_step), best_training_mse=float(state.best_loss_train),
                   full_test_mse=float(np.mean((pred-vy)**2)),
                   trimmed_test_mse=float(mean_squared_error_outlier(vy, pred)),
                   parameters=int(sum(np.prod(v.shape.as_list()) for v in tf.compat.v1.trainable_variables())),
                   data_generation_active_seconds=data_seconds,
                   training_active_seconds=training_active, training_wall_seconds=training_wall,
                   timer='Windows QueryUnbiasedInterruptTime; excludes sleep/hibernation, not CPU time')
    assert metrics['parameters'] == 44501
    for i in range(1 if check else 10):
        ex, ey = system.gen_example_data(space, 0.2 + 0.2*i, 30, 8, 'KL')
        ep = model.predict(ex)
        np.savez_compressed(str(out / ('example%d.npz' % i)), branch=ex[0], trunk=ex[1], reference=ey, prediction=ep)
        np.savetxt(str(out / ('example%d.dat' % i)), np.hstack((ex[1], ey, ep)))
    metrics.update(total_active_seconds=awake_seconds()-start_awake, total_wall_seconds=time.perf_counter()-start_wall)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print('STOCHASTIC_PDE_STD_CHECK_OK' if check else 'STOCHASTIC_PDE_STD_RUN_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check', action='store_true', help='Tiny validation only, not a formal experiment')
    args = parser.parse_args()
    if not args.check and list((ROOT / 'results').glob('stochastic_pde_std_repo_*')):
        raise SystemExit('A standard-deviation run already exists. Inspect it before rerunning.')
    base = ROOT / ('tmp' if args.check else 'results')
    prefix = 'stochastic_pde_std_check_' if args.check else 'stochastic_pde_std_repo_'
    out = base / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (out / 'model').mkdir(parents=True)
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(LoggingTee(sys.stdout, log)), contextlib.redirect_stderr(LoggingTee(sys.stderr, log)):
            print('Interpreter:', sys.executable, '\nResults:', out, flush=True)
            execute(out, args.seed, args.check)


if __name__ == '__main__':
    main()
