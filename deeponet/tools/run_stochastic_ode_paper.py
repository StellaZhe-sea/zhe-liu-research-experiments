"""Paper Problem 5 / Fig. 5: pathwise stochastic ODE, official data generator."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
from run_legendre_paper import Tee


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--problem', choices=('ode', 'pde'), default='ode')
    parser.add_argument('--clip-rv', action='store_true', help='PDE only: clip xi to [-3.1,3.1] before reference solve')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--show-config', action='store_true')
    modes.add_argument('--validate-graph', action='store_true')
    args = parser.parse_args(argv)
    is_pde = args.problem == 'pde'
    if args.clip_rv and not is_pde:
        parser.error('--clip-rv requires --problem pde')
    prefix = 'stochastic_pde_pathwise_paper_' if is_pde else 'stochastic_ode_pathwise_paper_'
    marker = 'STOCHASTIC_PDE_RUN_COMPLETE' if is_pde else 'STOCHASTIC_ODE_RUN_COMPLETE'
    if args.clip_rv:
        prefix = 'stochastic_pde_pathwise_clipped_'
        marker = 'STOCHASTIC_PDE_CLIPPED_RUN_COMPLETE'
    cfg = dict(case='stochastic_ode_pathwise', equation='dy/dt=k(t,omega)*y', initial_value=1,
               seed=args.seed, seed_note='Local seed, not a documented paper seed',
               kernel='RBF', length_scale_range=[1, 2], KL_modes=5, sensors_per_mode=20,
               reference_grid=100, branch=[100, 100, 100], trunk=[6, 100, 100],
               trunk_order='xi_1,...,xi_5,t (official code ordering)',
               train_functions=10000, test_functions=10000, points_per_function=100,
               train_points=1000000, test_points=1000000, epochs=50000,
               learning_rate=0.001, activation='relu', initializer='Glorot normal',
               optimizer='Adam', batch_size='full dataset, official run default',
               checkpoint_selection='minimum recorded full training MSE',
               paper_test_mse_mean=8e-5, interpreter=sys.executable,
               source='Main paper Problem 5/Fig.5; Appendix S2/S3; src/sde.py pathwise block')
    if is_pde:
        cfg.pop('initial_value')
        cfg.update(case='stochastic_pde_pathwise', equation='-d/dx(exp(b(x,omega))*du/dx)=10',
                   boundary_conditions='u(0)=u(1)=0', forcing=10, sigma=0.1,
                   trunk_order='xi_1,...,xi_5,x (official code ordering)',
                   random_variable_clipping=False,
                   scaling_note='Main paper specifies sigma=0.1; official pathwise block omits scaling. Local GRFs subclass scales both reference and branch KL modes by 0.1.',
                   paper_test_mse_mean=2e-3,
                   source='Main paper Problem 6/Fig.6a; Appendix S2/S3; official SPDESystem pathwise solver with explicit paper sigma')
    if args.clip_rv:
        cfg.update(case='stochastic_pde_pathwise_clipped', random_variable_clipping=True,
                   clipping_bounds=[-3.1,3.1], paper_test_mse_mean=9.4e-4,
                   clipping_note='Applied before reference solution and trunk assembly in training, test, and examples; not rejection sampling',
                   source='Main paper page 226 clipped-xi comparison; same local sigma=0.1 convention as completed unbounded run')
    if args.show_config:
        print(json.dumps(cfg, indent=2))
        return
    if not args.validate_graph:
        prior = list((ROOT / 'results').glob(prefix + '*/config.json'))
        for path in prior:
            if json.loads(path.read_text()).get('seed') == args.seed:
                parser.error('Existing run found; refusing duplicate: ' + str(path.parent))
    import numpy as np
    import tensorflow as tf
    import deepxde as dde
    from sde import SODESystem, SPDESystem, GRFs
    from utils import mean_squared_error_outlier
    def make_system():
        if args.clip_rv:
            from stochastic_pde_clipped_system import ClippedSPDESystem
            return ClippedSPDESystem(1, 10, 100, None, 100)
        return SPDESystem(1, 10, 100, None, 100) if is_pde else SODESystem(1, 1, Nx=100, npoints_output=100)

    def make_space():
        if is_pde:
            from stochastic_pde_paper_space import PaperSPDEGRFs
            return PaperSPDEGRFs(1, 'RBF', 1, 2, N=100, interp='linear')
        return GRFs(1, 'RBF', 1, 2, N=100, interp='linear')
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    net = dde.maps.OpNN(cfg['branch'], cfg['trunk'], 'relu', 'Glorot normal', use_bias=True, stacked=False)
    if args.validate_graph:
        # Exercise a batch beyond the old 65535 limit on this CPU installation.
        x = [np.random.randn(65536, 100).astype('float32'), np.random.randn(65536, 6).astype('float32')]
        y = np.zeros((65536, 1), dtype='float32')
        model = dde.Model(dde.data.OpDataSet(x, y, x, y), net)
        try:
            model.compile('adam', lr=0.001)
            model.sess.run(tf.compat.v1.global_variables_initializer())
            whole = model.predict(x)
            chunks = np.vstack([model.predict([v[i:i+8192] for v in x]) for i in range(0, len(y), 8192)])
            assert np.isfinite(whole).all() and np.allclose(whole, chunks, rtol=1e-5, atol=2e-6)
            count = sum(int(np.prod(v.shape.as_list())) for v in tf.compat.v1.trainable_variables())
            assert count == 31001
            system = make_system()
            if is_pde:
                grid = np.linspace(0, 1, 100)
                expected = 5 * grid * (1-grid)
                assert np.allclose(system.eval_s_path(np.zeros(500), np.ones(5)), expected, atol=1e-12)
                assert np.allclose(system.eval_s_path(np.ones(500), np.array([0.1, 0, 0, 0, 0])), expected*np.exp(-0.1), atol=1e-12)
                raw = GRFs(1, 'RBF', 1, 2, N=100, interp='linear')
                scaled = make_space()
                sensors = grid[::5, None]
                ls = np.array([[1.5]])
                assert np.allclose(scaled.eval_KL_bases(ls, sensors, 5), 0.1*raw.eval_KL_bases(ls, sensors, 5))
            else:
                assert np.allclose(system.eval_s_path(np.zeros(500), np.ones(5)), 1)
                assert np.allclose(system.eval_s_path(np.ones(500), np.array([1, 0, 0, 0, 0])), np.exp(np.linspace(0, 1, 100)))
            tiny_space = make_space()
            if args.clip_rv:
                from unittest.mock import patch
                raw_rv = np.array([[8.,-8.,1.,-1.,0.]])
                with patch('stochastic_pde_clipped_system.np.random.randn', return_value=raw_rv):
                    np.testing.assert_array_equal(system.draw_rv(1,5), np.clip(raw_rv,-3.1,3.1))
                    ex, ey = system.gen_example_data_path(tiny_space,1.5,20,5)
                bases=tiny_space.eval_KL_bases(np.array([[1.5]]),np.linspace(0,1,100)[:,None],5)
                expected=system.eval_s_path(bases[0],np.clip(raw_rv[0],-3.1,3.1))
                np.testing.assert_allclose(ey[:,0], expected)
                np.testing.assert_allclose(ex[1][:,:5],np.tile(np.clip(raw_rv,-3.1,3.1),(100,1)))
            tiny_x, tiny_y = system.gen_operator_data_path(tiny_space, 20, 5, 2)
            assert tiny_x[0].shape == (200, 100) and tiny_x[1].shape == (200, 6)
            assert tiny_y.shape == (200, 1) and np.isfinite(tiny_y).all()
            if args.clip_rv:
                assert np.max(np.abs(tiny_x[1][:,:5])) <= 3.1
                model.train(epochs=2, display_every=1)
                print('CLIPPED_REFERENCE_AND_INPUT_CHECK_OK; synthetic optimizer smoke test completed')
            assert np.allclose(tiny_y[tiny_x[1][:, -1] == 0], 0 if is_pde else 1)
            if is_pde:
                assert np.allclose(tiny_y[tiny_x[1][:, -1] == 1], 0, atol=1e-12)
            print(('STOCHASTIC_PDE_GRAPH_OK' if is_pde else 'STOCHASTIC_ODE_GRAPH_OK')
                  + ' parameters=31001 CPU_batch_65536_matches_chunks optimizer_updates=' + ('2' if args.clip_rv else '0'))
        finally:
            model.close()
        return

    out = ROOT / 'results' / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    (out / 'model').mkdir()
    (out / 'latest').mkdir()
    (out / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Interpreter:', sys.executable, '\nResults:', out, flush=True)
            print(json.dumps(cfg, indent=2), flush=True)
            system = make_system()
            space = make_space()
            print('Generating TRAIN data: 10000 random processes / 1000000 pairs', flush=True)
            train_x, train_y = system.gen_operator_data_path(space, 20, 5, 10000)
            np.savez_compressed(str(out / 'train.npz'), X_train0=train_x[0], X_train1=train_x[1], y_train=train_y)
            print('Generating TEST data: 10000 random processes / 1000000 pairs', flush=True)
            test_x, test_y = system.gen_operator_data_path(space, 20, 5, 10000)
            np.savez_compressed(str(out / 'test.npz'), X_test0=test_x[0], X_test1=test_x[1], y_test=test_y)
            assert train_x[0].shape == test_x[0].shape == (1000000, 100)
            assert train_x[1].shape == test_x[1].shape == (1000000, 6)
            assert all(np.isfinite(v).all() for v in list(train_x) + list(test_x) + [train_y, test_y])
            model = dde.Model(dde.data.OpDataSet(train_x, train_y, [v[:65535] for v in test_x], test_y[:65535]), net)
            model.compile('adam', lr=0.001, metrics=[mean_squared_error_outlier])
            model.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=2)
            latest = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=2)

            class Persist(dde.callbacks.Callback):
                def __init__(self):
                    super().__init__()
                    self.best, self.best_path = float('inf'), None

                def save(self):
                    state, history = model.train_state, model.losshistory
                    loss, step = float(np.sum(state.loss_train)), int(state.step)
                    if loss < self.best:
                        self.best = loss
                        self.best_path = model.saver.save(model.sess, str(out / 'model' / 'best'), global_step=step)
                    path = latest.save(model.sess, str(out / 'latest' / 'state'), global_step=step)
                    np.savetxt(str(out / 'loss.pending.dat'), np.column_stack((history.steps, history.loss_train, history.loss_test, history.metrics_test)), header='step train_mse monitoring_test_mse monitoring_trimmed_mse')
                    os.replace(str(out / 'loss.pending.dat'), str(out / 'loss.dat'))
                    (out / 'resume.pending.json').write_text(json.dumps(dict(step=step, latest_checkpoint=path, best_checkpoint=self.best_path, optimizer_state_saved=True)), encoding='utf-8')
                    os.replace(str(out / 'resume.pending.json'), str(out / 'resume.json'))

                def on_train_begin(self):
                    self.save()

                def on_epoch_end(self):
                    if model.train_state.step % 1000 == 0:
                        self.save()

            persist = Persist()
            try:
                from run_stochastic_pde_std import awake_seconds
                start = time.perf_counter()
                active_start = awake_seconds()
                _, state = model.train(epochs=50000, display_every=1000, callbacks=[persist])
                active_seconds = awake_seconds() - active_start
                seconds = time.perf_counter() - start
                model.restore(persist.best_path, verbose=1)
                pred = np.vstack([model.predict([v[i:i+65535] for v in test_x]) for i in range(0, len(test_y), 65535)])
                metrics = dict(full_test_mse=float(np.mean((pred-test_y)**2)), trimmed_test_mse=float(mean_squared_error_outlier(test_y, pred)),
                               best_step=int(state.best_step), best_training_mse=persist.best, parameters=31001, training_seconds=seconds,
                               training_active_seconds=active_seconds, training_wall_seconds=seconds,
                               timer='Windows QueryUnbiasedInterruptTime excludes sleep/hibernation; not CPU time')
                np.savez_compressed(str(out / 'test_predictions.npz'), reference=test_y, prediction=pred)
                (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
                print(json.dumps(metrics, indent=2), flush=True)
                for i in range(10):
                    x, y = system.gen_example_data_path(space, 1.5, 20, 5)
                    np.savez_compressed(str(out / ('example{}.npz'.format(i))), branch=x[0], trunk=x[1], reference=y, prediction=model.predict(x))
                print(marker, flush=True)
            finally:
                model.close()


if __name__ == '__main__':
    main()
