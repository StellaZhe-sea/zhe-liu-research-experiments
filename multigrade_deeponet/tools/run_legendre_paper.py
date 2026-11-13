"""Legendre-transform experiment: Appendix S7/S10, official local modules."""
import argparse
import contextlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(ROOT / 'src'))


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def train_with_full_checkpoints(system, space, net, output):
    """Keep the official data/objective, persist data and all Adam variables."""
    import numpy as np
    import tensorflow as tf
    import deepxde as dde
    from deeponet_pde import test_u_lt
    from utils import mean_squared_error_outlier

    train_x, train_y = system.gen_operator_data(space, 200, 500)
    test_x, test_y = system.gen_operator_data(space, 200, 1000)
    # Save before training, so recovery never needs a guessed reconstruction.
    np.savez_compressed(str(output / 'data.npz'), train_branch=train_x[0],
                        train_trunk=train_x[1], train_y=train_y,
                        test_branch=test_x[0], test_trunk=test_x[1], test_y=test_y)
    data = dde.data.OpDataSet(train_x, train_y, test_x, test_y)
    model = dde.Model(data, net)
    model.compile('adam', lr=0.0001, metrics=[mean_squared_error_outlier])
    # DeepXDE 0.11.2 creates its saver BEFORE Adam slots; replace it AFTER compile.
    model.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=2)
    latest_saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=2)
    (output / 'latest').mkdir()

    class DurableCheckpoint(dde.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = float('inf')
            self.best_path = None

        def persist(self):
            state = self.model.train_state
            step = int(state.step)
            loss = float(np.sum(state.loss_train))
            if loss < self.best:
                self.best = loss
                self.best_path = self.model.saver.save(self.model.sess,
                    str(output / 'model' / 'best'), global_step=step)
            latest = latest_saver.save(self.model.sess,
                str(output / 'latest' / 'state'), global_step=step)
            history = self.model.losshistory
            table = np.column_stack((history.steps, np.array(history.loss_train).reshape(-1),
                                     np.array(history.loss_test).reshape(-1),
                                     np.array(history.metrics_test).reshape(-1)))
            temporary = output / 'loss.pending.dat'
            np.savetxt(str(temporary), table, header='step train_mse test_mse trimmed_test_mse')
            os.replace(str(temporary), str(output / 'loss.dat'))
            pending = output / 'resume.pending.json'
            pending.write_text(json.dumps(dict(step=step, latest_checkpoint=latest,
                best_checkpoint=self.best_path, best_training_mse=self.best,
                optimizer_state_saved=True, data_file='data.npz'), indent=2), encoding='utf-8')
            os.replace(str(pending), str(output / 'resume.json'))

        def on_train_begin(self):
            self.persist()

        def on_epoch_end(self):
            if self.model.train_state.step % 1000 == 0:
                self.persist()

    checkpoint = DurableCheckpoint()
    try:
        history, state = model.train(epochs=2000000, display_every=1000, callbacks=[checkpoint])
        print('# Parameters:', int(sum(np.prod(v.shape.as_list()) for v in tf.compat.v1.trainable_variables())))
        dde.saveplot(history, state, issave=True, isplot=False, output_dir=str(output))
        model.restore(checkpoint.best_path, verbose=1)
        prediction = model.predict(test_x)
        mse = float(np.mean((test_y - prediction) ** 2))
        trimmed = float(mean_squared_error_outlier(test_y, prediction))
        np.savez_compressed(str(output / 'test_predictions.npz'), reference=test_y, prediction=prediction)
        (output / 'metrics.json').write_text(json.dumps(dict(full_test_mse=mse,
            trimmed_test_mse=trimmed, best_step=state.best_step,
            best_training_mse=float(state.best_loss_train)), indent=2), encoding='utf-8')
        print('Test MSE:', mse, '\nTest MSE w/o outliers:', trimmed)
        examples = [(lambda x: x, 'x.dat'), (lambda x: np.sin(np.pi*x), 'sinx.dat'),
                    (lambda x: np.sin(2*np.pi*x), 'sin2x.dat'),
                    (lambda x: x*np.sin(2*np.pi*x), 'xsin2x.dat')]
        for u, name in examples:
            test_u_lt('opnn', system, 2, 200, model, data, u, name)
        samples = space.eval_u(space.random(10), np.linspace(0, 2, 200)[:, None])
        for i, sample in enumerate(samples):
            test_u_lt('opnn', system, 2, 200, model, data,
                      lambda x, sample=sample: sample, str(i) + '.dat')
    except KeyboardInterrupt:
        print('Interrupted. Use the checkpoint referenced by resume.json; do not restart blindly.', flush=True)
        raise
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--show-config', action='store_true')
    mode.add_argument('--check', action='store_true', help='two-step smoke test, not a paper run')
    args = parser.parse_args()
    config = dict(case='legendre_transform', model='DeepONet', seed=args.seed,
                  sensors=200, train_functions=500, test_functions=1000,
                  coefficients=20, train_points=10000, test_points=20000,
                  epochs=2000000, learning_rate=0.0001, optimizer='Adam',
                  branch=[200, 100, 100], trunk=[1, 100, 100],
                  activation='relu', initializer='Glorot normal',
                  stacked=False, use_bias=True, grf_length_scale=0.2,
                  grf_grid_points=2000, grf_interpolation='cubic',
                  physical_domain=[-1, 1], grf_storage_domain=[0, 2],
                  coordinate_mapping='physical x = storage coordinate - 1; no rescaling',
                  reference='Official LTSystem.eval_s: cubic interpolation and 10000-point right rectangle sum',
                  checkpoint_selection='minimum recorded training loss',
                  interpreter=sys.executable, device='cpu',
                  checkpoint_policy='full TensorFlow state including Adam every 1000 steps; retain 2 latest and 2 best',
                  source='Appendix S7 Tables S2/S3 and S10; official src LTSystem and GRF')
    if args.show_config:
        print(json.dumps(config, indent=2))
        return
    if not args.check:
        for prior in (ROOT / 'results').glob('legendre_deeponet_paper_*'):
            cfg, log = prior / 'config.json', prior / 'run.log'
            if cfg.exists() and log.exists():
                if json.loads(cfg.read_text(encoding='utf-8')).get('seed') == args.seed and 'LEGENDRE_RUN_COMPLETE' in log.read_text(encoding='utf-8'):
                    parser.error('Completed run already exists: ' + str(prior))

    import numpy as np
    import tensorflow as tf
    import deepxde as dde
    from deeponet_pde import run, test_u_lt
    from spaces import GRF
    from system import LTSystem

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    prefix = 'legendre_check_' if args.check else 'legendre_deeponet_paper_'
    output = ROOT / 'results' / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('model', 'test'):
        (output / folder).mkdir()
    config['smoke_test'] = args.check
    if args.check:
        config.update(train_functions=2, test_functions=2, train_points=40,
                      test_points=40, epochs=2)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    os.chdir(output)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('Interpreter:', sys.executable, '\nResults:', output, flush=True)
            print(json.dumps(config, indent=2), flush=True)
            net = dde.maps.OpNN(config['branch'], config['trunk'], 'relu',
                               'Glorot normal', use_bias=True, stacked=False)
            system = LTSystem(20)
            space = GRF(2, length_scale=0.2, N=2000, interp='cubic')
            if args.check:
                x, y = system.gen_operator_data(space, 200, 2)
                assert x[0].shape == (40, 200) and x[1].shape == (40, 1) and y.shape == (40, 1)
                assert np.isfinite(y).all()
                data = dde.data.OpDataSet(x, y, x, y)
                model = dde.Model(data, net)
                model.compile('adam', lr=0.0001)
                model.train(epochs=2, display_every=1)
                assert np.isfinite(model.predict(x)).all()
                parameters = int(sum(np.prod(v.shape.as_list()) for v in tf.compat.v1.trainable_variables()))
                print('# Parameters:', parameters)
                test_u_lt('opnn', system, 2, 200, model, data, lambda z: z, 'x.dat')
                values = np.loadtxt('test/s_x.dat')
                assert values.shape == (20, 3) and np.isfinite(values).all()
                model.close()
                print('LEGENDRE_CHECK_OK', flush=True)
                return
            train_with_full_checkpoints(system, space, net, output)
            print('LEGENDRE_RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
