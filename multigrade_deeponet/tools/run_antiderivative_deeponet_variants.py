"""Run only the three antiderivative DeepONet variants missing locally.

The existing unstacked-with-bias full run is archived separately and is
intentionally not a selectable variant here, to prevent accidental reruns.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT / "src"))

import deepxde as dde
import numpy as np
import tensorflow as tf

from deeponet_pde import run
from spaces import GRF
from system import ODESystem


VARIANTS = {
    "stacked_bias": (True, True),
    "stacked_no_bias": (True, False),
    "unstacked_no_bias": (False, False),
}


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--validate-graph", action="store_true", help="build the graph without training")
    parser.add_argument("--check", action="store_true", help="two synthetic steps only")
    args = parser.parse_args()
    if sum((args.show_config, args.validate_graph, args.check)) > 1:
        parser.error("choose only one of --show-config, --validate-graph, and --check")

    stacked, use_bias = VARIANTS[args.variant]
    # DeepXDE 0.11.2's stacked initializer registry has He/LeCun but no
    # Glorot option. The appendix does not specify the initializer; record
    # this necessary implementation choice rather than silently patching it.
    initializer = "He normal" if stacked else "Glorot normal"
    config = {
        "case": "antiderivative",
        "model": "DeepONet",
        "variant": args.variant,
        "equation": "ds/dx=u(x), s(0)=0, x in [0,1]",
        "stacked": stacked,
        "use_bias": use_bias,
        "bias_semantics": "Official OpNN use_bias flag controls the final branch layer bias and scalar output bias",
        "branch": [100, 40, 40],
        "trunk": [1, 40, 40],
        "T": 1,
        "sensors": 100,
        "grf_length_scale": 0.2,
        "train_functions": 10000,
        "training_points_per_function": 1,
        "test_functions": 100000,
        "epochs": 50000,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "activation": "relu",
        "initializer": initializer,
        "initializer_note": "Appendix does not fix an initializer; He normal is supported by official stacked OpNN",
        "seed": args.seed,
        "interpreter": sys.executable,
        "checkpoint_selection": "minimum training loss, official repository convention",
        "source": "Paper Appendix Tables S2, S3, S6; official DeepXDE 0.11.2 OpNN",
    }
    if args.show_config:
        print(json.dumps(config, indent=2))
        return

    if not args.check:
        for prior in (ROOT / "results").glob("antiderivative_deeponet_" + args.variant + "_*"):
            prior_config = prior / "config.json"
            prior_log = prior / "run.log"
            if prior_config.exists() and prior_log.exists():
                recorded = json.loads(prior_config.read_text(encoding="utf-8"))
                if recorded.get("seed") == args.seed and "RUN_COMPLETE" in prior_log.read_text(encoding="utf-8"):
                    parser.error("completed run with this variant and seed already exists: " + str(prior))

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    net = dde.maps.OpNN(
        config["branch"], config["trunk"], "relu", initializer,
        use_bias=use_bias, stacked=stacked,
    )

    if args.validate_graph:
        branch = np.zeros((1, 100), dtype=np.float32)
        trunk = np.zeros((1, 1), dtype=np.float32)
        targets = np.zeros((1, 1), dtype=np.float32)
        data = dde.data.OpDataSet((branch, trunk), targets, (branch, trunk), targets)
        model = dde.Model(data, net)
        model.compile("adam", lr=0.001)
        parameters = int(sum(np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()))
        print("ANTIDERIVATIVE_DEEPONET_GRAPH_OK", args.variant, "parameters", parameters)
        model.close()
        return

    if args.check:
        branch = np.random.randn(16, 100).astype(np.float32)
        trunk = np.random.rand(16, 1).astype(np.float32)
        targets = np.zeros((16, 1), dtype=np.float32)
        data = dde.data.OpDataSet((branch, trunk), targets, (branch, trunk), targets)
        model = dde.Model(data, net)
        model.compile("adam", lr=0.001)
        model.train(epochs=2, display_every=1)
        assert np.isfinite(model.predict((branch, trunk))).all()
        print("ANTIDERIVATIVE_DEEPONET_CHECK_OK", args.variant, sys.executable)
        return

    output = ROOT / "results" / (
        "antiderivative_deeponet_" + args.variant + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "model").mkdir()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.chdir(output)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (output / "run.log").open("w", encoding="utf-8") as logfile:
        sys.stdout, sys.stderr = Tee(original_stdout, logfile), Tee(original_stderr, logfile)
        try:
            print("Interpreter:", sys.executable)
            print("Results:", output)
            system = ODESystem(lambda state, forcing, coordinate: forcing, [0], 1)
            space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
            run("ode", system, space, 1, 100, "opnn", net, 0.001, 50000, 10000, 100000)
            print("ANTIDERIVATIVE_DEEPONET_RUN_COMPLETE", args.variant)
            subprocess.run([sys.executable, str(ROOT / "tools" / "plot_ode_results.py"), str(output)], check=True)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
