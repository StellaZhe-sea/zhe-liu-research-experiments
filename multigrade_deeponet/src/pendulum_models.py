"""Untrained four-grade extension of the identified width-1000 pendulum model.

The parent's runner remains an ordinary single DeepONet. This constructor does
not replace that runner, train a model, or claim matched parameter counts.
"""
from pathlib import Path
import numpy as np
import torch
from models import MultiGradeDeepONet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT.parent / 'data/pendulum_T3_w1000_gpu_seed42/tf_reference.npz'


def pendulum_multigrade(grades=4, fixture=DEFAULT_FIXTURE):
    if not 1 <= grades <= 4:
        raise ValueError('This study specifies one through four grades.')
    model = MultiGradeDeepONet(sensors=100, width=1000)
    first = model.grades[0]
    # Parent TF layout: [input, output]; nn.Linear layout: [output, input].
    mapping = [
        (first.branch_hidden[0].weight, 0, True),
        (first.branch_head.weight, 1, True),
        (first.branch_hidden[0].bias, 2, False),
        (first.branch_head.bias, 3, False),
        (first.trunk_hidden[0].weight, 4, True),
        (first.trunk_head.weight, 5, True),
        (first.trunk_hidden[0].bias, 6, False),
        (first.trunk_head.bias, 7, False),
        (first.bias, 8, False),
    ]
    with np.load(fixture) as arrays, torch.no_grad():
        for parameter, index, transpose in mapping:
            value = arrays[f'step0_param_{index:02d}']
            value = value.T if transpose else value
            parameter.copy_(torch.as_tensor(value, device=parameter.device).reshape(parameter.shape))
    for _ in range(grades - 1):
        model.add_grade()
    return model
