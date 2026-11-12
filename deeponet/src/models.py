"""Official modern baseline and an explicitly separate MultiGrade extension."""
import runtime
import copy
import math
import torch
from torch import nn
import deepxde as dde


def legacy_glorot_normal(tensor):
    """Match TF's variance-corrected normal truncated at two raw stddevs."""
    fan_out, fan_in = tensor.shape
    raw_std = math.sqrt(2.0/(fan_in+fan_out))/0.87962566103423978
    return nn.init.trunc_normal_(tensor, mean=0.0, std=raw_std,
                                 a=-2*raw_std, b=2*raw_std)


def baseline(sensors=100, width=40):
    # Same lists, initialization, branch linear output, trunk ReLU output,
    # scalar bias and dot product as the snapshotted legacy OpNN defaults.
    return dde.nn.DeepONet([sensors, width, width], [1, width, width],
                          'relu', legacy_glorot_normal)


class Grade(nn.Module):
    def __init__(self, branch_in, trunk_in, width, zero_correction=False):
        super().__init__()
        self.branch_hidden = nn.Sequential(nn.Linear(branch_in, width), nn.ReLU())
        self.trunk_hidden = nn.Sequential(nn.Linear(trunk_in, width), nn.ReLU())
        self.branch_head = nn.Linear(width, width)
        self.trunk_head = nn.Linear(width, width)
        self.bias = nn.Parameter(torch.zeros(1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                legacy_glorot_normal(module.weight)
                nn.init.zeros_(module.bias)
        if zero_correction:
            nn.init.zeros_(self.branch_head.weight)
            nn.init.zeros_(self.branch_head.bias)

    def features(self, branch, trunk):
        return self.branch_hidden(branch), self.trunk_hidden(trunk)

    def correction_from_features(self, branch, trunk):
        b = self.branch_head(branch)
        t = torch.relu(self.trunk_head(trunk))
        return torch.einsum('bi,bi->b', b, t).unsqueeze(1) + self.bias

    def forward(self, branch, trunk):
        b, t = self.features(branch, trunk)
        return b, t, self.correction_from_features(b, t)


class MultiGradeDeepONet(nn.Module):
    def __init__(self, sensors=100, width=40):
        super().__init__()
        self.sensors, self.width = sensors, width
        # Copy official initial parameters rather than claiming identical RNG
        # draws across separately implemented constructors.
        original = baseline(sensors, width)
        first = Grade(sensors, 1, width)
        first.branch_hidden[0] = copy.deepcopy(original.branch.linears[0])
        first.branch_head = copy.deepcopy(original.branch.linears[1])
        first.trunk_hidden[0] = copy.deepcopy(original.trunk.linears[0])
        first.trunk_head = copy.deepcopy(original.trunk.linears[1])
        first.bias = nn.Parameter(original.b[0].detach().clone())
        self.grades = nn.ModuleList([first])

    def add_grade(self):
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        new = Grade(self.width, self.width, self.width, zero_correction=True)
        new.to(next(self.parameters()).device)
        self.grades.append(new)
        return new

    def forward(self, inputs):
        b, t = inputs
        total = torch.zeros((b.shape[0], 1), dtype=b.dtype, device=b.device)
        for grade in self.grades:
            b, t, correction = grade(b, t)
            total = total + correction
        return total

    @torch.no_grad()
    def cache_previous(self, inputs):
        b, t = inputs
        total = torch.zeros((b.shape[0], 1), dtype=b.dtype, device=b.device)
        for grade in self.grades[:-1]:
            b, t, correction = grade(b, t)
            total = total + correction
        return (b.detach(), t.detach()), total.detach()


class LegacyAdam(torch.optim.Optimizer):
    """TensorFlow 1 Adam epsilon placement, used by DeepXDE 0.11.2.

    TensorFlow: lr*sqrt(1-beta2**t)/(1-beta1**t) * m/(sqrt(v)+eps).
    This avoids substituting PyTorch Adam's bias-corrected epsilon placement.
    """
    def __init__(self, params, lr=0.001):
        super().__init__(params, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Only dense gradients are supported.')
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state['step'] += 1
                state['m'].mul_(beta1).add_(p.grad, alpha=1-beta1)
                state['v'].mul_(beta2).addcmul_(p.grad, p.grad, value=1-beta2)
                lr_t = group['lr'] * (1-beta2**state['step'])**0.5 / (1-beta1**state['step'])
                p.addcdiv_(state['m'], state['v'].sqrt().add_(group['eps']), value=-lr_t)
        return loss
