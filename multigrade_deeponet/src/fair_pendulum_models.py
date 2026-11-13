"""Exact-capacity comparison with ordinary DeepONet and hidden-feature MultiGrade."""
from pathlib import Path
import sys
import torch
from models import MultiGradeDeepONet, legacy_glorot_normal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from gpu_deeponet import UnstackedDeepONet


def make_pair(width,seed=42,grade_depths=None):
    if width<2: raise ValueError('Width must be at least 2')
    depths=[1]*4 if grade_depths is None else list(grade_depths)
    if len(depths)!=4 or any(type(d) is not int or d<1 for d in depths):
        raise ValueError('Specify four positive integer grade depths')
    depth=sum(depths)
    if depths!=[1]*4 and (width!=64 or depth>16):
        raise ValueError('Deeper matched pairs currently support width64 and total depth<=16')
    first_trunk=width-1 if depth==4 else 71-2*depth
    head=4*width+5 if depth==4 else 253+2*depth
    branch_layers=[100]+[width]*depth+[head]
    trunk_layers=[1,first_trunk]+[width-1]*(depth-1)+[head]
    torch.manual_seed(seed)
    traditional=UnstackedDeepONet(branch_layers,trunk_layers,activation='relu',use_bias=True)
    with torch.no_grad():
        for weights in [traditional.branch_weights,traditional.trunk_weights]:
            for weight in weights: legacy_glorot_normal(weight)
    torch.manual_seed(seed)
    multigrade=MultiGradeDeepONet(sensors=100,width=width)
    for _ in range(3): multigrade.add_grade()
    for grade,hidden_depth in zip(multigrade.grades,depths):
        for tower in (grade.branch_hidden,grade.trunk_hidden):
            for _ in range(hidden_depth-1):
                layer=torch.nn.Linear(width,width)
                legacy_glorot_normal(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                tower.append(layer);tower.append(torch.nn.ReLU())
    # All these parameters are learnable during their own grade; the trainer
    # selects the currently active grade. Count the entire learned model budget.
    for p in multigrade.parameters(): p.requires_grad_(True)
    expected=14*width*width+117*width+4+2*(depth-4)*(width*width+width)
    counts={name:sum(p.numel() for p in model.parameters())
            for name,model in [('traditional',traditional),('multigrade',multigrade)]}
    assert counts['traditional']==counts['multigrade']==expected,counts
    assert all(p.requires_grad for p in traditional.parameters())
    report=dict(parameters=counts,parameter_matched=True,width=width,
        traditional_branch=branch_layers,traditional_trunk=trunk_layers,
        multigrade_hidden_depths=depths,total_hidden_depth_per_tower=depth,
        multigrade_hidden_widths=[width]*4,multigrade_head_widths=[width]*4,
        longest_affine_path={'traditional':depth+1,'multigrade':depth+1},
        total_affine_maps={'traditional':2*(depth+1),'multigrade':2*(depth+4)},
        parameter_formula='14*w^2 + 117*w + 4 + 2*(sum(grade_depths)-4)*(w^2+w)',
        padding_parameters=0,
        limitations='Active trainable count differs by grade; internal widths/readout widths, total affine maps and runtime differ. Equal total capacity and longest path do not imply identical graphs or compute.')
    return {'traditional':traditional,'multigrade':multigrade},report
