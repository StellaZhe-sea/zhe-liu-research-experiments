"""Four-grade pendulum pilot; every update logged. Larger than original baseline."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import subprocess

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(ROOT/'src'))
import runtime
import numpy as np
import torch
from pendulum_models import pendulum_multigrade, DEFAULT_FIXTURE
from pendulum_validation import load_validation
sys.path.insert(0,str(ROOT.parent/'tools'))
from gpu_deeponet import TF1Adam, strict_cuda
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=ROOT.parent/'results/pendulum_T3_w200_n10000_seed42_20261020_150058_429249'


def grade_learning_rate(initial, decay_ratio, local_step, steps):
    """Cosine decay per grade; ratio is final/initial, not per-update decay."""
    if steps <= 1 or local_step <= 1 or decay_ratio == 1:
        return initial
    progress=(local_step-1)/(steps-1)
    return initial*(decay_ratio+(1-decay_ratio)*0.5*(1+math.cos(math.pi*progress)))


def plot(rows,out):
    updates=[r for r in rows if r['event']=='update']
    if not updates:
        return
    plt.rcParams.update({'svg.fonttype':'none','path.simplify':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=[r['epoch'] for r in updates]
    for ax,suffix in zip(axes,['mse','relative_l2']):
        for split,style in [('train','-'),('test','-')]:
            ax.semilogy(x,np.maximum([r[split+'_'+suffix] for r in updates],np.finfo(float).tiny),style,label=split)
        for r in rows:
            if r['event']=='initialized' and r['grade']>1:
                ax.axvline(r['epoch'],color='0.6',linestyle='-',linewidth=.8)
        ax.set(xlabel='Epoch (full-batch optimizer update)',ylabel=suffix.replace('_',' '))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle(f'Pendulum MultiGrade: every one of {len(updates)} recorded updates')
    fig.savefig(out/'metrics.svg',format='svg')
    fig.savefig(out/'metrics_preview.png',dpi=120)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=[42],default=42)
    parser.add_argument('--steps-per-grade',type=int,default=100)
    parser.add_argument('--grade-steps',type=int,nargs=4,metavar=('G1','G2','G3','G4'),
                        help='Updates for each grade; overrides --steps-per-grade')
    parser.add_argument('--output-dir',type=Path,help='New output directory inside this project')
    parser.add_argument('--validation-data',type=Path,help='Independent validation NPZ; monitoring only, no gradients')
    parser.add_argument('--lr',type=float,default=.001,help='Starting learning rate in each grade')
    parser.add_argument('--grade-lrs',type=float,nargs=4,metavar=('LR1','LR2','LR3','LR4'),
                        help='Per-grade starting LRs; overrides --lr')
    parser.add_argument('--decay-ratio',type=float,default=1.,help='Final/initial LR per grade; cosine decay, 1 keeps constant LR')
    parser.add_argument('--check',action='store_true',help='Pipeline test: 8 pairs per split, 2 updates per grade')
    parser.add_argument('--check-device',choices=['cpu','cuda'],default='cpu',help='Device for --check only; normal runs always require CUDA')
    args=parser.parse_args()
    if args.steps_per_grade<1:
        parser.error('steps-per-grade must be positive')
    if args.grade_steps is not None and any(k < 1 for k in args.grade_steps):
        parser.error('all grade-steps must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr must be finite and positive')
    if args.grade_lrs is not None and any(not math.isfinite(rate) or rate<=0 for rate in args.grade_lrs):
        parser.error('grade-lrs must be finite and positive')
    if not math.isfinite(args.decay_ratio) or not 0 < args.decay_ratio <= 1:
        parser.error('decay-ratio must be in (0, 1]')
    if args.output_dir is not None:
        args.output_dir=args.output_dir.resolve()
        if not args.output_dir.is_relative_to(ROOT) or args.output_dir.exists():
            parser.error('output-dir must be new and inside this project')
    torch.set_default_device('cpu')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device=torch.device('cpu') if args.check and args.check_device=='cpu' else strict_cuda()
    grade_steps=[2]*4 if args.check else (args.grade_steps or [args.steps_per_grade]*4)
    total_updates=sum(grade_steps)
    grade_lrs=args.grade_lrs or [args.lr]*4
    source_config=json.loads((SOURCE/'config.json').read_text())
    for key,value in {'T':3,'k':1,'sensors':100,'train_samples':10000,'test_samples':100000,'points_per_function':1}.items():
        assert source_config[key]==value, key
    data={}
    for split,count in [('train',10000),('test',100000)]:
        with np.load(SOURCE/(split+'.npz')) as archive:
            arrays=[archive[k].astype(np.float32) for k in ['branch','trunk','reference']]
        assert [a.shape for a in arrays]==[(count,100),(count,1),(count,1)]
        assert all(np.isfinite(a).all() for a in arrays)
        data[split]=tuple(torch.tensor(a[:8] if args.check else a,device=device) for a in arrays)
    if args.validation_data is not None:
        data['validation']=load_validation(args.validation_data,device,args.check)
    model=pendulum_multigrade().to(device)
    assert sum(p.numel() for p in model.parameters())==14117004
    out=args.output_dir or ROOT/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_pendulum_multigrade'+('_check' if args.check else ''))
    out.mkdir(parents=True)
    config=dict(seed=42,steps_per_grade=grade_steps[0] if len(set(grade_steps))==1 else None,
                grade_steps=grade_steps,total_updates=total_updates,device=str(device),python=sys.executable,
                source=str(SOURCE),width=1000,parameters=14117004,original_baseline_parameters=2105001,
                parameter_matched=False,diagnostic_only=args.check,learning_rate=args.lr,
                grade_learning_rates=grade_lrs,
                validation_data=str(args.validation_data.resolve()) if args.validation_data else None,
                decay_ratio=args.decay_ratio,learning_rate_schedule='Cosine per grade; first update lr, last update lr*decay_ratio; one-update grade uses lr',
                optimizer='Parent TF1Adam; restarted per grade',checkpoint_selection='Training MSE every 1000 grade updates, initial and final',
                evaluation='Full train and full test after EVERY update; test never selects checkpoints',
                hashes={})
    for path in [SOURCE/'train.npz',SOURCE/'test.npz',DEFAULT_FIXTURE,ROOT/'src/pendulum_models.py',ROOT/'src/models.py',Path(__file__),ROOT.parent/'tools/gpu_deeponet.py']:
        with path.open('rb') as handle:
            config['hashes'][str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    if args.validation_data:
        for path in [args.validation_data.resolve(),args.validation_data.resolve().with_name('manifest.json'),ROOT/'src/pendulum_validation.py']:
            with path.open('rb') as handle: config['hashes'][str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
        config['evaluation']='Full train/test/independent validation after EVERY update; training MSE selects checkpoints'
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'OUTPUT: {out}\nInterpreter: {sys.executable}\nDevice: {device}',flush=True)
    fields=['epoch','grade','grade_step','event','learning_rate','train_mse','test_mse','train_relative_l2','test_relative_l2','elapsed_seconds']
    if args.validation_data: fields+=['validation_mse','validation_relative_l2']
    rows=[]
    epoch=0
    start=time.perf_counter()
    summary=dict(status='running',config=config,grades=[])
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as cf,(out/'metrics.md').open('w',encoding='utf-8') as mf:
        writer=csv.DictWriter(cf,fieldnames=fields)
        writer.writeheader()
        mf.write('# Pendulum MultiGrade: every optimizer update\n\n'
                 'Full training and full test MSE, untrimmed. Independent validation is included when configured. Test never selects checkpoints. '
                 'Initialization/restoration events are additional rows, not updates. '
                 'This is a larger-model exploratory run, not a fair-capacity comparison.\n\n')
        mf.write('| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n')
        try:
            for grade in range(1,5):
                steps=grade_steps[grade-1]
                grade_lr=grade_lrs[grade-1]
                for p in model.parameters():
                    p.requires_grad_(False)
                    p.grad=None
                current=model.grades[grade-1]
                for p in current.parameters(): p.requires_grad_(True)
                optimizer=TF1Adam(current.parameters(),lr=grade_lr)
                frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
                cache={}
                with torch.no_grad():
                    for split,(u,t,y) in data.items():
                        bs,ts,olds=[],[],[]
                        for offset in range(0,len(u),4096):
                            b,x=u[offset:offset+4096],t[offset:offset+4096]
                            old=torch.zeros_like(y[offset:offset+4096])
                            for previous in model.grades[:grade-1]:
                                b,x,q=previous(b,x)
                                old=old+q
                            bs.append(b);ts.append(x);olds.append(old)
                        cache[split]=(torch.cat(bs),torch.cat(ts),torch.cat(olds),y)

                @torch.no_grad()
                def record(event,local_step):
                    result=dict(epoch=epoch,grade=grade,grade_step=local_step,event=event,
                                learning_rate=grade_learning_rate(grade_lr,args.decay_ratio,local_step,steps))
                    for split,(b,t,old,y) in cache.items():
                        squared=torch.zeros((),device=device,dtype=torch.float64)
                        for offset in range(0,len(b),4096):
                            error=old[offset:offset+4096]+current(b[offset:offset+4096],t[offset:offset+4096])[2]-y[offset:offset+4096]
                            squared+=error.double().square().sum()
                        result[split+'_mse']=(squared/len(y)).item()
                        result[split+'_relative_l2']=(squared/y.double().square().sum()).sqrt().item()
                    assert all(np.isfinite(result[k]) for k in ['train_mse','test_mse','train_relative_l2','test_relative_l2'])
                    if args.validation_data: assert np.isfinite(result['validation_mse']) and np.isfinite(result['validation_relative_l2'])
                    result['elapsed_seconds']=time.perf_counter()-start
                    rows.append(result); writer.writerow(result);cf.flush()
                    mf.write('| '+' | '.join(f'{result[k]:.12e}' if isinstance(result[k],float) else str(result[k]) for k in fields)+' |\n');mf.flush()
                    return result

                selected=record('initialized',0)
                best=selected['train_mse']; best_step=0
                checkpoint=out/f'grade_{grade}_best.pt'
                def save():
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),grade=grade,
                                    epoch=epoch,selected_grade_step=best_step,config=config),checkpoint)
                save()
                for local_step in range(1,steps+1):
                    for group in optimizer.param_groups:
                        group['lr']=grade_learning_rate(grade_lr,args.decay_ratio,local_step,steps)
                    b,t,old,y=cache['train']
                    optimizer.zero_grad(set_to_none=True)
                    loss=(current(b,t)[2]-(y-old)).square().mean()
                    loss.backward();optimizer.step();epoch+=1
                    row=record('update',local_step)
                    if local_step%1000==0 or local_step==steps:
                        if row['train_mse']<best:
                            best=row['train_mse'];best_step=local_step;save()
                    if local_step%100==0 or local_step==steps:
                        plot(rows,out)
                        print(f'Grade {grade}/4 | epoch {epoch}/{total_updates} | train {row["train_mse"]:.6e} | test {row["test_mse"]:.6e}',flush=True)
                model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'])
                selected=record('restored_checkpoint',best_step)
                for name,p in model.named_parameters():
                    if name in frozen: assert torch.equal(p,frozen[name]),name
                with torch.no_grad():
                    u,t,_=data['test'];b,x,old,_=cache['test']
                    torch.testing.assert_close(model((u[:8],t[:8])),old[:8]+current(b[:8],x[:8])[2],rtol=1e-5,atol=1e-6)
                summary['grades'].append(dict(grade=grade,selected_grade_step=best_step,metrics=selected,frozen_unchanged=True))
                del cache, frozen
            summary['status']='complete'
        except BaseException:
            summary['status']='interrupted_or_failed'
            raise
        finally:
            plot(rows,out)
            summary.update(completed_updates=epoch,wall_seconds=time.perf_counter()-start)
            (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Verify complete update sequence and every Markdown/CSV cell.
    assert [r['epoch'] for r in rows if r['event']=='update']==list(range(1,total_updates+1))
    table=[line for line in (out/'metrics.md').read_text().splitlines() if line.startswith('|')][2:]
    assert len(table)==len(rows)
    for line,row in zip(table,rows):
        cells=[x.strip() for x in line.strip('|').split('|')]
        for key,cell in zip(fields,cells):
            if key=='event': assert cell==row[key]
            else: np.testing.assert_allclose(float(cell),row[key],rtol=1e-11,atol=1e-12)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/verify_svg.py'),str(out)],check=True)
    print(f'COMPLETE: {out}',flush=True)


if __name__=='__main__':
    main()
