"""Parameter-matched pendulum training. Tuning never loads test data."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from run_pendulum_multigrade import ROOT, SOURCE, grade_learning_rate
from fair_pendulum_models import make_pair
from pendulum_validation import load_validation
from pendulum_preprocessing import fit_transform
from gpu_deeponet import strict_cuda, TF1Adam
from export_pendulum_comparison import export
import numpy as np
import torch


def digest(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def load_development(device,validation,check=False):
    # This function deliberately has no test-file argument or test-file read.
    with np.load(SOURCE/'train.npz') as archive:
        train=tuple(torch.tensor(archive[k][:16] if check else archive[k],dtype=torch.float32,device=device)
                    for k in ['branch','trunk','reference'])
    return {'train':train,'validation':load_validation(validation,device,check)}


def train(method,model,data,budgets,lrs,ratio,out,batch_size=0,batch_seed=1042,target_scale=1.,checkpoint_interval=1000,start_grade=1,residual_units=False):
    assert isinstance(checkpoint_interval,int) and checkpoint_interval>=1
    assert isinstance(start_grade,int) and start_grade>=1
    assert method=='multigrade' or start_grade==1
    assert not residual_units or method=='multigrade'
    directory=out/method; directory.mkdir()
    model=model.to(data['train'][0].device); device=next(model.parameters()).device
    total=sum(budgets); epoch=0; rows=[]; grades=[]; start=time.perf_counter()
    ntrain=len(data['train'][0]); actual_batch=ntrain if batch_size==0 else min(batch_size,ntrain)
    sampler=torch.Generator(device='cpu').manual_seed(batch_seed)
    order=torch.empty(0,dtype=torch.long,device='cpu');position=0;examples_seen=0;batch_digest=hashlib.sha256()
    fields=['epoch','grade','grade_step','event','learning_rate']
    for split in data: fields.extend([split+'_mse',split+'_relative_l2',split+'_rse'])
    fields.extend(['batch_size','examples_seen','elapsed_seconds'])
    segments=list(zip(budgets,lrs)) if method=='multigrade' else [(total,lrs[0])]
    ratios=list(ratio) if isinstance(ratio,(list,tuple)) else [ratio]*len(segments)
    assert len(ratios)==len(segments) and all(0<r<=1 for r in ratios)
    if method=='multigrade':
        assert len(budgets)==len(lrs) and start_grade+len(segments)-1<=len(model.grades)
    with (directory/'metrics.csv').open('w',newline='') as cf,(directory/'metrics.md').open('w') as mf:
        writer=csv.DictWriter(cf,fieldnames=fields); writer.writeheader()
        mf.write('# Every update; training-selected checkpoints\n\n| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n')
        for grade,((steps,initial_lr),segment_ratio) in enumerate(zip(segments,ratios),start_grade):
            if method=='multigrade':
                for p in model.parameters(): p.requires_grad_(False); p.grad=None
                current=model.grades[grade-1]
                for p in current.parameters(): p.requires_grad_(True)
                frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
                cache={}
                with torch.no_grad():
                    for split,(u,t,y) in data.items():
                        bs=[];ts=[];olds=[]
                        for offset in range(0,len(u),4096):
                            b,x=u[offset:offset+4096],t[offset:offset+4096]; old=torch.zeros_like(y[offset:offset+4096])
                            for previous in model.grades[:grade-1]: b,x,q=previous(b,x); old+=q
                            bs.append(b);ts.append(x);olds.append(old)
                        cache[split]=(torch.cat(bs),torch.cat(ts),torch.cat(olds),y)
                def predict(b,t): return current(b,t)[2]
                active=list(current.parameters())
            else:
                cache={split:(u,t,torch.zeros_like(y),y) for split,(u,t,y) in data.items()}
                def predict(b,t): return model(b,t)
                active=list(model.parameters()); frozen={}
            units=1.
            if residual_units:
                # The full TRAINING residual determines one fixed scale per grade.
                # predict() includes this scale; divide the residual loss by it.
                with torch.no_grad():
                    _,_,old,y=cache['train']
                    units=max(float((y-old).double().square().mean().sqrt()),1e-6)
                    current.output_scale.fill_(units)
            frozen_buffers={name:buffer.detach().clone() for name,buffer in model.named_buffers()}
            optimizer=TF1Adam(active,lr=initial_lr)

            @torch.no_grad()
            def record(event,local_step):
                row=dict(epoch=epoch,grade=grade,grade_step=local_step,event=event,
                         learning_rate=grade_learning_rate(initial_lr,segment_ratio,local_step,steps))
                for split,(b,t,old,y) in cache.items():
                    squared=torch.zeros((),device=device,dtype=torch.float64)
                    for offset in range(0,len(b),4096):
                        error=old[offset:offset+4096]+predict(b[offset:offset+4096],t[offset:offset+4096])-y[offset:offset+4096]
                        squared+=error.double().square().sum()
                    row[split+'_mse']=float(squared/len(y))*target_scale**2
                    row[split+'_relative_l2']=float((squared/y.double().square().sum()).sqrt())
                    row[split+'_rse']=float(squared/y.double().square().sum())
                assert all(np.isfinite(row[k]) for k in row if k!='event')
                row.update(batch_size=actual_batch,examples_seen=examples_seen,elapsed_seconds=time.perf_counter()-start)
                writer.writerow(row);cf.flush()
                mf.write('| '+' | '.join(str(row[k]) for k in fields)+' |\n');mf.flush()
                rows.append(row);return row

            chosen=record('initialized',0); best=chosen['train_mse']; best_step=0
            checkpoint=directory/(f'grade_{grade}_best.pt' if method=='multigrade' else 'best.pt')
            def save():
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                                grade=grade,selected_grade_step=best_step),checkpoint)
            save()
            for local_step in range(1,steps+1):
                for group in optimizer.param_groups: group['lr']=grade_learning_rate(initial_lr,segment_ratio,local_step,steps)
                b,t,old,y=cache['train']  # ONLY train labels enter gradients.
                if actual_batch<ntrain:
                    if position+actual_batch>len(order):
                        order=torch.randperm(ntrain,generator=sampler,device='cpu');position=0
                    indices=order[position:position+actual_batch];position+=actual_batch
                    batch_digest.update(indices.numpy().tobytes())
                    indices=indices.to(device)
                    b,t,old,y=(a[indices] for a in [b,t,old,y])
                else:batch_digest.update(b'full-training-set-in-original-order')
                optimizer.zero_grad(set_to_none=True)
                error=predict(b,t)-(y-old)
                loss=(error/units).square().mean() if residual_units else error.square().mean()
                loss.backward();optimizer.step();epoch+=1;examples_seen+=len(y)
                row=record('update',local_step)
                if local_step%checkpoint_interval==0 or local_step==steps:
                    if row['train_mse']<best: best=row['train_mse'];best_step=local_step;save()
                if local_step%1000==0 or local_step==steps:
                    print(f'{method} {epoch}/{total} train={row["train_mse"]:.5e} validation={row["validation_mse"]:.5e}',flush=True)
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'])
            chosen=record('restored_checkpoint',best_step)
            assert all(torch.equal(p,frozen[name]) for name,p in model.named_parameters() if name in frozen)
            assert all(torch.equal(buffer,frozen_buffers[name]) for name,buffer in model.named_buffers())
            grades.append(dict(grade=grade,selected_step=best_step,selected_metrics=chosen,frozen_unchanged=True,
                normalized_residual_rms=units,physical_residual_scale=target_scale*units,
                fixed_buffers_unchanged=True))
            del cache,frozen,optimizer
    assert [r['epoch'] for r in rows if r['event']=='update']==list(range(1,total+1))
    summary=dict(status='complete',parameters=sum(p.numel() for p in model.parameters()),
        grades=grades,selected_metrics=chosen,completed_updates=epoch,wall_seconds=time.perf_counter()-start,
        batch_size=actual_batch,examples_seen=examples_seen,equivalent_data_passes=examples_seen/ntrain,
        sample_sequence_sha256=batch_digest.hexdigest(),checkpoint_interval=checkpoint_interval)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def run_case(width,seed,budgets,traditional_lr,mg_lrs,ratio,validation,check=False,test_path=None,frozen_path=None,batch_size=0,normalization='none',traditional_decay_ratio=None):
    device=strict_cuda();torch.set_num_threads(4)
    data=load_development(device,validation,check)
    if test_path is not None:
        if frozen_path is None: raise ValueError('Final test requires frozen configuration')
        manifest=json.loads(test_path.with_name('manifest.json').read_text())
        assert manifest['role']=='final_test' and digest(test_path)==manifest['data_sha256']
        frozen=json.loads(frozen_path.read_text())
        assert manifest['frozen_config_sha256']==digest(frozen_path)
        assert manifest['seed']==frozen['final_test_seed'] and manifest['count']==frozen['final_test_count']
        with np.load(test_path) as archive:
            data['test']=tuple(torch.tensor(archive[k],dtype=torch.float32,device=device) for k in ['branch','trunk','reference'])
    pair,architecture=make_pair(width,seed)
    data,preprocessing=fit_transform(data,normalization)
    out=ROOT/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_fair_pendulum_'+('final' if test_path else 'tune')+('_check' if check else ''))
    out.mkdir(parents=True); sources=out/'sources';sources.mkdir()
    source_paths=[Path(__file__),ROOT/'src/fair_pendulum_models.py',ROOT/'src/models.py',
        ROOT/'src/pendulum_validation.py',ROOT/'src/pendulum_preprocessing.py',ROOT/'tools/run_pendulum_multigrade.py',
        ROOT/'tools/export_pendulum_comparison.py',ROOT/'src/plot_theme.py',ROOT/'tools/svg_epoch_data.py',
        ROOT/'tools/export_pendulum_time.py',
        ROOT/'tools/export_pendulum_rse.py',
        ROOT/'tools/gpu_deeponet.py']
    hashes={}
    for path in source_paths:
        shutil.copy2(path,sources/path.name);hashes[str(path)]=digest(path)
    for path in [SOURCE/'train.npz',validation]:hashes[str(path.resolve())]=digest(path)
    protocol=dict(status='running',methods=['traditional','multigrade'],seed=seed,width=width,
        grade_steps=budgets,total_updates=sum(budgets),traditional_lr=traditional_lr,mg_lrs=mg_lrs,decay_ratio=ratio,
        traditional_decay_ratio=ratio if traditional_decay_ratio is None else traditional_decay_ratio,
        batch_size=batch_size or len(data['train'][0]),normalization=normalization,normalization_fitted_on='training only',
        traditional_schedule='cosine',diagnostic_only=check,parameter_matched=True,
        parameters=architecture['parameters'],architecture=architecture,hashes=hashes,
        interpreter=sys.executable,device=str(device),validation_data=str(validation.resolve()),
        selection='Training MSE initial/every1000 updates/final within each training segment; validation ranks configurations only',
        fairness='Equal total learned parameters, longest affine path, data, full-batch update budget, optimizer family. Active parameters, internal/readout widths, total affine maps and runtime differ.',
        test_access='NONE: no test data loaded' if test_path is None else 'Final protocol frozen before fresh test generation; test is monitoring only',
        frozen_config_sha256=digest(frozen_path) if frozen_path else None,
        final_test_sha256=digest(test_path) if test_path else None)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(f'OUTPUT: {out}',flush=True)
    np.savez_compressed(out/'preprocessing.npz',**preprocessing)
    summaries={}
    try:
        for method in ['traditional','multigrade']:
            summaries[method]=train(method,pair.pop(method),data,budgets,
                [traditional_lr] if method=='traditional' else mg_lrs,
                protocol['traditional_decay_ratio'] if method=='traditional' else ratio,
                out,batch_size,seed+1000,float(preprocessing['target_scale']))
        assert summaries['traditional']['sample_sequence_sha256']==summaries['multigrade']['sample_sequence_sha256']
        assert summaries['traditional']['examples_seen']==summaries['multigrade']['examples_seen']
        (out/'summary.json').write_text(json.dumps(summaries,indent=2));export(out)
        from export_pendulum_time import export_time
        export_time(out)
        from export_pendulum_rse import export_rse
        export_rse(out)
        protocol['status']='complete'
        if test_path is not None:
            ratios={split:summaries['traditional']['selected_metrics'][split+'_mse']/max(summaries['multigrade']['selected_metrics'][split+'_mse'],1e-300) for split in ['train','test']}
            (out/'target_check.json').write_text(json.dumps(dict(required_ratio=20,ratios=ratios,
                achieved=all(r>=20 for r in ratios.values()) and not check,parameter_matched=True),indent=2))
    except BaseException:
        protocol['status']='interrupted_or_failed';raise
    finally: (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(f'COMPLETE: {out}',flush=True);return out,summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--grade-steps',type=int,nargs=4,default=[1000]*4)
    p.add_argument('--traditional-lr',type=float,default=.001)
    p.add_argument('--grade-lrs',type=float,nargs=4,default=[.001]*4)
    p.add_argument('--decay-ratio',type=float,default=.01)
    p.add_argument('--traditional-decay-ratio',type=float,help='Optional separate Traditional final/initial LR ratio; otherwise use decay-ratio')
    p.add_argument('--validation-data',type=Path,default=ROOT/'data/pendulum_validation_seed20261031_n5000/validation.npz')
    p.add_argument('--check',action='store_true',help='CUDA-only 8-update pipeline check')
    p.add_argument('--batch-size',type=int,default=0,help='0 means full training set; otherwise shared shuffled mini-batches')
    p.add_argument('--normalization',choices=['none','standard','whiten'],default='none')
    p.add_argument('--frozen-config',type=Path)
    p.add_argument('--final-test',type=Path)
    a=p.parse_args()
    if bool(a.frozen_config)!=bool(a.final_test):p.error('Both frozen-config and final-test are required together')
    if a.frozen_config:
        cfg=json.loads(a.frozen_config.read_text())
        for path,expected in cfg['implementation_sha256'].items():
            if digest(path)!=expected:raise ValueError('Implementation changed after protocol freeze: '+path)
        a.width=cfg['width'];a.seed=cfg['seed'];a.grade_steps=cfg['grade_steps'];a.traditional_lr=cfg['traditional_lr']
        a.grade_lrs=cfg['mg_lrs'];a.decay_ratio=cfg['decay_ratio'];a.validation_data=Path(cfg['validation_data'])
        a.batch_size=cfg.get('batch_size',0);a.normalization=cfg.get('normalization','none')
        a.traditional_decay_ratio=cfg.get('traditional_decay_ratio',a.decay_ratio)
        assert digest(SOURCE/'train.npz')==cfg['train_sha256'] and digest(a.validation_data)==cfg['validation_sha256']
    if a.width<2 or min(a.grade_steps)<1 or not all(np.isfinite(r) and r>0 for r in [a.traditional_lr,*a.grade_lrs]) or not 0<a.decay_ratio<=1:p.error('Invalid width, updates or rates')
    if a.batch_size<0:p.error('batch-size must be nonnegative')
    if a.traditional_decay_ratio is not None and not 0<a.traditional_decay_ratio<=1:p.error('Invalid traditional decay ratio')
    run_case(a.width,a.seed,[2]*4 if a.check else a.grade_steps,a.traditional_lr,a.grade_lrs,a.decay_ratio,
             a.validation_data,a.check,a.final_test,a.frozen_config,a.batch_size,a.normalization,a.traditional_decay_ratio)


if __name__=='__main__':main()
