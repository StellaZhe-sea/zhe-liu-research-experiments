"""Paired CUDA migrations of Appendix Table S6 unstacked bias/no-bias models."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from gpu_deeponet import UnstackedDeepONet,TF1Adam,strict_cuda
from run_caputo_gpu import active_seconds,save_json,save_checkpoint,Tee

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'antiderivative_gpu_seed42'


def make_model(variant,device):
    model=UnstackedDeepONet([100,40,40],[1,40,40],activation='relu',
                           use_bias=variant=='bias').to(device)
    with np.load(str(DATA/('tf_'+variant+'.npz'))) as fixture:
        model.load_tf_arrays(fixture)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==(7401 if variant=='bias' else 7360)
    return model


def validate(device):
    report={}
    for variant in ('bias','no_bias'):
        model=make_model(variant,device)
        optimizer=TF1Adam(model.tf_parameters())
        rows=[]
        with np.load(str(DATA/('tf_'+variant+'.npz'))) as fixture:
            u,x,y=[torch.as_tensor(fixture[k],device=device) for k in ('branch','trunk','target')]
            for step in range(11):
                if step in (0,1,10):
                    pred=model(u,x)
                    loss=(pred-y).square().mean()
                    pred_error=float(np.max(np.abs(pred.detach().cpu().numpy()-fixture['step%d_prediction'%step])))
                    loss_error=abs(loss.item()-float(fixture['step%d_loss'%step]))
                    param_error=max(float(np.max(np.abs(p.detach().cpu().numpy()-fixture['step%d_param_%02d'%(step,i)])))
                                    for i,p in enumerate(model.tf_parameters()))
                    assert pred_error<2e-4 and loss_error<2e-5 and param_error<2e-5,(variant,step,pred_error,loss_error,param_error)
                    rows.append(dict(step=step,prediction_max_absolute_error=pred_error,
                                     loss_absolute_error=loss_error,parameter_max_absolute_error=param_error))
                if step<10:
                    optimizer.zero_grad(set_to_none=True)
                    (model(u,x)-y).square().mean().backward()
                    optimizer.step()
        # The no-bias flag must ignore even explicitly nonzero disabled biases.
        if variant=='no_bias':
            with torch.no_grad():
                before=model(u,x).clone()
                model.output_bias.fill_(7); model.branch_biases[-1].fill_(5)
                assert torch.equal(before,model(u,x))
                model.output_bias.zero_(); model.branch_biases[-1].zero_()
        # Exercise an actual 10000-pair CUDA batch and exact optimizer recovery.
        ub=u.repeat(40,1)[:10000]; xb=x.repeat(40,1)[:10000]; yb=y.repeat(40,1)[:10000]
        snapshot=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        def one_update():
            optimizer.zero_grad(set_to_none=True)
            loss=(model(ub,xb)-yb).square().mean()
            assert torch.isfinite(loss).item()
            loss.backward(); optimizer.step()
        one_update()
        expected=[p.detach().clone() for p in model.tf_parameters()]
        model.load_state_dict(snapshot['model']); optimizer.load_state_dict(snapshot['optimizer'])
        one_update()
        assert all(torch.equal(a,b) for a,b in zip(expected,model.tf_parameters()))
        report[variant]=rows
    torch.cuda.synchronize()
    return dict(gpu=torch.cuda.get_device_name(device),torch=torch.__version__,
                reference='Official DeepXDE 0.11.2 TF OpNN; ten diagnostic updates per model',
                full_batch_cuda_and_optimizer_recovery='Passed both variants with 10000 diagnostic pairs',checks=report)


def execute(args,out,device,config):
    with np.load(str(DATA/'operator_data.npz')) as a:
        datasets={name:tuple(torch.as_tensor(a[name+'_'+k],device=device) for k in ('branch','trunk','target'))
                  for name in ('train','test','x','sinx','sin2x','xsin2x')}
    train,test=datasets['train'],datasets['test']
    assert [len(train[0]),len(test[0])]==[10000,100000]
    model=make_model(args.variant,device)
    optimizer=TF1Adam(model.tf_parameters())
    history=[]; best=float('inf'); best_step=0; completed=0; previous_active=previous_wall=0.
    if args.resume_run:
        state=torch.load(str(out/'latest.pt'),map_location=device,weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        completed=state['step']; history=state['history']; best=state['best']; best_step=state['best_step']
        previous_active=state['active']; previous_wall=state['wall']

    @torch.no_grad()
    def predict(dataset):
        u,x,_=dataset
        return torch.cat([model(u[i:i+16384],x[i:i+16384]) for i in range(0,len(u),16384)])

    @torch.no_grad()
    def final_metric(name):
        dataset=datasets[name]; pred=predict(dataset); ref=dataset[2]
        error=(pred.double()-ref.double()).square().reshape(-1)
        result=dict(mse=error.mean().item(),relative_l2=torch.sqrt(error.sum()/ref.double().square().sum()).item())
        if name=='test':
            # Match official utils.py: removes ceil(Q/1000) largest errors.
            result['trimmed_mse']=torch.sort(error).values[:(-len(error))//1000].mean().item()
        np.savez_compressed(str(out/(name+'_predictions.npz')),
                            coordinate=dataset[1].cpu().numpy(),reference=ref.cpu().numpy(),
                            prediction=pred.cpu().numpy())
        return result

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(); start_active=active_seconds(); start_wall=time.perf_counter()

    def record(step):
        nonlocal best,best_step
        with torch.no_grad():
            tr=(model(train[0],train[1])-train[2]).square().mean().item()
            # Preserve original monitored test subset, while final metrics use all 100000.
            subset=tuple(a[:65535] for a in test)
            te=(predict(subset)-subset[2]).square().mean().item()
        if not np.isfinite(tr+te): raise FloatingPointError('Nonfinite MSE')
        if tr<best:
            best,best_step=tr,step
            save_checkpoint(out/'best.pt',dict(model=model.state_dict(),step=step))
        history.append([step,tr,te])
        np.savetxt(str(out/'loss.dat'),np.asarray(history),header='update train_mse monitored_test_mse_65535')
        torch.cuda.synchronize()
        save_checkpoint(out/'latest.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            step=step,history=history,best=best,best_step=best_step,
            active=previous_active+active_seconds()-start_active,
            wall=previous_wall+time.perf_counter()-start_wall))
        print('Step {} | train MSE {:.9e} | test MSE (65535) {:.9e}'.format(step,tr,te),flush=True)

    if not args.resume_run: record(0)
    for step in range(completed+1,50001):
        optimizer.zero_grad(set_to_none=True)
        loss=(model(train[0],train[1])-train[2]).square().mean()
        loss.backward(); optimizer.step()
        if step%1000==0: record(step)
    torch.cuda.synchronize()
    seconds=previous_active+active_seconds()-start_active
    wall=previous_wall+time.perf_counter()-start_wall
    selected=torch.load(str(out/'best.pt'),map_location=device,weights_only=False)
    model.load_state_dict(selected['model'])
    metrics=dict(case='antiderivative',variant='unstacked_'+args.variant,
        gpu=torch.cuda.get_device_name(device),seed=42,optimizer_updates=50000,best_step=best_step,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train=final_metric('train'),test=final_metric('test'),
        examples={name:final_metric(name) for name in ('x','sinx','sin2x','xsin2x')},
        training_active_seconds=seconds,training_wall_seconds=wall,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        timer='CUDA synchronized Windows unbiased interrupt time; excludes sleep, includes evaluation/checkpointing',
        implementation='PyTorch CUDA migration with TF1Adam; not unchanged TensorFlow source')
    save_json(out/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    print('ANTIDERIVATIVE_GPU_RUN_COMPLETE',args.variant,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['bias','no_bias'],required=True)
    parser.add_argument('--seed',type=int,choices=[42],default=42)
    parser.add_argument('--check',action='store_true',help='Ten small TF/CUDA parity updates per variant; no formal training')
    parser.add_argument('--resume-run',type=Path)
    args=parser.parse_args() #Key Code reading the code in Powershell
    if args.check and args.resume_run: parser.error('Diagnostics cannot resume')
    torch.set_num_threads(4)
    device=strict_cuda()
    if not all((DATA/('tf_'+v+'.npz')).exists() for v in ('bias','no_bias')):
        subprocess.run([str(ROOT/'DeepONet_Official'/'python.exe'),
                        str(ROOT/'tools'/'prepare_antiderivative_gpu.py')],check=True)
    report=validate(device)
    destination=ROOT/'results'/'gpu_environment'
    destination.mkdir(parents=True,exist_ok=True)
    save_json(destination/'antiderivative_gpu_validation.json',report)
    print('ANTIDERIVATIVE_GPU_PARITY_OK',json.dumps(report),flush=True)
    if args.check: return
    prefix='antiderivative_unstacked_'+args.variant+'_gpu_seed42_'
    if args.resume_run:
        out=args.resume_run.resolve()
        if out.parent!=(ROOT/'results').resolve() or not out.name.startswith(prefix):
            parser.error('Resume must point to this variant under results')
        if (out/'metrics.json').exists(): parser.error('This run is already complete')
    elif list((ROOT/'results').glob(prefix+'*')):
        parser.error('Existing GPU run found; inspect it or use --resume-run')
    if not (DATA/'operator_data.npz').exists():
        print('Preparing shared full dataset on CPU using original solver; training will run on CUDA.',flush=True)
        subprocess.run([str(ROOT/'DeepONet_Official'/'python.exe'),
                        str(ROOT/'tools'/'prepare_antiderivative_gpu.py'),'--with-data'],check=True)
    config=dict(case='antiderivative',variant=args.variant,seed=42,seed_note='Local fixed seed, not a published seed',
        branch=[100,40,40],trunk=[1,40,40],activation='relu',
        branch_output='linear',trunk_output='relu',use_bias=args.variant=='bias',
        bias_semantics='Flag controls final branch bias and scalar output bias only',
        train_functions=10000,test_functions=100000,points_per_function=1,
        optimizer_updates=50000,batch_size=10000,optimizer='TF1Adam',learning_rate=.001,
        loss='mean((prediction-reference)^2), no normalization, no PDE residual',
        precision='float32; AMP/TF32 disabled',checkpoint_selection='Minimum training MSE every 1000 updates',
        initialization='Same exported official Glorot-normal seed42 weights for shared parameters',
        torch=torch.__version__,interpreter=sys.executable,
        paper='Appendix Tables S2, S3, S6; paired local GPU migration, not bit-identical TF training',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [DATA/'operator_data.npz',DATA/('tf_'+args.variant+'.npz'),
                 ROOT/'tools'/'gpu_deeponet.py',Path(__file__)]})
    if args.resume_run:
        if json.loads((out/'config.json').read_text())!=config: parser.error('Resume configuration or file hash changed')
    else:
        out=ROOT/'results'/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(); save_json(out/'config.json',config)
    with (out/'run.log').open('a',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            print('Interpreter:',sys.executable,'\nGPU:',torch.cuda.get_device_name(device),'\nResults:',out,flush=True)
            execute(args,out,device,config)


if __name__=='__main__': main()
