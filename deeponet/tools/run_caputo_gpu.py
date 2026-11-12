"""GPU Caputo reproduction using cached data, original initialization and TF1 Adam.

This is an explicitly migrated implementation, not unchanged TensorFlow source.
Default is the 1D Legendre case with the same 30000-update local CPU budget.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
import torch
from caputo_gpu_data import ROOT,load_data
from gpu_deeponet import UnstackedDeepONet,TF1Adam,normalized_mse,strict_cuda


def active_seconds():
    value=ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
        raise ctypes.WinError()
    return value.value/1e7


class Tee:
    def __init__(self,terminal,log): self.terminal,self.log=terminal,log
    def write(self,text): self.terminal.write(text); self.log.write(text); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def close(self): self.flush()


def save_json(path,value):
    pending=path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value,indent=2),encoding="utf-8")
    os.replace(str(pending),str(path))


def save_checkpoint(path,value):
    pending=path.with_suffix(".pending.pt")
    torch.save(value,str(pending))
    os.replace(str(pending),str(path))


def execute(args,out,config):
    device=strict_cuda()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("GPU:",torch.cuda.get_device_name(device),"| CUDA",torch.version.cuda,flush=True)
    datasets,_,source_run=load_data(args.case)
    # Inputs and targets reside on CUDA, including every optimizer batch.
    train,test,example=[tuple(torch.as_tensor(a,device=device) for a in d) for d in datasets]
    model=UnstackedDeepONet([15,40,40,40],[2,40,40,40]).to(device)
    initial=ROOT/"data"/"gpu_parity"/"caputo_tf_seed42.npz"
    with np.load(str(initial)) as archive:
        model.load_tf_arrays(archive)
    assert sum(p.numel() for p in model.parameters())==7321
    optimizer=TF1Adam(model.tf_parameters())

    @torch.no_grad()
    def evaluate(dataset,archive=None):
        ub,qt,reference=dataset
        predictions=[]
        for start in range(0,len(ub),32768):
            predictions.append(model(ub[start:start+32768],qt[start:start+32768]))
        values=torch.cat(predictions)
        error=(values.double()-reference.double()).square()
        mse=error.mean().item()
        nmse=(error.sum()/reference.double().square().sum()).item()
        if not np.isfinite(mse+nmse): raise FloatingPointError("Nonfinite evaluation")
        if archive:
            np.savez_compressed(str(archive),reference=reference.cpu().numpy(),prediction=values.cpu().numpy())
        return dict(mse=mse,normalized_mse=nmse,relative_l2=float(np.sqrt(nmse)))

    if args.benchmark:
        ids=torch.arange(100000,device=device)
        batch=tuple(a[ids] for a in train)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss=normalized_mse(model(batch[0],batch[1]),batch[2]); loss.backward(); optimizer.step()
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss=normalized_mse(model(batch[0],batch[1]),batch[2]); loss.backward(); optimizer.step()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        report=dict(gpu=torch.cuda.get_device_name(device),batch_size=100000,
            timed_updates=20,seconds_per_update=elapsed/20,
            projected_30000_update_seconds=elapsed/20*30000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            scope="Fixed resident diagnostic batch; excludes shuffling, evaluation and checkpoint I/O",
            formal_training_started=False)
        save_json(out/"benchmark.json",report)
        save_json(ROOT/"results"/"gpu_environment"/"caputo_gpu_benchmark.json",report)
        print(json.dumps(report,indent=2)); print("CAPUTO_GPU_BENCHMARK_COMPLETE"); return

    rng=np.random.RandomState(args.seed)
    count=len(train[0]); batch_size=config["batch_size"]; batches=count//batch_size
    indices=np.arange(count); history=[]; completed=0; best=float("inf"); best_step=0
    previous_active=previous_wall=0.0
    if args.resume_run:
        snapshot=torch.load(str(out/"latest.pt"),map_location=device,weights_only=False)
        if snapshot["config_digest"]!=config["protocol_digest"]: raise ValueError("Resume protocol mismatch")
        model.load_state_dict(snapshot["model"]); optimizer.load_state_dict(snapshot["optimizer"])
        indices=snapshot["indices"]; rng.set_state(snapshot["numpy_rng"])
        torch.set_rng_state(snapshot["torch_rng"].cpu())
        torch.cuda.set_rng_state(snapshot["cuda_rng"].cpu(),device)
        completed=snapshot["step"]; history=snapshot["history"]
        best,best_step=snapshot["best_nmse"],snapshot["best_step"]
        previous_active=snapshot["training_active_seconds"]
        previous_wall=snapshot["training_wall_seconds"]

    torch.cuda.synchronize(); start_active,start_wall=active_seconds(),time.perf_counter()

    def record(step):
        nonlocal best,best_step
        tr,te=evaluate(train),evaluate(test)
        if tr["normalized_mse"]<best:
            best,best_step=tr["normalized_mse"],step
            save_checkpoint(out/"best.pt",dict(model=model.state_dict(),step=step,train=tr,test=te))
        history.append(dict(step=step,train=tr,test=te))
        save_json(out/"history.json",history)
        table=np.array([[r["step"],r["train"]["mse"],r["test"]["mse"],
                         r["train"]["normalized_mse"],r["test"]["normalized_mse"]] for r in history])
        np.savetxt(str(out/"loss.dat"),table,header="update train_mse test_mse train_normalized_mse test_normalized_mse")
        torch.cuda.synchronize()
        save_checkpoint(out/"latest.pt",dict(config_digest=config["protocol_digest"],
            model=model.state_dict(),optimizer=optimizer.state_dict(),indices=indices,
            numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
            step=step,history=history,best_nmse=best,best_step=best_step,
            training_active_seconds=previous_active+active_seconds()-start_active,
            training_wall_seconds=previous_wall+time.perf_counter()-start_wall))
        print("Update {} | train NMSE {:.9e} | test NMSE {:.9e} | test MSE {:.9e}".format(
            step,tr["normalized_mse"],te["normalized_mse"],te["mse"]),flush=True)

    if not args.resume_run: record(0)
    steps=2 if args.check else args.steps
    try:
        for step in range(completed+1,steps+1):
            position=(step-1)%batches
            if position==0: rng.shuffle(indices)
            ids=torch.as_tensor(indices[position*batch_size:(position+1)*batch_size],device=device)
            ub,qt,reference=(a[ids] for a in train)
            optimizer.zero_grad(set_to_none=True)
            loss=normalized_mse(model(ub,qt),reference)
            loss.backward(); optimizer.step()
            if step%(1 if args.check else 100)==0 or step==steps:
                if not torch.isfinite(loss).item(): raise FloatingPointError("Nonfinite training loss")
                record(step)
    except KeyboardInterrupt:
        print("Interrupted. Recover from latest.pt with --resume-run",out,flush=True)
        raise
    torch.cuda.synchronize()
    training_active=previous_active+active_seconds()-start_active
    training_wall=previous_wall+time.perf_counter()-start_wall
    selected=torch.load(str(out/"best.pt"),map_location=device,weights_only=False)
    model.load_state_dict(selected["model"])
    metrics=dict(case=args.case,gpu=torch.cuda.get_device_name(device),parameters=7321,
        best_step=best_step,optimizer_updates=steps,train=evaluate(train),
        test=evaluate(test,out/"test_predictions.npz"),example=evaluate(example,out/"example_predictions.npz"),
        training_active_seconds=training_active,training_wall_seconds=training_wall,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        timer="CUDA synchronized Windows unbiased interrupt time; excludes sleep, includes periodic evaluation/checkpointing",
        implementation="PyTorch CUDA migration with TF1-style Adam; not unchanged TensorFlow source")
    save_json(out/"metrics.json",metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    print("CAPUTO_GPU_CHECK_OK" if args.check else "CAPUTO_GPU_RUN_COMPLETE",flush=True)


def main():
    from caputo_gpu_data import RUNS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=tuple(RUNS),default="legendre")
    parser.add_argument("--steps",type=int,default=30000)
    parser.add_argument("--seed",type=int,choices=[42],default=42)
    parser.add_argument("--resume-run",type=Path)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--show-config",action="store_true")
    modes.add_argument("--check",action="store_true")
    modes.add_argument("--benchmark",action="store_true")
    args=parser.parse_args()
    if args.steps<=0: parser.error("--steps must be positive")
    if args.resume_run and (args.check or args.benchmark): parser.error("Diagnostics do not resume formal runs")
    validation=ROOT/"results"/"gpu_environment"/"caputo_gpu_validation.json"
    initial=ROOT/"data"/"gpu_parity"/"caputo_tf_seed42.npz"
    if not initial.exists(): parser.error("Run export_caputo_tf_reference.py using the old CPU interpreter first")
    source_run=ROOT/"results"/RUNS[args.case]
    source_config=json.loads((source_run/"config.json").read_text())
    sources=([source_run/"data"/(name+".npz") for name in ("train","test","test0")]
             if args.case=="legendre" else [ROOT/"data"/"caputo_polyfractonomial_disjoint"/"compact_data.mat"]
             if args.case=="polyfractonomial" else [source_run/"compact_data.npz"])
    fingerprints={}
    for path in sources:
        with path.open("rb") as stream:
            fingerprints[str(path)]=hashlib.file_digest(stream,"sha256").hexdigest()
    cfg=dict(case="caputo_1d_"+args.case+"_gpu",seed=args.seed,branch=[15,40,40,40],
        trunk=[2,40,40,40],activation="tanh",branch_output="linear",trunk_output="tanh",
        output="dot product plus scalar bias",parameters=7321,batch_size=100000,
        optimizer_updates=args.steps,optimizer="TF1Adam compatibility rule",learning_rate=.001,
        beta1=.9,beta2=.999,epsilon=1e-8,
        loss="mean((prediction-reference)^2) / mean(reference^2), no denominator epsilon",
        precision="float32; AMP and TF32 disabled",device="cuda:0",torch=torch.__version__,
        cuda_runtime=torch.version.cuda,interpreter=sys.executable,source_run=str(source_run),
        source_config=source_config,initialization="exact exported TF seed42/operation_seed12345 tensors",
        data_sha256=fingerprints,
        implementation_sha256={name:hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest()
            for name in ("gpu_deeponet.py","caputo_gpu_data.py")},
        initialization_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
        training_protocol="same 30000-update local CPU budget; not relabeled as 30000 epochs",
        checkpoint_selection="minimum full-training normalized MSE, evaluated every 100 updates",
        migration_caveat="Backend and device changed; parallel float32 arithmetic is not bit-identical",
        sampler="same NumPy seed42 repeatedly shuffled index array and contiguous batches as local CPU runner")
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    cfg["protocol_digest"]=digest
    if args.show_config: print(json.dumps(cfg,indent=2)); return
    if not validation.exists(): parser.error("Run validate_caputo_gpu.py before training")
    checked=json.loads(validation.read_text())
    if checked["torch"]!=torch.__version__: parser.error("GPU validation uses a different PyTorch version")
    if checked.get("fixture_sha256")!=cfg["initialization_sha256"]:
        parser.error("TF reference changed; rerun validate_caputo_gpu.py")
    if checked.get("implementation_sha256")!=cfg["implementation_sha256"]:
        parser.error("GPU implementation changed; rerun validate_caputo_gpu.py")
    strict_cuda()
    prefix="caputo_1d_{}_gpu_seed42_".format(args.case)
    if args.resume_run:
        out=args.resume_run.resolve()
        if out.parent!=(ROOT/"results").resolve() or not out.name.startswith(prefix):
            parser.error("Resume must name this GPU case under results")
        if (out/"metrics.json").exists(): parser.error("This run is already complete")
        old=json.loads((out/"config.json").read_text())
        if old["protocol_digest"]!=digest: parser.error("Resume configuration differs")
    else:
        if not (args.check or args.benchmark) and list((ROOT/"results").glob(prefix+"*")):
            parser.error("Existing GPU run found; inspect or resume it")
        base=ROOT/("tmp" if args.check or args.benchmark else "results")
        out=base/(prefix+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        out.mkdir(parents=True)
        save_json(out/"config.json",cfg)
    with (out/"run.log").open("a" if args.resume_run else "w",encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            print("Interpreter:",sys.executable,"\nResults:",out,flush=True)
            execute(args,out,cfg)


if __name__=="__main__": main()
