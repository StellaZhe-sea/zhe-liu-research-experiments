"""Every-update accuracy/time and runtime SVGs from saved pendulum metrics only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import runtime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_theme import apply_theme
from svg_epoch_data import embed_epoch_data
from verify_svg import validate


def export_time(run):
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT):raise ValueError('Run must be inside project')
    protocol=json.loads((run/'protocol.json').read_text())
    summaries=json.loads((run/'summary.json').read_text())
    histories={};hashes={}
    for method in ['traditional','multigrade']:
        path=run/method/'metrics.csv';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as handle:
            rows=list(csv.DictReader(handle))
        rows=[{k:float(v) for k,v in r.items() if k!='event'} for r in rows if r.get('event','update')=='update']
        assert [int(r['epoch']) for r in rows]==list(range(1,protocol['total_updates']+1))
        assert all(np.isfinite(v) for r in rows for v in r.values())
        times=np.array([r['elapsed_seconds'] for r in rows])
        assert np.all(np.diff(times)>=0) and times[0]>=0
        assert summaries[method]['wall_seconds']>=times[-1]
        histories[method]=rows
    out=run/'time_review';out.mkdir(exist_ok=True)
    palette=apply_theme('navy')
    labels={'traditional':'DeepONet','multigrade':'MultiGrade DeepONet'}
    time_note='Includes metric evaluation, grade setup and checkpoint I/O; excludes final figure export.'
    def save(fig,name,metric):
        path=out/name
        fig.savefig(path,format='svg');fig.savefig(path.with_suffix('.png'),dpi=130)
        plt.close(fig)
        embed_epoch_data(path,histories,dict(metric=metric,timing_definition=time_note,
            timing_scope='Recorded model-loop wall time, not isolated CUDA kernel time',
            current_weight_curves=True,selected_sources=protocol.get('selected_sources')))
        validate(path)
    for key,name,title in [('train_mse','01_training_mse_vs_time.svg','Training MSE versus elapsed time'),
                           ('validation_mse','02_validation_mse_vs_time.svg','Validation MSE versus elapsed time')]:
        fig,ax=plt.subplots(figsize=(10.5,6),layout='constrained')
        for method,rows in histories.items():
            ax.semilogy([r['elapsed_seconds'] for r in rows],[max(r[key],1e-300) for r in rows],
                color=palette[method],ls='-' if method=='traditional' else '-',lw=1.6,label=labels[method])
        for boundary in np.cumsum(protocol['grade_steps'])[:-1]:
            row=histories['multigrade'][int(boundary)-1]
            ax.axvline(row['elapsed_seconds'],ls='-',lw=.8,color=palette['multigrade'],alpha=.4)
        ax.set(xlabel='Elapsed model-loop time (seconds)',ylabel='MSE in physical units',title=title)
        ax.grid(alpha=.2);ax.legend()
        fig.supxlabel('All optimizer updates retained; solid vertical lines mark completed MultiGrade budgets.\n'+time_note,fontsize=8)
        save(fig,name,key)
    fig,ax=plt.subplots(figsize=(10.5,5.8),layout='constrained')
    methods=['traditional','multigrade'];seconds=[float(summaries[m]['wall_seconds']) for m in methods]
    bars=ax.bar([labels[m] for m in methods],seconds,color=[palette[m] for m in methods],width=.55)
    bars[0].set_hatch('//')
    for bar,value in zip(bars,seconds):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f} s',ha='center',va='bottom')
    ax.set(ylim=(0,max(seconds)*1.2),ylabel='Measured model-loop wall time (seconds)',
           title=f'{protocol["total_updates"]:,} updates per model | DeepONet/MultiGrade time = {seconds[0]/seconds[1]:.2f}x')
    ax.grid(axis='y',alpha=.2)
    fig.supxlabel(time_note+'\nSingle timed run per selected configuration; accuracy must be assessed separately.',fontsize=8)
    save(fig,'03_total_runtime.svg','Total measured model-loop wall time')
    fig,ax=plt.subplots(figsize=(10.5,5.8),layout='constrained')
    for method,rows in histories.items():
        ax.plot([r['epoch'] for r in rows],[r['elapsed_seconds'] for r in rows],
            color=palette[method],ls='-' if method=='traditional' else '-',label=labels[method])
    ax.set(xlabel='Optimizer update (every update recorded)',ylabel='Elapsed model-loop time (seconds)',title='Time used to complete the update budget')
    ax.grid(alpha=.2);ax.legend();fig.supxlabel(time_note,fontsize=8)
    save(fig,'04_time_vs_update.svg','Elapsed time at every update')
    # Fixed thresholds avoid selecting an accuracy target solely to favor one method.
    thresholds={'train_mse':[1e-3,3e-4,1e-4],'validation_mse':[5e-3,3e-3,2e-3,1e-3]}
    crossings=[]
    for metric,values in thresholds.items():
        for value in values:
            for method,rows in histories.items():
                hit=next((r for r in rows if r[metric]<=value),None)
                crossings.append(dict(method=method,metric=metric,threshold=value,
                    reached=hit is not None,first_update=int(hit['epoch']) if hit else None,
                    first_seconds=hit['elapsed_seconds'] if hit else None))
    with (out/'time_to_threshold.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(crossings[0]));writer.writeheader();writer.writerows(crossings)
    paired=[]
    for i in range(protocol['total_updates']):
        row={'update':i+1}
        for method in methods:
            for key in ['elapsed_seconds','train_mse','validation_mse']:
                row[method+'_'+key]=histories[method][i][key]
        paired.append(row)
    with (out/'every_update_time.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    with (out/'every_update_time.md').open('w') as handle:
        fields=list(paired[0]);handle.write('# Every-update timing and accuracy\n\n'+time_note+'\n\n')
        handle.write('| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n')
        for row in paired:handle.write('| '+' | '.join(str(row[k]) for k in fields)+' |\n')
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    audit=dict(source_run=str(run),updates_per_model=protocol['total_updates'],source_metrics_unchanged=True,
        source_hashes=hashes,wall_seconds=dict(zip(methods,seconds)),time_ratio=seconds[0]/seconds[1],
        smoothing=False,subsampling=False,timing_definition=time_note,
        threshold_note='First crossing only; not sustained accuracy and not used to select training checkpoints',
        interpretation='Shorter time at equal updates is not by itself better accuracy or faster time to a common accuracy.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(f'TIME REPORT: {out}',flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    export_time(parser.parse_args().run)
