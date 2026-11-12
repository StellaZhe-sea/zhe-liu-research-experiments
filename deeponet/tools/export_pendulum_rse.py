"""Uncentered relative squared error and relative-L2 companion, every update."""
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


def export_rse(run):
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT):raise ValueError('Run must be inside project')
    protocol=json.loads((run/'protocol.json').read_text())
    history={};source_hashes={}
    for method in ('traditional','multigrade'):
        path=run/method/'metrics.csv'
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as f:rows=list(csv.DictReader(f))
        history[method]=[{k:float(v) for k,v in row.items() if k!='event'}
                         for row in rows if row.get('event','update')=='update']
        assert [r['epoch'] for r in history[method]]==list(range(1,protocol['total_updates']+1))
    splits=[s for s in ('train','validation','test') if s+'_relative_l2' in history['traditional'][0]]
    for rows in history.values():
        for row in rows:
            for split in splits:
                value=row[split+'_relative_l2']**2
                if split+'_rse' in row:np.testing.assert_allclose(row[split+'_rse'],value,rtol=1e-12,atol=1e-15)
                else:row[split+'_rse']=value
    out=run/'rse_review';out.mkdir(exist_ok=True)
    palette=apply_theme('navy')
    labels={'traditional':'DeepONet','multigrade':'MultiGrade DeepONet'}
    definition='RSE = sum((prediction - target)^2) / sum(target^2); relative L2 = sqrt(RSE).'
    for split in splits:
        for axis in ('epoch','elapsed_seconds'):
            fig,ax=plt.subplots(figsize=(10.5,5.8),layout='constrained')
            for method,rows in history.items():
                ax.semilogy([r[axis] for r in rows],[max(r[split+'_rse'],1e-300) for r in rows],
                    color=palette[method],ls='-' if method=='traditional' else '-',label=labels[method],lw=1.6)
            ax.set(title=f'{split.title()} relative squared error',ylabel='RSE (uncentered squared relative L2)',
                   xlabel='Optimizer update' if axis=='epoch' else 'Elapsed model-loop wall time (seconds)')
            ax.grid(alpha=.2);ax.legend()
            fig.supxlabel(definition+'\nEvery update retained; no smoothing.'+
                          (' Time includes evaluation and checkpoint I/O.' if axis=='elapsed_seconds' else ''),fontsize=8)
            path=out/f'{split}_rse_vs_{"update" if axis=="epoch" else "time"}.svg'
            fig.savefig(path,format='svg');fig.savefig(path.with_suffix('.png'),dpi=120);plt.close(fig)
            embed_epoch_data(path,history,dict(metric=split+'_rse',definition=definition,x=axis));validate(path)
    merged=[]
    for i in range(protocol['total_updates']):
        row={'epoch':i+1}
        for method,rows in history.items():
            row[method+'_seconds']=rows[i]['elapsed_seconds']
            for split in splits:
                for metric in ('mse','rse','relative_l2'):
                    row[method+'_'+split+'_'+metric]=rows[i][split+'_'+metric]
        merged.append(row)
    fields=list(merged[0])
    with (out/'every_update_rse.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(merged)
    with (out/'every_update_rse.md').open('w') as f:
        f.write('# Every-update relative errors\n\n'+definition+'\n\n| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n')
        for row in merged:f.write('| '+' | '.join(str(row[k]) for k in fields)+' |\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
    (out/'audit.json').write_text(json.dumps(dict(definition=definition,splits=splits,
        updates_per_model=len(merged),smoothing=False,subsampling=False,source_hashes=source_hashes,
        source_metrics_unchanged=True),indent=2))
    print(f'RSE REPORT: {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    export_rse(parser.parse_args().run)
