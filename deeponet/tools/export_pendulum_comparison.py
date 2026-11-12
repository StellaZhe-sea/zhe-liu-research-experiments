"""Navy two-model pendulum report, with every measured update embedded in SVG."""
import argparse
import csv
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


def export(run):
    run=Path(run).resolve()
    if not run.is_relative_to(ROOT):
        raise ValueError('Run must be inside the project')
    protocol=json.loads((run/'protocol.json').read_text())
    histories={}
    for method in ['traditional','multigrade']:
        with (run/method/'metrics.csv').open() as handle:
            raw=list(csv.DictReader(handle))
        table=[line for line in (run/method/'metrics.md').read_text().splitlines() if line.startswith('|')][2:]
        assert len(table)==len(raw)
        for r,line in zip(raw,table):
            cells=[cell.strip() for cell in line.strip('|').split('|')]
            assert len(cells)==len(r)
            for (key,value),cell in zip(r.items(),cells):
                if key=='event': assert cell==value
                else: np.testing.assert_allclose(float(cell),float(value),rtol=1e-11,atol=1e-12)
        histories[method]=[{k:float(v) for k,v in r.items() if k!='event'}
                           for r in raw if r.get('event','update')=='update']
        assert [r['epoch'] for r in histories[method]]==list(range(1,protocol['total_updates']+1))
        assert all(np.isfinite(v) for r in histories[method] for v in r.values())
    rates_match=[r['learning_rate'] for r in histories['traditional']]==[r['learning_rate'] for r in histories['multigrade']]
    if protocol.get('traditional_schedule','matched')=='matched': assert rates_match
    figures=run/'figures'; figures.mkdir(exist_ok=True)
    review=run/'epoch_review'; review.mkdir(exist_ok=True)
    palette=apply_theme('navy')
    labels={'traditional':'DeepONet','multigrade':'MultiGrade DeepONet'}
    specs=[('01_training_relative_l2','train_relative_l2','Training relative L2 (global)'),
           ('02_training_mse','train_mse','Training MSE')]
    has_test=all('test_mse' in rows[0] for rows in histories.values())
    if has_test:
        specs.extend([('03_test_relative_l2','test_relative_l2','Test relative L2 (global)'),
                      ('04_test_mse','test_mse','Test MSE')])
    if all('validation_mse' in rows[0] for rows in histories.values()):
        specs.extend([('05_validation_mse','validation_mse','Validation MSE'),
                      ('06_validation_relative_l2','validation_relative_l2','Validation relative L2 (global)')])
    boundaries=np.cumsum(protocol['grade_steps'])[:-1]

    def draw(ax,key,label):
        for method,rows in histories.items():
            ax.semilogy([r['epoch'] for r in rows],np.maximum([r[key] for r in rows],1e-300),
                        '-' if method=='traditional' else '-',color=palette[method],
                        linewidth=1.7 if method=='traditional' else 2.7,label=labels[method])
        for boundary in boundaries:
            ax.axvline(boundary+.5,color=palette['foreground'],alpha=.22,linewidth=.8,linestyle='-')
        ax.set(xlabel='Optimizer update (every update recorded)',ylabel=label)
        ax.grid(alpha=.18); ax.legend(framealpha=.8)

    def save(fig,path,description):
        fig.savefig(path,format='svg')
        fig.savefig(path.with_suffix('.png'),dpi=120)
        plt.close(fig)
        embed_epoch_data(path,histories,{'metric':description,'diagnostic_only':protocol['diagnostic_only'],
            'grade_boundaries':boundaries.tolist(),'curves':'post-update current weights, not best-so-far',
            'relative_l2_definition':'sqrt(sum(error^2)/sum(reference^2))'})
        validate(path)

    for name,key,label in specs:
        fig,ax=plt.subplots(figsize=(9.4,5.8),layout='constrained')
        draw(ax,key,label)
        if protocol['diagnostic_only']: ax.set_title('Pipeline check only: reduced data and update budgets')
        save(fig,figures/(name+'.svg'),key)
    nrows=len(specs)//2
    fig,axes=plt.subplots(nrows,2,figsize=(14,4.5*nrows),layout='constrained')
    for ax,(_,key,label) in zip(axes.flat,specs): draw(ax,key,label)
    fig.suptitle(('Pipeline check only; ' if protocol['diagnostic_only'] else '')+
                 f"{protocol['total_updates']:,} updates per model; all points retained\n"+
                 ('Equal total parameter counts; ' if protocol.get('parameter_matched') else 'Parameters differ; ')+
                 'vertical lines mark MultiGrade boundaries')
    save(fig,review/'every_epoch.svg','Training/test MSE and global relative L2')
    merged=[]
    for index in range(protocol['total_updates']):
        row={'epoch':index+1}
        for method,rows in histories.items():
            row.update({method+'_'+k:v for k,v in rows[index].items() if k!='epoch'})
        for split in (['train','test'] if has_test else ['train']):
            row[split+'_traditional_over_multigrade']=histories['traditional'][index][split+'_mse']/max(histories['multigrade'][index][split+'_mse'],1e-300)
        merged.append(row)
    fields=list(merged[0])
    with (review/'every_epoch.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(merged)
    with (review/'every_epoch.md').open('w') as handle:
        handle.write('# Every update: Traditional and MultiGrade DeepONet\n\n'
                     '| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n')
        for row in merged: handle.write('| '+' | '.join(str(row[k]) for k in fields)+' |\n')
    (review/'audit.json').write_text(json.dumps(dict(updates_per_model=len(merged),
        methods=list(histories),missing_updates=0,smoothing=False,subsampling=False,
        identical_learning_rate_sequence=rates_match,source_csv_markdown_agreement=True),indent=2))
    (figures/'README.md').write_text('All optimizer updates are plotted and embedded numerically in each SVG.\n'
        'Dark navy style follows results/20261031_175333_937660_multiscale_reaction/figures.\n'
        'Validation curves appear only when independent validation was actually recorded. No MAE or spectral curves are fabricated.\n'
        'Tuning runs omit all test data and test figures. Final-test curves are generated only by a frozen final protocol.\n'
        'Initialization/restoration events remain in the MultiGrade source metrics; plotted rows are updates only.\n'
        'See ../epoch_review/every_epoch.csv and every_epoch.md for paired data.\n')
    print(f'Validated {len(specs)+1} every-update SVGs: {run}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    export(parser.parse_args().run)
