"""Recompute record-level PAC capability summaries and diagnostic figures."""
import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def read(root, name):
    with gzip.open(root/name/'records.json.gz', 'rt') as f:
        return json.load(f)


def rate_interval(k, n):
    z = 1.959963984540054
    p = k/n
    scale = 1+z*z/n
    center = (p+z*z/(2*n))/scale
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return [center-half, center+half]


def paired_difference(rows, a, b):
    keyed = {(r['group'], r['seed'], r['variant']): r for r in rows}
    rng = np.random.default_rng(3329)
    means, boot = [], []
    for group in sorted({r['group'] for r in rows}):
        ids = sorted({r['seed'] for r in rows if r['group'] == group})
        diff = np.array([keyed[group, seed, a]['curve_rmse']-
                         keyed[group, seed, b]['curve_rmse'] for seed in ids])
        means.append(float(diff.mean()))
        boot.append(diff[rng.integers(len(diff), size=(2000, len(diff)))].mean(1))
    return dict(mean_reduction=float(np.mean(means)),
                paired_record_bootstrap_ci95=np.quantile(np.mean(boot, axis=0), [.025, .975]).tolist(),
                unit='Independent signal record; resampled within fixed scenario, then macro-averaged')


def analyze(root):
    recovery = read(root, 'recovery-confirm')
    calibration = read(root, 'null-calibration')
    null = read(root, 'null-confirm')
    grid = read(root, 'grid-confirm')
    local = read(root, 'local-confirm')
    summary = dict(protocol='capabilities-v2', detection={}, recovery={}, grid={}, local={})
    names=sorted({r['variant'] for r in recovery})
    for name in names:
        cal = [r['record_score'] for r in calibration if r['variant'] == name]
        threshold = float(np.quantile(cal, .95, method='higher'))
        negatives = [r for r in null if r['variant'] == name]
        count = sum(r['record_score'] > threshold for r in negatives)
        groups = {}
        for group in sorted({r['group'] for r in recovery}):
            rows = [r for r in recovery if r['variant'] == name and r['group'] == group]
            groups[group] = dict(n=len(rows),
                curve_rmse=float(np.mean([r['curve_rmse'] for r in rows])),
                true_pair_detection=float(np.mean([r['pair_score'] > threshold for r in rows])),
                record_detection=float(np.mean([r['record_score'] > threshold for r in rows])),
                valid_fraction=float(np.mean([r['valid'] for r in rows])))
        summary['detection'][name] = dict(threshold=threshold, calibration_records=len(cal),
            null_records=len(negatives), false_positives=count, observed_fpr=count/len(negatives),
            fpr_wilson_ci95=rate_interval(count,len(negatives)))
        summary['recovery'][name] = groups
        summary['grid'][name] = {}
        for seconds in (5, 10, 20):
            for shape in ('single', 'double'):
                rows = [r for r in grid if r['variant'] == name and r['seconds'] == seconds and r['shape'] == shape]
                valid = [r for r in rows if r['valid']]
                representable = [r for r in rows if r['valid'] and r['truth_supported']]
                summary['grid'][name][f'{seconds}s/{shape}'] = dict(
                    n=len(rows), valid_fraction=len(valid)/len(rows),
                    truth_supported_and_valid_fraction=len(representable)/len(rows),
                    curve_rmse_all=float(np.mean([r['curve_rmse'] for r in rows])),
                    curve_rmse_supported_valid=float(np.mean([r['curve_rmse'] for r in representable])) if representable else None,
                    nearest_anchor_top1=float(np.mean([r['top1_correct'] for r in rows])))
    for name in ('global','local'):
        rows = [r for r in local if r['variant'] == name]
        summary['local'][name] = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
            for key in ('all_anchor_error','local_valid_error','fallback_error','local_fraction')
            if any(r[key] is not None for r in rows)}
    if 'adaptive' in names:
        summary['paired_recovery_curve_to_adaptive'] = paired_difference(recovery,'curve','adaptive')
    else:
        summary['paired_recovery_h1_to_curve'] = paired_difference(recovery,'h1','curve')
    summary['notes'] = [
        'Detection uses independent-record calibration and confirmation; scores are not p-values.',
        'Nearest-anchor top-1 is diagnostic: overlapping amplitude filters need not identify a unique carrier.',
        'Unsupported second harmonics remain in all-record errors and are separately marked.',
        'H1 is an internal expressivity control, not a strong competing PAC estimator.']
    (root/'capability-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return recovery, grid, summary


def plots(root, recovery, grid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    variants = [('h1','#747474','H1 control'),('curve','#3b6d9a','Fixed H1+H2'),
                ('adaptive','#c47743','Cycle-adaptive (experimental)')]
    variants=[v for v in variants if any(r['variant']==v[0] for r in recovery)]
    fig, axes = plt.subplots(1,3,figsize=(10.4,2.8),layout='constrained')
    for ax, prefix, points, xlabel in [
        (axes[0],'depth/',[0.,.02,.05,.1,.25,.5],'True modulation depth'),
        (axes[1],'snr/',[0.,5.,10.,20.],'Target-band SNR (dB)')]:
        for name,color,label in variants:
            means,errors=[],[]
            for value in points:
                rows=[r['curve_rmse'] for r in recovery if r['variant']==name and r['group']==prefix+str(value)]
                means.append(np.mean(rows));errors.append(np.std(rows,ddof=1)/math.sqrt(len(rows))*1.96)
            ax.errorbar(points,means,yerr=errors,color=color,label=label,lw=1.4,marker='o',ms=3,capsize=2)
        ax.set(xlabel=xlabel,ylabel='Curve RMSE',ylim=(0,None))
        ax.grid(axis='y',alpha=.15)
    groups=['depth/0.5','double','mixed','nonlinear']
    for offset,(name,color,label) in enumerate(variants):
        data=[[r['curve_rmse'] for r in recovery if r['variant']==name and r['group']==group] for group in groups]
        axes[2].bar(np.arange(4)+(offset-(len(variants)-1)/2)*.22,[np.mean(x) for x in data],width=.2,color=color,label=label)
    axes[2].set(xticks=np.arange(4),xticklabels=['Single','Double','Mixed','Nonlinear'],ylabel='Curve RMSE',ylim=(0,None))
    axes[0].legend(frameon=False,fontsize=7)
    for ax,title in zip(axes,['Strength sweep','Noise sensitivity','Shape recovery']):ax.set_title(title)
    for suffix in ('pdf','png'):fig.savefig(root/f'recovery.{suffix}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9.2,4.7),layout='constrained')
    for col,seconds in enumerate((5,10,20)):
        rows=[r for r in grid if r['variant']=='curve' and r['seconds']==seconds and r['shape']=='single']
        error=np.zeros((6,5));coverage=error.copy()
        for i in range(6):
            for j in range(5):
                cell=[r for r in rows if r['source_index']==i and r['target_index']==j]
                error[i,j]=np.mean([r['curve_rmse'] for r in cell])
                coverage[i,j]=np.mean([r['valid'] for r in cell])
        im0=axes[0,col].imshow(error,vmin=0,vmax=.3,cmap='magma_r')
        im1=axes[1,col].imshow(coverage,vmin=0,vmax=1,cmap='cividis')
        axes[0,col].set_title(f'{seconds} s')
        for row in (0,1):
            axes[row,col].set(xticks=range(5),xticklabels=[18,24,32,40,48],
                yticks=range(6),yticklabels=[2,3,4,5,6,8],xlabel='Target anchor (Hz)')
        if col==0:
            axes[0,col].set_ylabel('RMSE\nSource anchor (Hz)')
            axes[1,col].set_ylabel('Validity\nSource anchor (Hz)')
    fig.colorbar(im0,ax=axes[0,:],shrink=.85,label='All-record RMSE (zero fallback)')
    fig.colorbar(im1,ax=axes[1,:],shrink=.85,label='Valid fraction')
    for suffix in ('pdf','png'):fig.savefig(root/f'frequency-coverage.{suffix}',dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    recovery,grid,summary=analyze(args.root)
    plots(args.root,recovery,grid)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('recovery','grid')},indent=2))
