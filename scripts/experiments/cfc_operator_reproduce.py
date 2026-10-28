"""Run or summarize the frozen operator validation and same-q factorial.

All outputs belong in a separate evidence directory, not in the source tree.
An evidence tag can supply records.json.gz instead of records.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


ARMS = {
    'legacy_h1': ['--harmonics','1','--phase-cycles','5'],
    'fixed_h1': ['--harmonics','1','--phase-cycles','9'],
    'fixed_h2': ['--harmonics','2','--phase-cycles','9'],
    'refined_h1': ['--harmonics','1','--phase-cycles','9','--refine-phase'],
    'refined_h2': ['--harmonics','2','--phase-cycles','9','--refine-phase'],
}


def records(root: Path, relative: str) -> list[dict]:
    path=root/relative/'records.json'
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(path.with_suffix('.json.gz'),'rt') as handle:
        return json.load(handle)


def paired_errors(a: list[dict], b: list[dict]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if [(r['condition'],r['seed']) for r in a] != [(r['condition'],r['seed']) for r in b]:
        raise ValueError('Records must be paired by condition and independent record seed')
    selected=[i for i,r in enumerate(a) if r['d1']+r['d2']>0]
    ea=np.array([a[i]['curve_rmse'] if a[i]['valid'] else 1. for i in selected])
    eb=np.array([b[i]['curve_rmse'] if b[i]['valid'] else 1. for i in selected])
    groups=np.array([a[i]['condition'] for i in selected])
    return ea,eb,groups


def comparison(a: list[dict], b: list[dict]) -> dict:
    ea,eb,groups=paired_errors(a,b)
    differences=[(ea-eb)[groups==g] for g in sorted(set(groups))]
    rng=np.random.default_rng(99101)
    boot=[np.mean([rng.choice(d,len(d),replace=True).mean() for d in differences]) for _ in range(2000)]
    first=~np.isin(groups,['double10','mixed10'])
    return dict(records=len(ea),a_rmse=float(ea.mean()),b_rmse=float(eb.mean()),
        relative_reduction=float(1-eb.mean()/ea.mean()),
        paired_reduction_interval_95=np.quantile(boot,[.025,.975]).tolist(),
        first_harmonic_conditions=dict(a_rmse=float(ea[first].mean()),b_rmse=float(eb[first].mean()),
            relative_reduction=float(1-eb[first].mean()/ea[first].mean())),
        by_condition={str(g):dict(a_rmse=float(ea[groups==g].mean()),b_rmse=float(eb[groups==g].mean())) for g in sorted(set(groups))})


def wilson(hits: int, total: int) -> list[float]:
    p=hits/total;z=1.959963984540054;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [mid-half,mid+half]


def detection(calibration: list[dict], test: list[dict]) -> dict:
    null_names=('aac_only','independent')
    scores=[r['predictive_score'] for r in calibration if r['condition'] in null_names]
    index=min(len(scores)-1,math.ceil((len(scores)+1)*.95)-1)
    threshold=float(np.sort(scores)[index])
    null=[r for r in test if r['condition'] in null_names]
    positive=[r for r in test if r['d1']+r['d2']>0]
    fp=sum(r['predictive_score']>threshold for r in null)
    return dict(threshold=threshold,calibration_records=len(scores),null_records=len(null),
        false_positives=fp,false_positive_rate=fp/len(null),false_positive_interval_95=wilson(fp,len(null)),
        positive_records=len(positive),sensitivity=sum(r['predictive_score']>threshold for r in positive)/len(positive),
        true_pair_detection=sum(r['pair_predictive_score']>threshold for r in positive)/len(positive),
        auroc=float(roc_auc_score([0]*len(null)+[1]*len(positive),[r['predictive_score'] for r in null+positive])),
        stress_above_threshold={c:sum(r['predictive_score']>threshold for r in test if r['condition']==c)/sum(r['condition']==c for r in test)
            for c in ('transient','nonsinusoidal')})


def summarize(root: Path) -> dict:
    manifest=root/'manifest-sha256.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text()).items():
            path=root/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Evidence checksum mismatch: {name}')
    for name,flags in ARMS.items():
        meta=json.loads((root/'arms'/name/'summary.json').read_text())
        settings=meta['args']
        wanted_harmonics=int(flags[flags.index('--harmonics')+1])
        wanted_cycles=float(flags[flags.index('--phase-cycles')+1])
        expected=(wanted_harmonics,wanted_cycles,'--refine-phase' in flags,False,False)
        actual=(settings['harmonics'],settings['phase_cycles'],settings['refine_phase'],settings['control_amplitude'],settings['local'])
        if actual!=expected:
            raise ValueError(f'Unexpected operator settings for {name}: {actual}')
    arms={name:records(root,'arms/'+name) for name in ARMS}
    results={
        'original_full_upgrade':comparison(arms['legacy_h1'],arms['refined_h2']),
        'refinement_same_h1_q9':comparison(arms['fixed_h1'],arms['refined_h1']),
        'refinement_same_h2_q9':comparison(arms['fixed_h2'],arms['refined_h2']),
        'harmonics_fixed_q9':comparison(arms['fixed_h1'],arms['fixed_h2']),
        'harmonics_refined_q9':comparison(arms['refined_h1'],arms['refined_h2']),
    }
    conditions=results['original_full_upgrade']['by_condition']
    total=sum(v['a_rmse']-v['b_rmse'] for v in conditions.values())
    two=sum(conditions[c]['a_rmse']-conditions[c]['b_rmse'] for c in ('double10','mixed10'))
    results['second_harmonic_share_of_original_absolute_gain']=two/total
    results['detection']={name:detection(records(root,'calibration/'+name),arms[name]) for name in ('legacy_h1','refined_h2')}
    results['scope']='Fixed synthetic suite; same-q comparisons isolate harmonic order and frequency refinement. Cycle score is internally dependent; calibration records are independent.'
    return results


def run(root: Path, cases: int, calibration_cases: int) -> None:
    root.mkdir(parents=True,exist_ok=False)
    jobs=[('arms/'+name,'release',cases,flags) for name,flags in ARMS.items()]
    jobs.extend(('calibration/'+name,'calibration',calibration_cases,ARMS[name]) for name in ('legacy_h1','refined_h2'))
    for name,split,count,flags in jobs:
        output=root/name;output.parent.mkdir(exist_ok=True)
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,'-m','scripts.experiments.cfc_operator_benchmark',
                '--output',str(output),'--split',split,'--cases',str(count),
                '--confidence-folds','cycles',*flags],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('completed',name,flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','summarize'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=500)
    parser.add_argument('--calibration-cases',type=int,default=200)
    parser.add_argument('--summary-output',type=Path)
    args=parser.parse_args()
    if args.cases<=0 or args.calibration_cases<=0:
        parser.error('case counts must be positive')
    if args.command=='run':
        run(args.root,args.cases,args.calibration_cases)
    result=summarize(args.root)
    output=args.summary_output or (args.root/'recomputed-summary.json' if args.command=='run' else None)
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,value in result.items():
        if isinstance(value,dict) and 'a_rmse' in value:
            print(name,round(value['a_rmse'],6),'->',round(value['b_rmse'],6),flush=True)


if __name__=='__main__':
    main()
