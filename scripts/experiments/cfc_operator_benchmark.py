"""Frozen fixed-bank PAC benchmark. Generated frequencies never enter operator.

Synthetic truth is scored for the predeclared nearest physical band, not the
most favorable measured pair. Separate development, calibration and final RNG.
All records and all invalid measurements remain in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator

SOURCE=np.array([2.,3.,4.,5.,6.,8.])
TARGET=np.array([18.,24.,32.,40.,48.])
CONDITIONS=('single20','single10','offgrid','double10','mixed10','aac_only',
            'independent','pac_aac','chirp','transient','nonsinusoidal')
SPLIT_SEEDS={'dev':210000,'calibration':310000,'final':410000,'confirm':510000}


def generate(seed: int, condition: str, seconds: float=5.) -> dict:
    rng=np.random.default_rng(seed);t=np.arange(round(seconds*200))/200
    f=rng.uniform(3.2,4.3) if condition in ('double10','mixed10') else rng.uniform(3.2,6.3)
    carrier=rng.uniform(30.2,33.8);offset=rng.uniform(-np.pi,np.pi)
    phi=2*np.pi*f*t+offset
    if condition=='chirp':phi+=.25*np.sin(2*np.pi*.55*t+rng.uniform(-np.pi,np.pi))
    source_amp=1+.3*np.sin(2*np.pi*.47*t+rng.uniform(-np.pi,np.pi))
    d1=rng.uniform(.25,.65);d2=0.
    theta1,theta2=rng.uniform(-np.pi,np.pi,2)
    if condition=='double10':d1=0.;d2=rng.uniform(.3,.65)
    if condition=='mixed10':d1=rng.uniform(.2,.4);d2=rng.uniform(.2,.4)
    if condition in ('aac_only','independent','transient','nonsinusoidal'):d1=d2=0.
    modulation=d1*np.cos(phi-theta1)+d2*np.cos(2*phi-theta2)
    baseline=np.ones_like(t)
    if condition in ('aac_only','pac_aac'):
        loga=np.log(source_amp);loga=(loga-loga.mean())/loga.std()
        baseline+=.2*loga
    if condition=='independent':
        baseline+=.35*np.sin(2*np.pi*rng.uniform(.31,.8)*t+rng.uniform(-np.pi,np.pi))
    low=.8*source_amp*np.cos(phi)
    high=(baseline+modulation)*np.cos(2*np.pi*carrier*t+rng.uniform(-np.pi,np.pi))
    raw=low+high
    if condition=='transient':
        pulse=np.exp(-((t-rng.uniform(1.5,3.5))/.07)**2)
        raw+=2*pulse
    if condition=='nonsinusoidal':
        raw=.8*(np.cos(phi)+.25*np.cos(2*phi)+.1*np.cos(3*phi))+high
    noise_db=20. if condition in ('single20','offgrid') else 10.
    if condition in ('aac_only','independent','transient','nonsinusoidal'):noise_db=20.
    noise=rng.normal(size=t.size)
    frequencies=np.fft.rfftfreq(t.size,1/200)
    noise=np.fft.irfft(np.fft.rfft(noise)/np.sqrt(1+frequencies),n=t.size)
    noise*=np.sqrt(np.mean(raw**2)/10**(noise_db/10))/(noise.std()+1e-12)
    raw+=noise
    return dict(raw=raw,phi=phi,modulation=modulation,d1=d1,d2=d2,theta1=theta1,
                theta2=theta2,source_hz=f,target_hz=carrier,noise_db=noise_db,
                source_index=int(np.argmin(abs(SOURCE-f))),target_index=2,condition=condition,seed=seed)


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    op=PACCurveOperator(harmonics=args.harmonics,control_amplitude=not args.no_amplitude,
                        local=args.local,ridge=args.ridge,local_ridge=args.local_ridge,
                        amplitude_floor=args.amplitude_floor,phase_cycles=args.phase_cycles)
    records=[];start=time.monotonic()
    base=SPLIT_SEEDS[args.split]
    for ci,condition in enumerate(CONDITIONS):
        for first in range(0,args.cases,16):
            examples=[generate(base+ci*1000+k,condition,args.seconds)
                      for k in range(first,min(first+16,args.cases))]
            raw=torch.tensor(np.stack([e['raw'] for e in examples]),dtype=torch.float32)[:,None]
            out=op(raw)
            center=out.coef.shape[2]//2
            times=np.arange(raw.shape[-1])/200
            central=(times>=.4*args.seconds)&(times<=.6*args.seconds)
            for n,e in enumerate(examples):
                i,j=e['source_index'],e['target_index']
                good=bool(out.valid[n,0,center,i,j])
                coef=out.coef[n,0,center,i,j].numpy()
                baseline=float(out.baseline[n,0,center,i,j])
                measured_phase=out.source_phase[n,0,i].numpy()
                predicted=sum((coef[k].conjugate()*measured_phase**(k+1)).real for k in range(len(coef)))/max(baseline,1e-12)
                rmse=float(np.sqrt(np.mean((predicted[central]-e['modulation'][central])**2)))
                true_sd=float(np.std(e['modulation'][central]))
                depth1=float(abs(coef[0])/max(baseline,1e-12))
                depth2=float(abs(coef[1])/max(baseline,1e-12)) if len(coef)>1 else 0.
                delta=float(np.angle(np.exp(1j*(np.angle(coef[0])-e['theta1']))))
                depths=out.coef[n,0,center].abs().sum(-1)/out.baseline[n,0,center].clamp_min(1e-12)
                all_valid=out.valid[n,0,center]
                record_score=float(torch.where(all_valid,depths,0).max())
                records.append({k:e[k] for k in ('condition','seed','source_hz','target_hz','noise_db','d1','d2')}|dict(
                    valid=good,depth1=depth1,depth2=depth2,
                    depth1_error=abs(depth1-e['d1']),depth2_error=abs(depth2-e['d2']),
                    phase1_error_deg=abs(delta)*180/math.pi,
                    curve_rmse=rmse,curve_nrmse=rmse/true_sd if true_sd>1e-8 else None,
                    record_score=record_score,valid_pairs=int(all_valid.sum()),
                    predictive_score=float(out.predictive_gain[n,0].max()),
                    global_coverage=float(out.global_valid[n].float().mean()),
                    measured_coef_real=[float(v.real) for v in coef],measured_coef_imag=[float(v.imag) for v in coef]))
        print(json.dumps({'condition':condition,'records':len(records),'seconds':time.monotonic()-start}),flush=True)
    per_condition={}
    for condition in CONDITIONS:
        rows=[r for r in records if r['condition']==condition];good=[r for r in rows if r['valid']]
        per_condition[condition]=dict(n=len(rows),valid_fraction=len(good)/len(rows),
            mean_curve_rmse=float(np.mean([r['curve_rmse'] for r in good])) if good else None,
            mean_curve_nrmse=float(np.mean([r['curve_nrmse'] for r in good if r['curve_nrmse'] is not None])) if any(r['curve_nrmse'] is not None for r in good) else None,
            depth1_mae=float(np.mean([r['depth1_error'] for r in good])) if good else None,
            depth2_mae=float(np.mean([r['depth2_error'] for r in good])) if good else None,
            phase1_mae_deg=float(np.mean([r['phase1_error_deg'] for r in good if r['d1']>0])) if any(r['d1']>0 for r in good) else None,
            record_score_q95=float(np.quantile([r['record_score'] for r in rows],.95)),
            predictive_score_q95=float(np.quantile([r['predictive_score'] for r in rows],.95)))
    positives=[r for r in records if r['d1']+r['d2']>0]
    # Invalid positive records incur a fixed full-scale error, never disappear.
    metric=float(np.mean([r['curve_rmse'] if r['valid'] else 1. for r in positives]))
    summary=dict(split=args.split,cases_per_condition=args.cases,seconds_per_record=args.seconds,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        metric_invalid_penalized_curve_rmse=metric,per_condition=per_condition,
        seconds=time.monotonic()-start,operator_sha256=hashlib.sha256(Path(__import__('paclock_bench.models.paclock.frontend.measurement_curve',fromlist=['']).__file__).read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=SPLIT_SEEDS,default='dev')
    parser.add_argument('--cases',type=int,default=24)
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--harmonics',type=int,default=2)
    parser.add_argument('--no-amplitude',action='store_true')
    parser.add_argument('--local',action='store_true')
    parser.add_argument('--ridge',type=float,default=1e-3)
    parser.add_argument('--local-ridge',type=float,default=.05)
    parser.add_argument('--amplitude-floor',type=float,default=.02)
    parser.add_argument('--phase-cycles',type=float,default=5.)
    evaluate(parser.parse_args())
