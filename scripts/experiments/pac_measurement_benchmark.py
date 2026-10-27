"""Frozen raw-signal PAC recovery evaluator. Outputs every case and a summary.

Development and holdout use different frequencies, durations and RNG seeds.
The estimator receives raw EEG only. Oracle phase/envelope never enters its
forward path; they are used solely for scoring and algebra diagnostics.
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
from sklearn.metrics import roc_auc_score

from paclock_bench.models.paclock.frontend.measurement import (
    PhaseAmplitudeMeasurement, linear_analytic_filter,
)


def signals(source_hz: float, target_hz: float, duration: float, fs: int,
            seed: int, noise_db: float, independent: bool = False):
    rng = np.random.default_rng(seed)
    t = np.arange(round(fs*duration))/fs
    examples, truth = [], []
    for depth in ([0.] if independent else [0., .2, .5, .8]):
        for theta in [0., math.pi/2, math.pi, 3*math.pi/2]:
            offset, carrier_phase = rng.uniform(-math.pi, math.pi, 2)
            phase = 2*math.pi*source_hz*t+offset
            if independent:
                # An independent slowly varying envelope, not a shifted copy
                # of the source sinusoid (which would still have PAC).
                e = rng.normal(size=t.size)
                f = np.fft.rfftfreq(t.size, 1/fs)
                e = np.fft.irfft(np.fft.rfft(e)*np.exp(-(f/1.3)**4), n=t.size)
                e = e/(e.std()+1e-12)
                envelope = np.exp(.18*e)
            else:
                envelope = 1+depth*np.cos(phase-theta)
            raw = .8*np.cos(phase)+envelope*np.cos(2*math.pi*target_hz*t+carrier_phase)
            if math.isfinite(noise_db):
                noise = rng.normal(size=t.size)
                # Colored rather than purely white observation noise.
                f = np.fft.rfftfreq(t.size, 1/fs)
                noise = np.fft.irfft(np.fft.rfft(noise)/np.sqrt(1+f), n=t.size)
                noise *= np.sqrt(np.mean(raw**2)/10**(noise_db/10))/(noise.std()+1e-12)
                raw = raw+noise
            examples.append(raw)
            truth.append(dict(depth=depth, theta=theta, independent=independent,
                              source_hz=source_hz, target_hz=target_hz,
                              seconds=duration, noise_db=noise_db if math.isfinite(noise_db) else 'inf', seed=seed))
    return torch.tensor(np.array(examples), dtype=torch.float32)[:, None], truth


@torch.no_grad()
def evaluate(split: str, output: Path, phase_cycles: float = 5., detrend: bool = True):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    fs, patch = 200, 50
    settings = {'dev': ([(4.,32.), (6.,40.), (8.,55.)], [5.,7.], 1100),
                'holdout': ([(3.7,28.), (5.3,37.), (9.1,59.)], [5.5,7.5], 9100),
                'confirmation': ([(3.4,29.), (5.7,39.), (8.3,57.)], [5.25,8.25], 19100)}
    pairs, durations, seed_base = settings[split]
    records = []
    start = time.monotonic()
    for pair_index, (low, high) in enumerate(pairs):
        module = PhaseAmplitudeMeasurement([low, high], fs, patch, phase_cycles=phase_cycles, detrend=detrend)
        for duration in durations:
            for noise in [float('inf'),20.,10.]:
                for independent in [False, True]:
                    x, truth = signals(low, high, duration, fs, seed_base+pair_index*10+int(duration), noise, independent)
                    result = module(x)
                    ph = result.phase_unit[:,0,0]
                    env = linear_analytic_filter(x, module.amplitude_kernels)[:,0,1].abs()
                    patches = result.valid.shape[2]
                    ph_patch = ph[:,:patches*patch].reshape(x.shape[0],patches,patch)
                    env_patch = env[:,:patches*patch].reshape(x.shape[0],patches,patch)
                    moment = (ph_patch*(env_patch-env_patch.mean(-1,keepdim=True))).mean(-1)
                    short_depth = 2*moment.abs()/env_patch.mean(-1).clamp_min(1e-12)
                    win=math.ceil(max(module.min_seconds,module.min_cycles/float(module.phase_lower[0]))*fs)+1
                    index=((torch.arange(patches)+.5)*patch).long()[:,None]+torch.arange(win)[None,:]-win//2
                    bounded=index.clamp(0,x.shape[-1]-1)
                    mask=(module.phase_guard[0]<=index)&(index<x.shape[-1]-module.phase_guard[0])
                    mask=mask&(module.amplitude_guard[1]<=index)&(index<x.shape[-1]-module.amplitude_guard[1])
                    weight=mask/mask.sum(-1,keepdim=True).clamp_min(1)
                    a=env[:,bounded];p_unit=ph[:,bounded]
                    mean_a=(a*weight).sum(-1)
                    long_z=(p_unit*(a-mean_a[...,None])*weight).sum(-1)
                    long_depth=2*long_z.abs()/mean_a.clamp_min(1e-12)
                    for n, info in enumerate(truth):
                        for p in range(patches):
                            valid = bool(result.valid[n,0,p,0,1])
                            estimate = float(result.depth[n,0,p,0,1])
                            theta = float(result.coef[n,0,p,0,1].angle()) if valid else float('nan')
                            error = abs(math.atan2(math.sin(theta-info['theta']),math.cos(theta-info['theta']))) if valid and info['depth'] else float('nan')
                            anchor = (p+.5)*patch/fs
                            records.append({**info, 'patch':p,'anchor_seconds':anchor,
                                'central':.4*duration <= anchor <= .6*duration,
                                'valid':valid,'estimated_depth':estimate,'estimated_theta':theta,
                                'depth_error':abs(estimate-info['depth']) if valid else float('nan'),
                                'phase_error_deg':error*180/math.pi,
                                'short_moment_depth':float(short_depth[n,p]),
                                'short_moment_phase_error_deg':abs(math.atan2(math.sin(float(torch.angle(moment[n,p]))-info['theta']),math.cos(float(torch.angle(moment[n,p]))-info['theta'])))*180/math.pi,
                                'long_moment_depth':float(long_depth[n,p]),
                                'long_moment_phase_error_deg':abs(math.atan2(math.sin(float(torch.angle(long_z[n,p]))-info['theta']),math.cos(float(torch.angle(long_z[n,p]))-info['theta'])))*180/math.pi,
                                'cycles':float(result.observed_cycles[n,0,p,0,1]),
                                'context_count':float(result.context_count[n,0,p,0,1])})
    with (output/'estimates.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    primary = [r for r in records if r['central'] and not r['independent'] and r['noise_db'] != 10.]
    valid = [r for r in primary if r['valid']]
    coupled = [r for r in valid if r['depth'] >= .2]
    null = [r for r in records if r['central'] and r['valid'] and (r['independent'] or r['depth']==0)]
    noiseless = [r for r in coupled if r['noise_db']=='inf']
    per_case = {}
    for r in records:
        if r['central'] and r['valid']:
            key = (r['source_hz'],r['target_hz'],r['seconds'],r['noise_db'],r['seed'],r['independent'],r['theta'],r['depth'])
            per_case.setdefault(key,[]).append(r['estimated_depth'])
    labels, scores = [], []
    for key, values in per_case.items():
        labels.append(key[-1]>=.2 and not key[-3]);scores.append(float(np.mean(values)))
    summary = dict(split=split, phase_cycles=phase_cycles, detrend=detrend,
        seconds=time.monotonic()-start, patches=len(records), cases=len(per_case),
        primary_valid_coverage=len(valid)/max(1,len(primary)),
        all_anchor_coverage=float(np.mean([r['valid'] for r in records])),
        depth_mae=float(np.mean([r['depth_error'] for r in coupled])) if coupled else None,
        phase_mae_deg=float(np.mean([r['phase_error_deg'] for r in coupled])) if coupled else None,
        noiseless_depth_mae=float(np.mean([r['depth_error'] for r in noiseless])) if noiseless else None,
        null_depth_q95=float(np.quantile([r['estimated_depth'] for r in null],.95)) if null else None,
        detection_auroc=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,
        short_moment_depth_mae=float(np.mean([abs(r['short_moment_depth']-r['depth']) for r in coupled])) if coupled else None,
        short_moment_phase_mae_deg=float(np.mean([r['short_moment_phase_error_deg'] for r in coupled])) if coupled else None,
        long_moment_depth_mae=float(np.mean([abs(r['long_moment_depth']-r['depth']) for r in coupled])) if coupled else None,
        long_moment_phase_mae_deg=float(np.mean([r['long_moment_phase_error_deg'] for r in coupled])) if coupled else None)
    summary['gates_pass'] = (summary['primary_valid_coverage'] >= .8 and summary['depth_mae'] is not None
        and summary['depth_mae'] <= .08 and summary['phase_mae_deg'] <= 10)
    source = Path(__file__).parents[2]/'paclock_bench/models/paclock/frontend/measurement.py'
    summary['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','holdout','confirmation'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase-cycles',type=float,default=5.)
    parser.add_argument('--no-detrend',action='store_true')
    args=parser.parse_args()
    evaluate(args.split,args.output,args.phase_cycles,not args.no_detrend)
