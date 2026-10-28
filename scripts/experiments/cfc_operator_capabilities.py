"""Truth-based PAC capability checks; no estimator-ranking or classifier claims.

The operator sees only raw samples. Physical frequencies choose predeclared
output cells for scoring, never analysis-filter centers. All invalid cases stay
in the record stream. This evaluator does not modify the frozen 20261028 suite.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator

SOURCE = np.array([2., 3., 4., 5., 6., 8.])
TARGET = np.array([18., 24., 32., 40., 48.])
SEEDS = dict(dev=12000000, calibration=14000000, confirm=16000000)


def specifications(suite: str) -> list[dict]:
    if suite == 'grid':
        return [dict(group=f'{seconds}s/{i}/{j}/{shape}', seconds=seconds,
                     source=i, target=j, depth=.4, snr_db=10., shape=shape)
                for seconds in (5, 10, 20) for i in range(6) for j in range(5)
                for shape in ('single', 'double')]
    if suite == 'null':
        return [dict(group=mode, seconds=10, source=2, target=2,
                     depth=0., snr_db=10., shape=mode)
                for mode in ('independent', 'aac')]
    if suite == 'local':
        return [dict(group='drifting_phase', seconds=10, source=2, target=2,
                     depth=.5, snr_db=10., shape='drifting')]
    if suite != 'recovery':
        raise ValueError(suite)
    rows = [dict(group=f'depth/{d}', depth=d, snr_db=10., shape='single')
            for d in (0., .02, .05, .1, .25, .5)]
    rows += [dict(group=f'snr/{s}', depth=.25, snr_db=s, shape='single')
             for s in (0., 5., 10., 20.)]
    rows += [dict(group=f'phase/{k}', depth=.4, snr_db=10., shape='single',
                  theta=2*math.pi*k/8) for k in range(8)]
    rows += [dict(group=shape, depth=.5, snr_db=10., shape=shape)
             for shape in ('double', 'mixed', 'nonlinear')]
    return [dict(seconds=10, source=2, target=2) | r for r in rows]


def band_power(x: np.ndarray, low: float, high: float, fs: int = 200) -> float:
    """Full-record FFT energy in an explicitly reported physical band."""
    f = np.fft.rfftfreq(len(x), 1/fs)
    selected = (f >= low) & (f <= high)
    y = np.fft.irfft(np.fft.rfft(x)*selected, n=len(x))
    return float(np.mean(y*y))


def generate(seed: int, spec: dict) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(round(200*spec['seconds']))/200
    i, j = spec['source'], spec['target']
    f = float(SOURCE[i]+rng.uniform(-.08, .08))
    carrier = float(TARGET[j]+rng.uniform(-.3, .3))
    phi = 2*np.pi*f*t+rng.uniform(-np.pi, np.pi)
    theta = float(spec.get('theta', rng.uniform(-np.pi, np.pi)))
    theta2 = float(rng.uniform(-np.pi, np.pi))
    depth, shape = spec['depth'], spec['shape']
    d1 = depth if shape in ('single', 'drifting') else 0.
    d2 = depth if shape == 'double' else 0.
    if shape == 'mixed':
        d1 = d2 = depth/2
    preference = np.full_like(t, theta)
    if shape == 'drifting':
        preference += .9*np.sin(2*np.pi*.2*t)
    modulation = d1*np.cos(phi-preference)+d2*np.cos(2*phi-theta2)
    baseline = np.ones_like(t)
    source_amp = 1+.25*np.sin(2*np.pi*.43*t+rng.uniform(-np.pi, np.pi))
    if shape == 'nonlinear':
        modulation = np.exp(depth*np.cos(phi-theta))/np.i0(depth)-1
        angles = np.arange(8192)*2*np.pi/8192
        curve = np.exp(depth*np.cos(angles))/np.i0(depth)-1
        d1 = float(2*np.mean(curve*np.cos(angles)))
        d2 = float(2*np.mean(curve*np.cos(2*angles)))
        theta2 = 2*theta
    if shape in ('independent', 'aac'):
        modulation[:] = 0
        if shape == 'aac':
            baseline = 1+.25*(source_amp-1)/.25
        else:
            baseline += .25*np.sin(2*np.pi*.37*t+rng.uniform(-np.pi, np.pi))
    low = .8*source_amp*np.cos(phi)
    high = (baseline+modulation)*np.cos(2*np.pi*carrier*t+rng.uniform(-np.pi, np.pi))
    noise = rng.normal(size=len(t))
    freq = np.fft.rfftfreq(len(t), 1/200)
    noise = np.fft.irfft(np.fft.rfft(noise)/np.sqrt(1+freq), n=len(t))
    lo, hi = max(.5, carrier-2*f), min(75., carrier+2*f)
    noise *= math.sqrt(band_power(high, lo, hi)/
                       max(band_power(noise, lo, hi), 1e-20)/10**(spec['snr_db']/10))
    raw = low+high+noise
    actual_snr = 10*math.log10(band_power(high, lo, hi)/band_power(noise, lo, hi))
    return dict(raw=raw, phi=phi, modulation=modulation, preference=preference,
                d1=d1, d2=d2, theta=theta, theta2=theta2, source_hz=f,
                target_hz=carrier, target_band_snr_db=actual_snr,
                total_snr_db=10*math.log10(float(np.mean((low+high)**2)/np.mean(noise**2))),
                source_band_snr_db=10*math.log10(band_power(low, max(.5, f-.5), f+.5)/
                                               band_power(noise, max(.5, f-.5), f+.5)),
                seed=seed, spec=spec)


def summarize(records: list[dict]) -> dict:
    result = {}
    for key in sorted({(r['variant'], r['group']) for r in records}):
        rows = [r for r in records if (r['variant'], r['group']) == key]
        good = [r for r in rows if r['valid']]
        phase = [r['phase1_error_deg'] for r in good if r['phase1_error_deg'] is not None]
        phase2 = [r['phase2_error_deg'] for r in good if r.get('phase2_error_deg') is not None]
        result['/'.join(key)] = dict(n=len(rows), valid_fraction=len(good)/len(rows),
            curve_rmse_all=float(np.mean([r['curve_rmse'] for r in rows])),
            curve_rmse_valid=float(np.mean([r['curve_rmse'] for r in good])) if good else None,
            depth1_mae_valid=float(np.mean([r['depth1_error'] for r in good])) if good else None,
            depth2_mae_valid=float(np.mean([r['depth2_error'] for r in good if 'depth2_error' in r])) if any('depth2_error' in r for r in good) else None,
            phase1_mae_deg_valid=float(np.mean(phase)) if phase else None,
            phase2_mae_deg_valid=float(np.mean(phase2)) if phase2 else None,
            pair_top1_fraction=float(np.mean([r['top1_correct'] for r in rows])),
            mean_source_hz_error=float(np.mean([r['source_hz_error'] for r in rows])),
            predictive_score_mean=float(np.mean([r['record_score'] for r in rows])))
    return result


@torch.no_grad()
def evaluate(args) -> None:
    if not 1 <= args.cases <= 1000:
        raise ValueError('cases must be 1..1000 to preserve disjoint ID blocks')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    specs = specifications(args.suite)
    variants = {'h1': PACCurveOperator(harmonics=1), 'curve': PACCurveOperator()}
    if args.adaptive:
        variants['adaptive'] = PACCurveOperator(adaptive_harmonics=True)
    if args.suite == 'local':
        variants = {'global': PACCurveOperator(), 'local': PACCurveOperator(local=True)}
    config = dict(protocol='capabilities-v2', arguments=vars(args) | {'output': str(args.output)},
                  specifications=specs,
                  operator_configs={k: v.configuration for k, v in variants.items()},
                  notes='Raw input only. Zero modulation fallback for invalid fits; coverage reported separately.',
                  python=__import__('sys').version, numpy=np.__version__, torch=torch.__version__)
    operator_file = Path(__import__(PACCurveOperator.__module__, fromlist=['']).__file__)
    config['operator_sha256'] = hashlib.sha256(operator_file.read_bytes()).hexdigest()
    config['evaluator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(config, indent=2)+'\n')
    records = []
    begin = time.monotonic()
    for ci, spec in enumerate(specs):
        for first in range(0, args.cases, 16):
            examples = [generate(SEEDS[args.split]+ci*10000+k, spec)
                        for k in range(first, min(first+16, args.cases))]
            raw = torch.tensor(np.stack([e['raw'] for e in examples]), dtype=torch.float32)[:, None]
            for name, operator in variants.items():
                fit = operator(raw)
                p = fit.coef.shape[2]//2
                for n, e in enumerate(examples):
                    i, j = spec['source'], spec['target']
                    good = bool(fit.valid[n, 0, p, i, j])
                    base = float(fit.baseline[n, 0, p, i, j])
                    coef = fit.coef[n, 0, p, i, j].numpy()/max(base, 1e-12)
                    prediction = fit.modulation(i, j, p)[n, 0].numpy()
                    center = slice(round(.4*len(prediction)), round(.6*len(prediction)))
                    error = float(np.sqrt(np.mean((prediction[center]-e['modulation'][center])**2)))
                    scores = fit.predictive_gain[n, 0].numpy()
                    winner = np.unravel_index(scores.argmax(), scores.shape)
                    record = {k: e[k] for k in ('seed', 'source_hz', 'target_hz',
                              'target_band_snr_db', 'total_snr_db', 'source_band_snr_db')}
                    record |= dict(variant=name, group=spec['group'], shape=spec['shape'],
                        seconds=spec['seconds'], source_index=i, target_index=j,
                        depth=spec['depth'], valid=good, curve_rmse=error,
                        depth1_error=abs(float(abs(coef[0]))-e['d1']),
                        depth2_error=abs(float(abs(coef[1]) if len(coef)>1 else 0)-e['d2']),
                        phase1_error_deg=(abs(float(np.angle(coef[0]*np.exp(-1j*e['theta']))))*180/np.pi
                                          if e['d1'] > 0 and spec['shape'] != 'drifting' else None),
                        phase2_error_deg=(abs(float(np.angle(coef[1]*np.exp(-1j*e['theta2']))))*90/np.pi
                                          if e['d2'] > 0 and len(coef)>1 else None),
                        source_hz_error=abs(float(fit.source_frequency[n, 0, i])-e['source_hz']),
                        record_score=float(scores.max()), pair_score=float(scores[i, j]),
                        top1_correct=bool(tuple(winner) == (i, j) and scores.max() > 0),
                        harmonic_support=fit.harmonic_support[i,j].numpy().tolist(),
                        truth_supported=bool(fit.harmonic_support[i,j,0] and
                            (e['d2'] == 0 or (fit.harmonic_support.shape[-1]>1 and fit.harmonic_support[i,j,1]))),
                        coef_real=coef.real.tolist(), coef_imag=coef.imag.tolist(),
                        all_valid=fit.global_valid[n, 0].numpy().tolist(),
                        pair_scores=scores.tolist(), raw_sha256=hashlib.sha256(e['raw'].tobytes()).hexdigest())
                    if args.suite == 'local':
                        anchors=(np.arange(fit.coef.shape[2])+.5)*.25
                        theta=np.interp(anchors, np.arange(len(e['phi']))/200, e['preference'])
                        truth=e['d1']*np.exp(1j*theta)
                        est=fit.coef[n,0,:,i,j,0].numpy()/fit.baseline[n,0,:,i,j].numpy().clip(1e-12)
                        mask=fit.local_valid[n,0,:,i,j].numpy()
                        errors=abs(est-truth)
                        record |= dict(local_fraction=float(mask.mean()),
                            all_anchor_error=float(errors.mean()),
                            local_valid_error=float(errors[mask].mean()) if mask.any() else None,
                            fallback_error=float(errors[~mask].mean()) if (~mask).any() else None)
                    records.append(record)
        print(json.dumps(dict(group=spec['group'], rows=len(records), seconds=round(time.monotonic()-begin, 2))), flush=True)
    summary = dict(suite=args.suite, split=args.split, cases_per_group=args.cases,
                   records=len(records), seconds=time.monotonic()-begin, groups=summarize(records))
    if args.suite == 'local':
        summary['local'] = {name: {key: float(np.mean([r[key] for r in records if r['variant']==name and r[key] is not None]))
                           for key in ('local_fraction', 'all_anchor_error', 'local_valid_error', 'fallback_error')
                           if any(r['variant']==name and r[key] is not None for r in records)} for name in variants}
    with gzip.open(args.output/'records.json.gz', 'wt') as f:
        json.dump(records, f, allow_nan=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'groups'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['recovery', 'grid', 'null', 'local'], required=True)
    parser.add_argument('--split', choices=SEEDS, default='dev')
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--adaptive', action='store_true', help='include experimental held-cycle H1/H2 blending')
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args())
