"""Cached full-token candidate search; validation chooses all readout settings."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import torch

from paclock_bench.probing.features import load_project_tokenizer
from paclock_bench.probing.synthetic import generate_dataset, curve_from_coefficients
from paclock_bench.probing.scoring import score_curves
from scripts.experiments.pac_metric_calibration import signal_reference
from scripts.experiments.pac_token_candidates import PACResidualCandidates

ALPHAS = (1e-6, 1e-4, .01, 1., 100., 10000.)


def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def probe(features, data, degree):
    tr, va, te = (data['split'] == s for s in ('train','val','test'))
    scaler = StandardScaler().fit(features[tr])
    x = scaler.transform(features).astype(np.float64)
    k = (x@x[tr].T)/features.shape[1]
    if degree > 1:
        k = (1+k)**degree
    col = k[tr].mean(0)
    k = k-k.mean(1, keepdims=True)-col[None]+col.mean()
    values, vectors = eigh(k[tr], check_finite=False)
    values = np.maximum(values, 0.)
    mean = data['curves'][tr].mean(0)
    projected = vectors.T@(data['curves'][tr]-mean)
    best, records = None, []
    for a in ALPHAS:
        weights = vectors@(projected/(values[:,None]+a))
        pred = k[va]@weights+mean
        error = float(np.mean((pred-data['curves'][va])**2))
        records.append(dict(alpha=a, mse=error))
        if best is None or error < best[0]:
            best = (error, a, weights)
    pred = k[te]@best[2]+mean
    return dict(degree=degree, alpha=best[1], validation_mse=best[0],
                validation_grid=records, overall=score_curves(data['curves'][te], pred, data['coupled'][te])), pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1103)
    p.add_argument('--cache-from', type=Path)
    p.add_argument('--train', type=int, default=2048)
    p.add_argument('--val', type=int, default=512)
    p.add_argument('--test', type=int, default=1024)
    p.add_argument('--gains', type=float, nargs='+', default=[.03,.1])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    if args.cache_from:
        with np.load(args.cache_from/'data.npz',allow_pickle=False) as a:
            data=dict(a)
        assert int(data['seed'])==args.seed
        assert [int((data['split']==s).sum()) for s in ('train','val','test')]==[args.train,args.val,args.test]
    else:
        data = generate_dataset(args.train,args.val,args.test,seed=args.seed)
    np.savez_compressed(args.output/'data.npz', **data)
    dump(args.output/'protocol.json', dict(seed=args.seed, train=args.train,val=args.val,test=args.test,
         gains=args.gains, degrees=[1,2], alphas=ALPHAS, selection='validation MSE only',
         dimension='complete original 16x20x128 token coordinates, flatten only',
         perturbation_budget='RMS(delta) / RMS(original) <= gain for every record',
         scope='frozen deterministic candidate feasibility; no clinical or PAC-supervised tokenizer training',
         residual_scaling='per-kind train RMS matched, capped per record at original RMS before gain',
         cache_from=str(args.cache_from),
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         candidate_source_sha256=hashlib.sha256(Path(__file__).with_name('pac_token_candidates.py').read_bytes()).hexdigest()))
    model, meta = load_project_tokenizer(args.reference/'original-training.yaml',
        args.reference/'original-best.pt',(16,1000),normalization='div100')
    candidates = PACResidualCandidates(model.frontend)
    n = len(data['signals']); shape = (n,40960)
    arrays = {key: (np.load(args.cache_from/(key+'.npy'),mmap_mode='r') if args.cache_from else
                   np.lib.format.open_memmap(args.output/(key+'.npy'),mode='w+',dtype=np.float32,shape=shape))
              for key in ('base','coefficient','content','tangent')}
    assert all(a.shape==shape for a in arrays.values())
    coefficient, mask = [], []
    for first in range(0,0 if args.cache_from else n,32):
        x = torch.from_numpy(data['signals'][first:first+32])*.2
        base, additive, content, tangent, packet = candidates.components(x)
        for key, value in zip(arrays, (base,additive,content,tangent)):
            arrays[key][first:first+len(x)] = value.numpy().reshape(len(x),-1)
        coefficient.append(packet['features'][:,0,0,2,2].flatten(-2).numpy())
        mask.append(packet['harmonic_valid'][:,0,0,2,2].numpy())
        if first % 512 == 0:
            print('extracted',first,'/',n,round(time.monotonic()-start,2),flush=True)
    for a in arrays.values():
        a.flush()
    if args.cache_from:
        with np.load(args.cache_from/'operator.npz',allow_pickle=False) as a:
            coef=a['coefficient'];mask=[a['valid']]
    else:
        coef = np.concatenate(coefficient)
    np.savez_compressed(args.output/'operator.npz',coefficient=coef,valid=np.concatenate(mask))
    te = data['split']=='test'
    report = dict(operator=score_curves(data['curves'][te],curve_from_coefficients(coef[te]),data['coupled'][te]),
                  signal_reference=score_curves(data['curves'][te],curve_from_coefficients(signal_reference(data['signals'][te])),data['coupled'][te]),
                  checkpoint=meta,rows={})
    dump(args.output/'measurement.json',{k:v for k,v in report.items() if k!='rows'})
    base = arrays['base']
    # One amplitude scale fitted on train only, no per-record ground truth.
    scale = float(np.sqrt(np.mean(np.square(base[data['split']=='train'],dtype=np.float64))))
    residual_scales={key:scale/max(float(np.sqrt(np.mean(np.square(arrays[key][data['split']=='train'],dtype=np.float64)))),1e-12)
                     for key in ('coefficient','content','tangent')}
    specs=[('original',None,0.)]+[(f'{kind}_{g:g}',kind,g)for kind in ('coefficient','content','tangent')for g in args.gains]
    with (args.output/'results.tsv').open('x') as table:
        table.write('variant\tdegree\tvalidation_mse\tcurve_r2\tdetection_auroc\tseconds\tmax_relative_perturbation\n')
        for name,kind,gain in specs:
            begin = time.monotonic()
            feature = np.array(base)
            perturb = np.zeros(n)
            if kind:
                for first in range(0,n,64):
                    b = np.array(base[first:first+64]); delta = np.array(arrays[kind][first:first+64])*residual_scales[kind]
                    norm = np.sqrt(np.mean(b*b,axis=1,keepdims=True))
                    size = np.sqrt(np.mean(delta*delta,axis=1,keepdims=True))
                    delta *= np.minimum(1.,norm/np.maximum(size,1e-12))
                    feature[first:first+len(b)] += gain*delta
                    perturb[first:first+len(b)] = np.sqrt(np.mean((gain*delta)**2,axis=1))/norm[:,0]
            row = dict(mean_relative_perturbation=float(perturb.mean()),max_relative_perturbation=float(perturb.max()),readouts=[])
            predictions=[]
            for degree in (1,2):
                result,pred = probe(feature,data,degree)
                result['families']={}
                for f in ('h1','h2','mixed','null'):
                    select=data['family'][te]==f
                    result['families'][f]=score_curves(data['curves'][te][select],pred[select],data['coupled'][te][select])
                row['readouts'].append(result);predictions.append(pred)
                table.write(f'{name}\t{degree}\t{result["validation_mse"]:.9f}\t{result["overall"]["curve_r2"]:.9f}\t{result["overall"]["detection_auroc"]:.9f}\t{time.monotonic()-begin:.2f}\t{perturb.max():.9f}\n');table.flush()
                print(name,degree,result['overall'],'seconds',round(time.monotonic()-begin,2),flush=True)
            row['selected_degree']=min(row['readouts'],key=lambda v:v['validation_mse'])['degree']
            report['rows'][name]=row
            dump(args.output/(name+'.json'),row)
            np.savez_compressed(args.output/(name+'-predictions.npz'),predictions=predictions,
                sample_id=data['sample_id'][te],target=data['curves'][te],coupled=data['coupled'][te])
            del feature
    report['elapsed_seconds']=time.monotonic()-start
    report['train_derived_base_rms']=scale
    report['train_derived_residual_scales']=residual_scales
    dump(args.output/'scores.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
