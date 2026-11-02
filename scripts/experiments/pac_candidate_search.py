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

from paclock_bench.probing.features import load_project_tokenizer, linear_token_budget
from paclock_bench.probing.synthetic import generate_dataset, curve_from_coefficients
from paclock_bench.probing.scoring import score_curves
from scripts.experiments.pac_metric_calibration import signal_reference
from scripts.experiments.pac_token_candidates import PACResidualCandidates

ALPHAS = (1e-6, 1e-4, .01, 1., 100., 10000.)


def cfft_sequence(flat, bands=16, patches=20, width=128):
    """Native band-major cache -> physical time-major readout input."""
    x=np.asarray(flat).reshape(len(flat),bands,patches,width)
    return x.transpose(0,2,1,3).reshape(len(flat),patches,bands*width)


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
    p.add_argument('--dataset', type=Path)
    p.add_argument('--cell',type=int,nargs=2,default=[2,2])
    p.add_argument('--query-pair',type=float,nargs=2)
    p.add_argument('--common-budget',action='store_true')
    p.add_argument('--variants',nargs='+',choices=['coefficient','content','tangent'],default=['coefficient','content','tangent'])
    p.add_argument('--include-shuffle',action='store_true')
    p.add_argument('--train', type=int, default=2048)
    p.add_argument('--val', type=int, default=512)
    p.add_argument('--test', type=int, default=1024)
    p.add_argument('--gains', type=float, nargs='+', default=[.03,.1])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    if args.dataset or args.cache_from:
        with np.load(args.dataset or args.cache_from/'data.npz',allow_pickle=False) as a:
            data=dict(a)
        assert int(data['seed'])==args.seed
        assert [int((data['split']==s).sum()) for s in ('train','val','test')]==[args.train,args.val,args.test]
    else:
        data = generate_dataset(args.train,args.val,args.test,seed=args.seed)
    np.savez_compressed(args.output/'data.npz', **data)
    dump(args.output/'protocol.json', dict(seed=args.seed, train=args.train,val=args.val,test=args.test,
         gains=args.gains, variants=args.variants, shuffle=args.include_shuffle, degrees=[1,2], alphas=ALPHAS, selection='validation MSE only',
         dimension='complete original 16x20x128 token coordinates, flatten only',
         perturbation_budget='RMS(delta) / RMS(original) <= gain for every record',
         scope='frozen deterministic candidate feasibility; no clinical or PAC-supervised tokenizer training',
         residual_scaling='per-kind train RMS matched, capped per record at original RMS before gain',
         cache_from=str(args.cache_from),dataset=str(args.dataset),cell=args.cell,query_pair=args.query_pair,common_budget=args.common_budget,
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         candidate_source_sha256=hashlib.sha256(Path(__file__).with_name('pac_token_candidates.py').read_bytes()).hexdigest()))
    model, meta = load_project_tokenizer(args.reference/'original-training.yaml',
        args.reference/'original-best.pt',(16,1000),normalization='div100')
    candidates = PACResidualCandidates(model.frontend)
    n = len(data['signals']); shape = (n,40960)
    array_keys=('base',)+tuple(dict.fromkeys(args.variants+(['coefficient']if args.include_shuffle else [])))
    arrays = {key: (np.load(args.cache_from/(key+'.npy'),mmap_mode='r') if args.cache_from else
                   np.lib.format.open_memmap(args.output/(key+'.npy'),mode='w+',dtype=np.float32,shape=shape))
              for key in array_keys}
    assert all(a.shape==shape for a in arrays.values())
    coefficient, mask = [], []
    for first in range(0,0 if args.cache_from else n,32):
        x = torch.from_numpy(data['signals'][first:first+32])*.2
        base, additive, content, tangent, packet = candidates.components(x)
        for key in arrays:
            value=dict(base=base,coefficient=additive,content=content,tangent=tangent)[key]
            arrays[key][first:first+len(x)] = value.numpy().reshape(len(x),-1)
        coefficient.append(packet['features'][:,0,0,args.cell[0],args.cell[1]].flatten(-2).numpy())
        mask.append(packet['harmonic_valid'][:,0,0,args.cell[0],args.cell[1]].numpy())
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
    ref_kwargs={}
    if args.query_pair:
        source,target=args.query_pair
        ref_kwargs=dict(source_band=(source-1.2,source+1.2),target_band=(target-2*source-3,target+2*source+3))
    report = dict(operator=score_curves(data['curves'][te],curve_from_coefficients(coef[te]),data['coupled'][te]),
                  signal_reference=score_curves(data['curves'][te],curve_from_coefficients(signal_reference(data['signals'][te],**ref_kwargs)),data['coupled'][te]),
                  checkpoint=meta,rows={})
    dump(args.output/'measurement.json',{k:v for k,v in report.items() if k!='rows'})
    base = arrays['base']
    # One amplitude scale fitted on train only, no per-record ground truth.
    scale = float(np.sqrt(np.mean(np.square(base[data['split']=='train'],dtype=np.float64))))
    residual_scales={key:scale/max(float(np.sqrt(np.mean(np.square(arrays[key][data['split']=='train'],dtype=np.float64)))),1e-12)
                     for key in array_keys if key!='base'}
    specs=[('original',None,0.)]+[(f'{kind}_{g:g}',kind,g)for kind in args.variants for g in args.gains]
    if args.include_shuffle:
        specs += [(f'shuffled_coefficient_{g:g}','shuffled',g)for g in args.gains]
    shuffle=np.arange(n)
    rng=np.random.default_rng(91103)
    for split in ('train','val','test'):
        ids=np.flatnonzero(data['split']==split)
        # Circular random nonzero shift is a derangement with no split mixing.
        shuffle[ids]=np.roll(ids,int(rng.integers(1,len(ids))))
    with (args.output/'results.tsv').open('x') as table:
        table.write('variant\tdegree\tvalidation_mse\tcurve_r2\tdetection_auroc\tseconds\tmax_relative_perturbation\n')
        for name,kind,gain in specs:
            begin = time.monotonic()
            feature = np.array(base)
            perturb = np.zeros(n)
            if kind:
                for first in range(0,n,64):
                    b = np.array(base[first:first+64])
                    source_kind='coefficient' if kind=='shuffled' else kind
                    ids=shuffle[first:first+64] if kind=='shuffled' else slice(first,first+64)
                    delta = np.array(arrays[source_kind][ids])*residual_scales[source_kind]
                    norm = np.sqrt(np.mean(b*b,axis=1,keepdims=True))
                    size = np.sqrt(np.mean(delta*delta,axis=1,keepdims=True))
                    delta *= np.minimum(1.,norm/np.maximum(size,1e-12))
                    feature[first:first+len(b)] += gain*delta
                    perturb[first:first+len(b)] = np.sqrt(np.mean((gain*delta)**2,axis=1))/norm[:,0]
            if args.common_budget:
                feature=linear_token_budget(cfft_sequence(feature),time_bins=9,width=64,seed=0)
            row = dict(readout_features=int(feature.shape[1]),mean_relative_perturbation=float(perturb.mean()),max_relative_perturbation=float(perturb.max()),readouts=[])
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
