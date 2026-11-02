"""Compare band-bounded residuals and one extra relation-context time slot.

PCA placement uses original training tokens only, no PAC labels or metric
projection. It protects high-variance content directions while retaining the
same 128-dimensional token width. The extra-slot alternative leaves every
original content token exactly unchanged, adding one context slot per band.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paclock_bench.probing.features import linear_token_budget
from scripts.experiments.pac_candidate_search import probe,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--variants',nargs='+')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=dict(np.load(args.cache/'data.npz',allow_pickle=False));tr=data['split']=='train'
    base=np.load(args.cache/'base.npy',mmap_mode='r').reshape(-1,16,20,128)
    residual=np.load(args.cache/'coefficient.npy',mmap_mode='r').reshape(base.shape)
    generator=torch.Generator().manual_seed(1103)
    q,_=torch.linalg.qr(torch.randn(128,24,generator=generator));projection=q.T.numpy()
    basis=[];variance=[]
    for band in range(16):
        x=np.array(base[tr,band]).reshape(-1,128).astype(np.float64)
        covariance=np.cov(x,rowvar=False)
        value,vector=np.linalg.eigh(covariance)
        basis.append(vector[:,:24].T.astype(np.float32));variance.append(value.tolist())
    basis=np.stack(basis)
    np.savez_compressed(args.output/'placement.npz',basis=basis,original_projection=projection)
    # Recover relation coordinates from their label-independent orthogonal lift.
    coeff=np.asarray(residual)@projection.T
    placed=np.einsum('nbpk,bkd->nbpd',coeff,basis,optimize=True)
    zero=np.sqrt(np.mean(np.array(base)**2,axis=(2,3),keepdims=True))
    reference={}
    for kind,r in [('random',residual),('pca',placed)]:
        norms=np.sqrt(np.mean(np.asarray(r)**2,axis=(2,3)))
        reference[kind]=np.maximum(np.quantile(norms[tr],.95,axis=0),1e-12)[None,:,None,None]
    specs=[('band_random_3pct','random',.03),('band_pca_3pct','pca',.03),
           ('band_pca_10pct','pca',.1),('relation_context_slot','context',0.),('shuffled_context_slot','shuffled_context',0.)]
    if args.variants:specs=[v for v in specs if v[0]in args.variants]
    shuffle=np.arange(len(base));rng=np.random.default_rng(17103)
    for split in ('train','val','test'):
        ids=np.flatnonzero(data['split']==split)
        shuffle[ids]=np.roll(ids,int(rng.integers(1,len(ids))))
    result=dict(scope='synthetic token retention prototypes; no clinical efficacy evidence',
        pca_fit='original training tokens only, lowest 24 covariance eigenvectors per band',
        residual_reference='train-only per-band 95th percentile RMS; recordwise band cap',rows={})
    begin=time.monotonic()
    for name,kind,gain in specs:
        b=np.array(base)
        if kind in ('context','shuffled_context'):
            context=np.array(residual).mean(2,keepdims=True)
            scale=float(np.sqrt(np.mean(b[tr]**2)))/max(float(np.sqrt(np.mean(context[tr]**2))),1e-12)
            if kind=='shuffled_context':context=context[shuffle]
            token=np.concatenate([b,context*scale],axis=2)
            deviation=dict(original_content_exact=True,extra_tokens_fraction=.05,context_scale=scale)
        else:
            r=np.array(residual if kind=='random' else placed)
            size=np.sqrt(np.mean(r*r,axis=(2,3),keepdims=True))
            delta=gain*zero*r/np.maximum(size,reference[kind])
            token=b+delta
            fraction=np.sqrt(np.mean(delta*delta,axis=(2,3)))/np.maximum(zero[:,:,0,0],1e-12)
            deviation=dict(max_band_relative_rms=float(fraction.max()),mean_band_relative_rms=float(fraction.mean()))
        row=dict(deviation=deviation,readouts={})
        predictions={}
        for budget in ('native','common576'):
            flat=token.reshape(len(token),-1) if budget=='native' else linear_token_budget(token.transpose(0,2,1,3).reshape(len(token),token.shape[2],-1),9,64,seed=0)
            choices=[];preds=[]
            for degree in (1,2):
                report,pred=probe(flat,data,degree);choices.append(report);preds.append(pred)
            chosen=min(choices,key=lambda v:v['validation_mse'])
            row['readouts'][budget]=dict(features=flat.shape[1],selected_degree=chosen['degree'],choices=choices)
            predictions[budget]=np.stack(preds)
            print(name,budget,chosen['overall'],deviation,flush=True)
        result['rows'][name]=row;dump(args.output/(name+'.json'),row)
        np.savez_compressed(args.output/(name+'-predictions.npz'),**predictions)
    result['seconds']=time.monotonic()-begin
    dump(args.output/'scores.json',result)


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
