"""Diagnostic nonlinear kernels and conditional parameter readouts.

These are not replacements for the frozen primary curve benchmark. Strength
and phase fits condition on coupling being present; they do not detect PAC.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.experiments.pac_candidate_search import dump


def fit(features,target,split):
    tr,va,te=(split==s for s in ('train','val','test'))
    sc=StandardScaler().fit(features[tr]);x=sc.transform(features).astype(np.float64)
    linear=x@x[tr].T/x.shape[1]
    norm=np.mean(x*x,axis=1);distance=np.maximum(0,norm[:,None]+norm[tr][None]-2*linear)
    mean=target[tr].mean(0);best=None;history=[]
    for kind in ('linear','quadratic','cubic','rbf_0.1','rbf_1','rbf_10'):
        if kind.startswith('rbf'):k=np.exp(-float(kind.split('_')[1])*distance)
        else:k=linear if kind=='linear'else(1+linear)**({'quadratic':2,'cubic':3}[kind])
        col=k[tr].mean(0);k=k-k.mean(1,keepdims=True)-col[None]+col.mean()
        values,vectors=eigh(k[tr],check_finite=False);values=values.clip(0)
        z=vectors.T@(target[tr]-mean)
        for alpha in (1e-6,1e-4,.01,1.,100.,10000.):
            weights=vectors@(z/(values[:,None]+alpha));pred=k[va]@weights+mean
            error=float(np.mean((pred-target[va])**2));history.append(dict(kind=kind,alpha=alpha,mse=error))
            if best is None or error<best[0]:best=(error,kind,alpha,k[te]@weights+mean)
    y=target[te];pred=best[3];sst=np.square(y-y.mean(0)).sum()
    return dict(validation_mse=best[0],kernel=best[1],alpha=best[2],test_r2=float(1-np.square(y-pred).sum()/sst),test_rmse=float(np.sqrt(np.mean((y-pred)**2))),
                n_train=int(tr.sum()),n_validation=int(va.sum()),n_test=int(te.sum()),choices=history)


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=dict(np.load(a.diagnosis/'data.npz',allow_pickle=False));result={}
    for name in ('original','identity','local_moments','global_moments'):
        x=np.load(a.diagnosis/(name+'.npy'),mmap_mode='r')
        # All tasks below use positive records only. The original joint
        # positive/null curve scores remain in the primary diagnosis.
        split=d['split'].copy();split[~d['coupled']]='unused'
        strength=np.linalg.norm(d['coefficients'],axis=1,keepdims=True)
        records=dict(positive_curve=fit(x,d['curves'],split),positive_strength=fit(x,strength,split))
        phase_split=split.copy();phase_split[(d['family']!='h1')&(d['family']!='mixed')]='unused'
        phase=d['coefficients'][:,:2];phase=phase/np.maximum(np.linalg.norm(phase,axis=1,keepdims=True),1e-12)
        records['positive_h1_phase']=fit(x,phase,phase_split)
        result[name]=records;dump(a.output/(name+'.json'),records)
        print(name,{k:{q:v[q]for q in ('test_r2','kernel','alpha')}for k,v in records.items()},flush=True)
    dump(a.output/'scores.json',result)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
