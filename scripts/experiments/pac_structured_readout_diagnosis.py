"""Can an all-pairs token coherence readout recover original PAC?

This is an additional representation diagnostic, not a change to the primary
benchmark. No raw signal, PAC estimate or ground-truth pair enters its input.
"""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.experiments.pac_candidate_search import probe,dump


def coherence_features(flat,*,phase_only=False):
    native=np.asarray(flat).reshape(len(flat),16,20,64,2)
    z=native[...,0]+1j*native[...,1]
    if phase_only:z=z/np.maximum(abs(z),1e-6)
    gram=np.einsum('nbpk,ncpk->nbc',z,z.conj(),optimize=True)/(20*64)
    scale=np.sqrt(np.maximum(gram.diagonal(axis1=-2,axis2=-1).real,1e-12))
    gram=gram/np.maximum(scale[:,:,None]*scale[:,None,:],1e-12)
    pairs=np.triu_indices(16,k=1)
    upper=gram[:,pairs[0],pairs[1]]
    return np.concatenate([upper.real,upper.imag],axis=1).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=dict(np.load(args.diagnosis/'data.npz',allow_pickle=False));report={}
    for name in ('original','identity'):
        x=np.load(args.diagnosis/(name+'.npy'),mmap_mode='r')
        for phase_only in (False,True):
            features=coherence_features(x,phase_only=phase_only)
            choices=[probe(features,data,d)[0]for d in (1,2)]
            best=min(choices,key=lambda r:r['validation_mse'])
            key=name+('_phase'if phase_only else '_coherence')
            report[key]=dict(feature_count=features.shape[1],choices=choices,selected_degree=best['degree'])
            print(key,best['overall'],flush=True)
    dump(args.output/'scores.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
