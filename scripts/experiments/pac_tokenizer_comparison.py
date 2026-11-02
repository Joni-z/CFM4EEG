"""Same 9x64 readout budget for four real frozen tokenizer endpoints."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paclock_bench.probing.benchmark import load_dataset,file_sha256
from paclock_bench.probing.features import (load_project_tokenizer,project_tokens,
    load_tfm_tokenizer,tokenize_tfm,linear_token_budget)
from scripts.experiments.pac_candidate_search import probe,dump,cfft_sequence


def main():
    p=argparse.ArgumentParser()
    for key in ('dataset','candidate-cache','wavepatch','tfm-root','tfm-checkpoint','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--candidate-mode',choices=['residual','context'],default='residual')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=load_dataset(args.dataset)
    cached=load_dataset(args.candidate_cache/'data.npz')
    for key in ('signals','curves','sample_id','split'):
        if not np.array_equal(data[key],cached[key]):raise ValueError('candidate cache mismatch: '+key)
    cache_report=json.loads((args.candidate_cache/'scores.json').read_text())
    base=np.load(args.candidate_cache/'base.npy',mmap_mode='r')
    residual=np.load(args.candidate_cache/'coefficient.npy',mmap_mode='r')
    scale=cache_report['train_derived_residual_scales']['coefficient']
    tfm,tfm_meta=load_tfm_tokenizer(args.tfm_root,args.tfm_checkpoint)
    wave,wave_meta=load_project_tokenizer(args.wavepatch/'original-training.yaml',
        args.wavepatch/'original-best.pt',(16,1000),normalization='div100')
    n=len(base)
    candidate_name='pac_residual_3pct' if args.candidate_mode=='residual' else 'pac_context_slot'
    if args.candidate_mode=='context':
        tr=data['split']=='train'
        context=np.array(residual[tr]).reshape(-1,16,20,128).mean(2,keepdims=True)
        scale=float(np.sqrt(np.mean(np.array(base[tr])**2)))/max(float(np.sqrt(np.mean(context**2))),1e-12)
    features={k:[]for k in ('original_cfft',candidate_name,'tfm_released_multi','wavepatch')}
    start=time.monotonic()
    for first in range(0,n,32):
        b=np.array(base[first:first+32]);r=np.array(residual[first:first+32])*scale
        bn=np.sqrt(np.mean(b*b,axis=1,keepdims=True));rn=np.sqrt(np.mean(r*r,axis=1,keepdims=True))
        r*=np.minimum(1.,bn/np.maximum(rn,1e-12));candidate=b+.03*r
        features['original_cfft'].append(linear_token_budget(cfft_sequence(b),9,64,seed=0))
        if args.candidate_mode=='context':
            context=np.array(residual[first:first+32]).reshape(len(b),16,20,128).mean(2,keepdims=True)*scale
            candidate=np.concatenate([b.reshape(len(b),16,20,128),context],axis=2)
        features[candidate_name].append(linear_token_budget(cfft_sequence(candidate,patches=21 if args.candidate_mode=='context' else 20),9,64,seed=0))
        x=torch.from_numpy(data['signals'][first:first+32])*20
        with torch.no_grad():
            a=tokenize_tfm(tfm,x).numpy()
            w=project_tokens(wave,x/100,kind='wavepatch').numpy()
        features['tfm_released_multi'].append(linear_token_budget(a,9,64,seed=0))
        features['wavepatch'].append(linear_token_budget(w,9,64,seed=0))
        if first%512==0:print('extracted',first,'/',n,round(time.monotonic()-start,1),flush=True)
    report=dict(scope='common-budget frozen PAC readability; training corpora differ; no clinical claims',
        dataset_sha256=file_sha256(args.dataset),candidate_mode=args.candidate_mode,budget=[9,64],projection_seed=0,
        tfm=tfm_meta,wavepatch=wave_meta,cfft=cache_report['checkpoint'],rows={})
    test=data['split']=='test'
    for name,parts in features.items():
        x=np.concatenate(parts);np.save(args.output/(name+'-features.npy'),x,allow_pickle=False)
        assert x.shape==(n,576)
        row=[];predictions=[]
        for degree in (1,2):
            result,pred=probe(x,data,degree);row.append(result);predictions.append(pred)
        chosen=min(row,key=lambda r:r['validation_mse'])
        report['rows'][name]=dict(readouts=row,selected_degree=chosen['degree'])
        np.savez_compressed(args.output/(name+'-predictions.npz'),sample_id=data['sample_id'][test],
            predictions=predictions,target=data['curves'][test],coupled=data['coupled'][test])
        dump(args.output/(name+'.json'),report['rows'][name])
        print(name,chosen['overall'],flush=True)
    report['elapsed_seconds']=time.monotonic()-start
    dump(args.output/'scores.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
