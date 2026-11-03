"""Common-budget screening of content-only PAC fusion; no clinical training."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paclock_bench.probing.features import load_project_tokenizer,linear_token_budget
from scripts.experiments.pac_candidate_search import probe,dump,cfft_sequence
from scripts.experiments.pac_content_fusion import PACContentFusion


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['trace_gain','carrier_binding','aligned_binding','radial_binding'],default=['trace_gain','carrier_binding','aligned_binding'])
    p.add_argument('--gains',type=float,nargs='+',default=[.03,.1,.3])
    p.add_argument('--native',action='store_true',help='Also fit readouts on all native content coordinates')
    p.add_argument('--calibrate',action='store_true',help='Calibrate binding RMS per band using training inputs only')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=dict(np.load(args.dataset,allow_pickle=False));n=len(data['signals'])
    m,meta=load_project_tokenizer(args.reference/'original-training.yaml',args.reference/'original-best.pt',(16,1000),normalization='div100')
    fusion=PACContentFusion(m.frontend).eval();specs=[('original',None,0.,False)]
    if args.calibrate:
        if args.modes!=['radial_binding']:raise ValueError('calibration currently applies to radial_binding only')
        count=0;delta_power=torch.zeros(16,dtype=torch.float64);base_power=delta_power.clone()
        ids=np.flatnonzero(data['split']=='train')
        with torch.no_grad():
            for start in range(0,len(ids),32):
                x=torch.from_numpy(data['signals'][ids[start:start+32]])*.2
                a=fusion.prepare(x);out=fusion.fuse(a,mode='radial_binding',gain=1.)
                base_power+=a['baseline'].double().square().sum((0,1,3,4))
                delta_power+=(out-a['baseline']).double().square().sum((0,1,3,4));count+=len(x)
            scale=torch.where(delta_power>1e-20,(base_power/delta_power.clamp_min(1e-20)).sqrt(),0.)
            fusion.residual_scale.copy_(scale.float().view(1,1,16,1,1))
        print('train-only band RMS calibration',scale.tolist(),'records',count,flush=True)
    for mode in args.modes:
        specs.extend((f'{mode}_{g}',mode,g,False)for g in args.gains)
        specs.append((mode+'_wrong_0.1',mode,.1,True))
    features={name:np.empty((n,576),np.float32)for name,*_ in specs}
    native={name:np.lib.format.open_memmap(args.output/(name+'-native.npy'),mode='w+',dtype=np.float32,shape=(n,40960))for name,*_ in specs}if args.native else {}
    deviation={name:[]for name,*_ in specs}
    shuffle=np.arange(n);rng=np.random.default_rng(7131)
    for split in ('train','val','test'):
        ids=np.flatnonzero(data['split']==split);shuffle[ids]=np.roll(ids,int(rng.integers(1,len(ids))))
    begin=time.monotonic()
    with torch.no_grad():
        for start in range(0,n,32):
            x=torch.from_numpy(data['signals'][start:start+32])*.2
            a=fusion.prepare(x);b=a['baseline']
            other=torch.from_numpy(data['signals'][shuffle[start:start+32]])*.2
            wrong=fusion.coefficients(fusion.operator(other))
            norm=b.square().mean((-1,-2)).sqrt().clamp_min(1e-12)
            for name,mode,gain,bad in specs:
                out=b if mode is None else fusion.fuse(a,mode=mode,gain=gain,coefficient=wrong if bad else None)
                assert out.shape==b.shape and bool(torch.isfinite(out).all())
                features[name][start:start+len(x)]=linear_token_budget(cfft_sequence(out.numpy().reshape(len(x),-1)),9,64,seed=0)
                if native:native[name][start:start+len(x)]=out.numpy().reshape(len(x),-1)
                d=(out-b).square().mean((-1,-2)).sqrt()/norm
                deviation[name].append(d.numpy())
            if start%512==0:print('extracted',start,'/',n,round(time.monotonic()-begin,1),flush=True)
    report=dict(checkpoint=meta,scope='20 content positions only; no PAC-only additive vector',budget=[9,64],rows={},native_rows={})
    dump(args.output/'protocol.json',dict(modes=args.modes,gains=args.gains,native=args.native,calibrate=args.calibrate,
        residual_scales=fusion.residual_scale.flatten().tolist(),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        fusion_sha256=hashlib.sha256(Path(__file__).with_name('pac_content_fusion.py').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='validation only; architecture development, not final confirmation'))
    test=data['split']=='test'
    for name,*_ in specs:
        x=features[name];np.save(args.output/(name+'-features.npy'),x,allow_pickle=False)
        choices=[];predictions=[]
        for degree in (1,2):
            r,pred=probe(x,data,degree);choices.append(r);predictions.append(pred)
        chosen=min(choices,key=lambda v:v['validation_mse']);dev=np.concatenate(deviation[name])
        row=dict(choices=choices,selected_degree=chosen['degree'],max_band_relative_rms=float(dev.max()),mean_band_relative_rms=float(dev.mean()))
        report['rows'][name]=row;dump(args.output/(name+'.json'),row)
        np.savez_compressed(args.output/(name+'-predictions.npz'),predictions=predictions,sample_id=data['sample_id'][test],target=data['curves'][test],coupled=data['coupled'][test])
        print(name,chosen['overall'],'bandmax',row['max_band_relative_rms'],flush=True)
        if native:
            native[name].flush();choices=[];predictions=[]
            for degree in (1,2):
                r,pred=probe(native[name],data,degree);choices.append(r);predictions.append(pred)
            chosen=min(choices,key=lambda v:v['validation_mse'])
            report['native_rows'][name]=dict(choices=choices,selected_degree=chosen['degree'])
            np.savez_compressed(args.output/(name+'-native-predictions.npz'),predictions=predictions,sample_id=data['sample_id'][test],target=data['curves'][test],coupled=data['coupled'][test])
            print(name,'native',chosen['overall'],flush=True)
    report['seconds']=time.monotonic()-begin;dump(args.output/'scores.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
