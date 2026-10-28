"""Portable versions of the original frozen secondary PAC challenges."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator

def time_varying(output: Path, cases: int, split: str) -> None:
    torch.set_num_threads(4)
    operators=[PACCurveOperator(local=False),PACCurveOperator(local=True)]
    records=[];base=610000 if split=='dev' else 710000
    for start in range(0,cases,8):
     signals=[];truth=[]
     for seed in range(base+start,base+min(cases,start+8)):
      rng=np.random.default_rng(seed);t=np.arange(2000)/200
      f=rng.uniform(3.7,4.3);phi=2*np.pi*f*t+rng.uniform(-np.pi,np.pi)
      theta=.9*np.sin(2*np.pi*.2*t+rng.uniform(-np.pi,np.pi))
      depth=rng.uniform(.4,.7)
      raw=.8*np.cos(phi)+(1+depth*np.cos(phi-theta))*np.cos(2*np.pi*rng.uniform(31,33)*t)
      raw+=rng.normal(0,.15,len(t));signals.append(raw)
      anchors=(np.arange(40)+.5)*.25
      target=depth*np.exp(1j*np.interp(anchors,t,theta))
      truth.append(target)
     x=torch.tensor(np.stack(signals),dtype=torch.float32)[:,None]
     global_fit,local_fit=[op(x) for op in operators]
     for n,target in enumerate(truth):
      mask=local_fit.local_valid[n,0,:,2,2].numpy()
      a=global_fit.coef[n,0,:,2,2,0].numpy()/np.maximum(global_fit.baseline[n,0,:,2,2].numpy(),1e-12)
      b=local_fit.coef[n,0,:,2,2,0].numpy()/np.maximum(local_fit.baseline[n,0,:,2,2].numpy(),1e-12)
      records.append(dict(seed=base+start+n,local_fraction=float(mask.mean()),
       global_error=float(abs(a[mask]-target[mask]).mean()) if mask.any() else None,
       local_error=float(abs(b[mask]-target[mask]).mean()) if mask.any() else None))
    summary=dict(split=split,records=cases,source_seconds=10,
     mean_local_fraction=float(np.mean([r['local_fraction'] for r in records])),
     global_error=float(np.mean([r['global_error'] for r in records if r['global_error'] is not None])),
     local_error=float(np.mean([r['local_error'] for r in records if r['local_error'] is not None])),
     metric='Complex coefficient error, local-valid anchors; full-record fallback is not relabeled local.',cases=records)
    output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


def generalization(output: Path, cases: int) -> None:
    torch.set_num_threads(4)
    models=[PACCurveOperator(harmonics=1,phase_cycles=5,refine_phase=False),PACCurveOperator()]
    records=[];t=np.arange(1000)/200;central=(t>=2)&(t<=3)
    for mode in ('nonlinear_envelope','two_carriers'):
     for start in range(0,cases,16):
      examples=[];truths=[]
      for n in range(start,min(cases,start+16)):
       rng=np.random.default_rng((910000 if mode=='nonlinear_envelope' else 1010000)+n)
       f=rng.uniform(3.6,4.3);phi=2*np.pi*f*t+rng.uniform(-np.pi,np.pi)
       if mode=='nonlinear_envelope':
        theta=rng.uniform(-np.pi,np.pi);kappa=rng.uniform(.45,.8)
        env=np.exp(kappa*np.cos(phi-theta))/np.i0(kappa)
        raw=.8*np.cos(phi)+env*np.cos(2*np.pi*rng.uniform(31,33)*t)
        truth=[(2,2,env-1)]
       else:
        p1=2*np.pi*rng.uniform(2.9,3.1)*t+rng.uniform(-np.pi,np.pi)
        p2=2*np.pi*rng.uniform(4.9,5.1)*t+rng.uniform(-np.pi,np.pi)
        m1=.5*np.cos(p1-rng.uniform(-np.pi,np.pi));m2=.5*np.cos(p2-rng.uniform(-np.pi,np.pi))
        raw=.8*np.cos(p1)+.8*np.cos(p2)+(1+m1)*np.cos(2*np.pi*18*t)+(1+m2)*np.cos(2*np.pi*40*t)
        truth=[(1,0,m1),(3,3,m2)]
       noise=rng.normal(size=len(t));freq=np.fft.rfftfreq(len(t),1/200)
       noise=np.fft.irfft(np.fft.rfft(noise)/np.sqrt(1+freq),n=len(t))
       noise*=np.sqrt(np.mean(raw**2)/10)/(noise.std()+1e-12)
       examples.append(raw+noise);truths.append(truth)
      x=torch.tensor(np.stack(examples),dtype=torch.float32)[:,None]
      with torch.no_grad(): outputs=[m(x) for m in models]
      for n,truth in enumerate(truths):
       for i,j,target in truth:
        row=dict(mode=mode,record=start+n,source_index=i,target_index=j)
        for name,result in zip(('baseline','new'),outputs):
         good=bool(result.valid[n,0,10,i,j]);pred=result.modulation(i,j,10)[n,0].numpy()
         row[name+'_valid']=good
         row[name+'_rmse']=float(np.sqrt(np.mean((pred[central]-target[central])**2))) if good else 1.
        records.append(row)
    summary={mode:dict(records=cases,measured_relations=sum(r['mode']==mode for r in records),
     baseline_rmse=float(np.mean([r['baseline_rmse'] for r in records if r['mode']==mode])),
     new_rmse=float(np.mean([r['new_rmse'] for r in records if r['mode']==mode])),
     new_coverage=float(np.mean([r['new_valid'] for r in records if r['mode']==mode]))) for mode in ('nonlinear_envelope','two_carriers')}
    summary['scope']='Independent generators, all invalid measurements penalized; no parameter selection.'
    output.write_text(json.dumps(dict(summary=summary,cases=records),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('challenge',choices=['time-varying','generalization'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=200)
    parser.add_argument('--split',choices=['dev','confirm'],default='confirm')
    args=parser.parse_args()
    if args.cases<=0: parser.error('cases must be positive')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.challenge=='time-varying':time_varying(args.output,args.cases,args.split)
    else:generalization(args.output,args.cases)
