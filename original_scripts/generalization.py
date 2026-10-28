"""Frozen secondary generators: nonlinear envelope and two simultaneous PACs."""
import json,math
from pathlib import Path
import numpy as np
import torch
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator

root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
models=[PACCurveOperator(harmonics=1,phase_cycles=5,refine_phase=False),PACCurveOperator()]
records=[];t=np.arange(1000)/200;central=(t>=2)&(t<=3)
for mode in ('nonlinear_envelope','two_carriers'):
 for start in range(0,200,16):
  examples=[];truths=[]
  for n in range(start,min(200,start+16)):
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
summary={mode:dict(records=200,measured_relations=sum(r['mode']==mode for r in records),
 baseline_rmse=float(np.mean([r['baseline_rmse'] for r in records if r['mode']==mode])),
 new_rmse=float(np.mean([r['new_rmse'] for r in records if r['mode']==mode])),
 new_coverage=float(np.mean([r['new_valid'] for r in records if r['mode']==mode]))) for mode in ('nonlinear_envelope','two_carriers')}
summary['scope']='Independent generators, all invalid measurements penalized; no parameter selection.'
(root/'artifacts'/'generalization-challenge.json').write_text(json.dumps(dict(summary=summary,cases=records),indent=2)+'\n')
print(json.dumps(summary,indent=2))
