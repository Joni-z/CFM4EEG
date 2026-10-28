"""Secondary frozen challenge: continuously changing preferred PAC phase."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator

parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','confirm'],required=True)
parser.add_argument('--cases',type=int,default=100);args=parser.parse_args()
torch.set_num_threads(4);root=Path(__file__).resolve().parents[1]
operators=[PACCurveOperator(local=False),PACCurveOperator(local=True)]
records=[];base=610000 if args.split=='dev' else 710000
for start in range(0,args.cases,8):
 signals=[];truth=[]
 for seed in range(base+start,base+min(args.cases,start+8)):
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
summary=dict(split=args.split,records=args.cases,source_seconds=10,
 mean_local_fraction=float(np.mean([r['local_fraction'] for r in records])),
 global_error=float(np.mean([r['global_error'] for r in records if r['global_error'] is not None])),
 local_error=float(np.mean([r['local_error'] for r in records if r['local_error'] is not None])),
 metric='Complex coefficient error, local-valid anchors; full-record fallback is not relabeled local.',cases=records)
(root/'artifacts'/f'time-varying-{args.split}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))
