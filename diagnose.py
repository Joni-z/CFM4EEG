"""Oracle factorization of phase extraction and envelope extraction errors."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator,curve_regression
from paclock_bench.models.paclock.frontend.measurement import linear_analytic_filter
from scripts.experiments.cfc_operator_capabilities import generate

torch.set_num_threads(4)
root=Path(__file__).parent
op=PACCurveOperator();rows=[];begin=time.monotonic()
specs=[dict(seconds=10,source=2,target=2,depth=d,snr_db=s,shape=shape)
       for shape,d in [('single',.05),('single',.25),('single',.5),('double',.5),('mixed',.5)]
       for s in (0.,10.,20.)]
with torch.no_grad():
 for ci,spec in enumerate(specs):
  for start in range(0,64,16):
   examples=[generate(22000000+ci*1000+k,spec) for k in range(start,start+16)]
   x=torch.tensor(np.stack([e['raw'] for e in examples]),dtype=torch.float32)[:,None]
   source,freq,guard=op._source_analysis(x)
   measured_phase=source[:,:,2]/source[:,:,2].abs().clamp_min(1e-12)
   measured_env=linear_analytic_filter(x,op.analysis.amplitude_kernels[6:]).abs()[:,:,2]
   true_phase=torch.tensor(np.stack([np.exp(1j*e['phi']) for e in examples]),dtype=torch.complex64)[:,None]
   true_env=torch.tensor(np.stack([1+e['modulation'] for e in examples]),dtype=torch.float32)[:,None]
   length=x.shape[-1];times=torch.arange(length)
   guard=guard[:,:,2].clamp_min(int(op.analysis.amplitude_guard[8]))
   mask=((times>=guard[...,None])&(times<length-guard[...,None])).float()
   for pname,phase in [('measured',measured_phase),('oracle',true_phase)]:
    for aname,env in [('measured',measured_env),('oracle',true_env)]:
     fit=curve_regression(phase,env,mask)
     pred=sum((fit['coef'][...,k,None].conj()*phase**(k+1)).real for k in range(2))/fit['baseline'][...,None]
     for n,e in enumerate(examples):
      truth=e['modulation'];central=slice(800,1200)
      rows.append(dict(group=f"{spec['shape']}/{spec['depth']}/{spec['snr_db']}",seed=e['seed'],
       phase=pname,envelope=aname,rmse=float(np.sqrt(np.mean((pred[n,0].numpy()[central]-truth[central])**2))),
       depth1=float(abs(fit['coef'][n,0,0])/fit['baseline'][n,0]),truth_depth1=e['d1']))
summary={}
for group in sorted({r['group'] for r in rows}):
 summary[group]={p+'/'+a:float(np.mean([r['rmse'] for r in rows if r['group']==group and r['phase']==p and r['envelope']==a]))
                 for p in ('measured','oracle') for a in ('measured','oracle')}
out=dict(seconds=time.monotonic()-begin,signals=len(rows)//4,summary=summary,records=rows,
         note='Oracle arms are diagnostics only, not candidate performance.',
         evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'oracle-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
