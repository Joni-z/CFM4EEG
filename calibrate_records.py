"""Prospective calibration with new iid null-mixture records; no test tuning."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.cfc_operator_capabilities import generate,specifications
from paclock_bench.models.paclock.frontend.measurement_uncertain import UncertainPACOperator
from paclock_bench.models.paclock.frontend.pac_calibration import PACCalibration,record_scores

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--seconds',type=int,default=10);p.add_argument('--cases',type=int,default=2000)
p.add_argument('--seed',type=int,required=True);p.add_argument('--fit',action='store_true')
p.add_argument('--calibration',type=Path);a=p.parse_args()
if not a.fit and a.calibration is None:p.error('confirmation requires --calibration')
torch.set_num_threads(3);a.output.mkdir(exist_ok=False)
op=UncertainPACOperator(block_cycles=1,shrink=False);rows=[];begin=time.monotonic()
null_specs=specifications('null')
with torch.no_grad():
 for start in range(0,a.cases,32):
  examples=[]
  for i in range(start,min(start+32,a.cases)):
   seed=a.seed+i;rng=np.random.default_rng(seed+900000000)
   spec=null_specs[int(rng.integers(2))]|dict(seconds=a.seconds)
   examples.append(generate(seed,spec))
  x=torch.tensor(np.stack([e['raw'] for e in examples]),dtype=torch.float32)[:,None]
  out=op(x);scores=record_scores(out).numpy()
  rows.extend(dict(seed=e['seed'],shape=e['spec']['shape'],score=float(s),
                   raw_sha256=hashlib.sha256(e['raw'].tobytes()).hexdigest()) for e,s in zip(examples,scores))
values=[r['score'] for r in rows];summary=dict(n=len(rows),seconds=a.seconds,runtime=time.monotonic()-begin,seed=a.seed,
 note='iid random 50/50 mixture of independent-envelope and AAC-only nulls, at target-band SNR10. Scope is this synthetic distribution, not physiological EEG significance.')
if a.fit:
 for name,confidence in [('marginal',None),('tolerance95',.95)]:
  cal=PACCalibration.fit(values,op,1,a.seconds*200,confidence=confidence)
  cal.save(a.output/(name+'.json'))
else:
 summary['detections']={}
 for name in ('marginal','tolerance95'):
  cal=PACCalibration.load(a.calibration/(name+'.json'))
  k=sum(v>cal.threshold for v in values) if cal.threshold is not None else 0
  summary['detections'][name]=dict(threshold=cal.threshold,rank=cal.rank,false_positives=k,
                                  observed_fpr=k/len(values),calibration_n=cal.n)
summary['configuration']=op.configuration
summary['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(a.output/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
