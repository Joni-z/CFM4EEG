"""PAC recovery with stochastic fast carriers, beyond coherent toy sinusoids."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.cfc_operator_capabilities import band_power
from paclock_bench.models.paclock.frontend.measurement_uncertain import UncertainPACOperator

p=argparse.ArgumentParser();p.add_argument('--cases',type=int,default=64)
p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(3)
op=UncertainPACOperator(block_cycles=1,shrink=False);rows=[];begin=time.monotonic()
with torch.no_grad():
 for ci,(shape,depth) in enumerate((s,d) for s in ('single','double','mixed') for d in (.1,.4)):
  for start in range(0,a.cases,16):
   examples=[]
   for k in range(start,min(start+16,a.cases)):
    seed=a.seed+ci*10000+k;rng=np.random.default_rng(seed);t=np.arange(2000)/200
    f=rng.uniform(3.8,4.2);fc=rng.uniform(31.7,32.3)
    phi=2*np.pi*f*t+rng.uniform(-np.pi,np.pi);theta=rng.uniform(-np.pi,np.pi);theta2=rng.uniform(-np.pi,np.pi)
    d1=depth if shape=='single' else (depth/2 if shape=='mixed' else 0.)
    d2=depth if shape=='double' else (depth/2 if shape=='mixed' else 0.)
    truth=d1*np.cos(phi-theta)+d2*np.cos(2*phi-theta2)
    fq=np.fft.fftfreq(len(t),1/200)
    z=np.fft.ifft(np.fft.fft(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))*np.exp(-(np.abs(fq)/1.5)**8))
    z=z/np.mean(np.abs(z))
    high=(1+truth)*(z*np.exp(2j*np.pi*fc*t)).real
    low=.8*np.cos(phi)
    noise=rng.normal(size=len(t));freq=np.fft.rfftfreq(len(t),1/200)
    noise=np.fft.irfft(np.fft.rfft(noise)/np.sqrt(1+freq),n=len(t))
    lo,hi=fc-2*f-2,fc+2*f+2
    noise*=np.sqrt(band_power(high,lo,hi)/band_power(noise,lo,hi)/10)
    raw=low+high+noise
    examples.append((seed,raw,truth))
   x=torch.tensor(np.stack([e[1] for e in examples]),dtype=torch.float32)[:,None]
   out=op(x);phase=out.source_phase[:,0,2].numpy()
   for n,(seed,raw,truth) in enumerate(examples):
    b=float(out.baseline[n,0,20,2,2]);coef=out.raw_coef[n,0,20,2,2].numpy()/max(b,1e-12)
    stable=coef*out.reliability[n,0,20,2,2].numpy()
    for name,cc in [('raw',coef),('stable',stable)]:
     estimate=sum((cc[k].conjugate()*phase[n]**(k+1)).real for k in range(2))
     error=float(np.sqrt(np.mean((estimate[800:1200]-truth[800:1200])**2)))
     rows.append(dict(seed=seed,shape=shape,depth=depth,variant=name,rmse=error,
                      valid=bool(out.valid[n,0,20,2,2]),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
summary={}
for shape in ('single','double','mixed'):
 for depth in (.1,.4):
  summary[f'{shape}/{depth}']={name:float(np.mean([r['rmse'] for r in rows if r['shape']==shape and r['depth']==depth and r['variant']==name])) for name in ('raw','stable')}
out=dict(summary=summary,signals=len(rows)//2,seconds=time.monotonic()-begin,
         truth='Population phase-conditional modulation, stochastic independent complex-Gaussian baseband carrier; not deterministic instantaneous envelope truth.',
         driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=rows)
(a.output/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
