"""NumPy FFT and regression replay of final raw-signal operator outputs."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.cfc_operator_benchmark import generate,CONDITIONS
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator



def audit(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    op=PACCurveOperator()
    rows=[]
    for ci,condition in enumerate(CONDITIONS):
     for k in range(4):
      e=generate(410000+ci*1000+k,condition)
      raw=e['raw'].astype(np.float32);result=op(torch.tensor(raw)[None,None])
      i,j=e['source_index'],e['target_index'];fc=float(result.source_frequency[0,0,i])
      spectrum=abs(np.fft.rfft(raw*np.hanning(len(raw)+1)[:-1]))**2
      frequencies=np.fft.rfftfreq(len(raw),1/200)
      allowed=(frequencies>=float(op.search_low[i]))&(frequencies<=float(op.search_high[i]))
      peak=np.argmax(np.where(allowed,spectrum,-1));logs=np.log(np.maximum(spectrum,1e-20))
      correction=.5*(logs[peak-1]-logs[peak+1])/(logs[peak-1]-2*logs[peak]+logs[peak+1])
      estimate=np.clip((peak+np.clip(correction,-.5,.5))*200/len(raw),float(op.search_low[i]),float(op.search_high[i]))
      radius=math.ceil(3*9/(2*np.pi*fc)*200);t=np.arange(-radius,radius+1)/200
      sigma=9/(2*np.pi*fc);wave=np.exp(2j*np.pi*fc*t)
      kernel=np.exp(-t*t/(2*sigma*sigma))*(wave-np.exp(-81/2))
      kernel=2*kernel/abs(np.sum(kernel*wave.conj()))
      source=np.convolve(raw,kernel,'full')[radius:radius+len(raw)]
      ak=op.analysis.amplitude_kernels[op.n_source+j].numpy()
      env=abs(np.convolve(raw,ak,'full')[len(ak)//2:len(ak)//2+len(raw)])
      phase=source/np.maximum(abs(source),1e-12)
      guard=max(radius,int(op.analysis.amplitude_guard[op.n_source+j]))
      mask=(np.arange(len(raw))>=guard)&(np.arange(len(raw))<len(raw)-guard)
      mask&=abs(source)>max(np.sqrt(np.mean(raw**2))*1e-4,1e-12)
      order=int(op.harmonic_support[i,j].sum());w=mask/max(mask.sum(),1)
      powers=[phase**h for h in range(1,order+1)]
      X=np.stack([a for p in powers for a in (p.real,p.imag)],-1)
      mx=np.sum(X*w[:,None],0);my=np.sum(env*w)
      u=np.linspace(-1,1,len(raw),dtype=np.float32);u=u-np.sum(w*u)
      xc=X-mx;yc=env-my
      variance=np.sum(w*u*u);npen=1e-8*max(variance,1e-4)
      xr=xc-u[:,None]*(np.sum(xc*(w*u)[:,None],0)/(variance+npen))
      yr=yc-u*(np.sum(yc*w*u)/(variance+npen))
      G=xr.T@(w[:,None]*xr);rhs=xr.T@(w*yr)
      penalty=.001*max(np.trace(G)/(2*order),1e-6)+1e-10
      beta=np.linalg.solve(G+penalty*np.eye(2*order),rhs)
      baseline=my-mx@beta
      actual=torch.view_as_real(result.coef[0,0,10,i,j,:order]).flatten().numpy()
      coef_error=float(np.max(abs(actual-beta)))
      phase_error=float(np.max(abs(result.source_phase[0,0,i].numpy()[mask]-phase[mask])))
      base_error=abs(float(result.baseline[0,0,10,i,j])-baseline)
      row=dict(condition=condition,seed=e['seed'],coefficient_max_error=coef_error,
               baseline_error=base_error,phase_max_error=phase_error,
               frequency_difference_hz=abs(fc-estimate))
      rows.append(row)
      assert coef_error<1e-5 and base_error<1e-5 and phase_error<2e-5 and abs(fc-estimate)<1e-4,row
    summary=dict(cases=len(rows),max_coefficient_error=max(r['coefficient_max_error'] for r in rows),
                 max_baseline_error=max(r['baseline_error'] for r in rows),
                 max_phase_error=max(r['phase_max_error'] for r in rows),
                 max_frequency_error_hz=max(r['frequency_difference_hz'] for r in rows),
                 audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases_checked=rows)
    output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases_checked'},indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
