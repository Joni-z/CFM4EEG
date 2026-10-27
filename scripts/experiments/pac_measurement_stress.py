"""Post-freeze measurement stress report; no classifier labels or fitting."""
import argparse,json,math,time,sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import iirnotch,filtfilt,freqz
from paclock_bench.models.paclock.frontend.measurement import PhaseAmplitudeMeasurement,harmonic_regression

torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
out=args.output
out.mkdir(parents=True,exist_ok=False)
records=[]
for fs,low,high,duration in [(200,2.1,32.,8.3),(200,5.3,37.,5.5),(100,5.3,31.,5.5)]:
    upper=min(75.,fs/2-1)
    module=PhaseAmplitudeMeasurement([low,high],sample_rate=fs,patch_len=round(.25*fs),passband=(.5,upper))
    t=np.arange(round(fs*duration))/fs
    rng=np.random.default_rng(271828+fs)
    examples=[];info=[]
    for mode in ['constant','ramp','coupled','second_harmonic','chirped_coupled','transient','high_only','dead']:
        phi=2*np.pi*low*t+.3
        if mode=='chirped_coupled':phi=phi+.5/.3*np.sin(2*np.pi*.3*t)
        env=np.ones_like(t)
        if mode in ('coupled','chirped_coupled'):env=1+.6*np.cos(phi-.8)
        if mode=='second_harmonic':env=1+.6*np.cos(2*phi-.8)
        if mode=='ramp':env=1+.4*(t/duration-.5)
        if mode=='transient':env=1+2*np.exp(-.5*((t-duration/2)/.15)**2)
        raw=.8*np.cos(phi)+env*np.cos(2*np.pi*high*t+.7)
        if mode=='high_only':raw=np.cos(2*np.pi*high*t+.7)
        if mode=='dead':raw*=0
        examples.append(raw);info.append(mode)
    x=torch.tensor(np.array(examples),dtype=torch.float32)[:,None]
    result=module(x)
    for i,mode in enumerate(info):
        valid=result.valid[i,0,:,0,1];depth=result.depth[i,0,:,0,1]
        records.append(dict(fs=fs,low=low,high=high,seconds=duration,mode=mode,
            valid_fraction=float(valid.float().mean()),mean_depth=float(depth[valid].mean()) if valid.any() else None,
            phase_deg=float(result.coef[i,0,:,0,1][valid].mean().angle()*180/math.pi) if valid.any() else None))
    # Preserve precisely the entire raw Fourier magnitude, randomize phases.
    raw=examples[4];spectrum=np.fft.rfft(raw)
    surrogates=[]
    for k in range(48):
        phase=rng.uniform(-np.pi,np.pi,len(spectrum));phase[0]=np.angle(spectrum[0])
        if len(raw)%2==0:phase[-1]=np.angle(spectrum[-1])
        surrogates.append(np.fft.irfft(abs(spectrum)*np.exp(1j*phase),n=len(raw)))
    sur=np.array(surrogates)
    r=module(torch.tensor(sur,dtype=torch.float32)[:,None])
    weights=r.valid[:,0,:,0,1]
    estimates=(r.depth[:,0,:,0,1]*weights).sum(-1)/weights.sum(-1).clamp_min(1)
    np.savez_compressed(out/f'surrogates-{fs}-{low}.npz',raw=raw,surrogates=sur,estimated_depth=estimates.numpy(),valid=weights.numpy())
    records.append(dict(fs=fs,low=low,high=high,mode='spectrum_preserving_surrogates',
        fourier_magnitude_max_error=float(np.max(abs(abs(np.fft.rfft(sur,axis=-1))-abs(spectrum)))),
        mean_depth=float(estimates.mean()),q95_depth=float(torch.quantile(estimates,.95))))

# Context, Nyquist and all-sample safety checks.
module=PhaseAmplitudeMeasurement([6.,40.])
short=module(torch.randn(1,1,100))
assert not short.valid.any()
bad=torch.randn(1,1,1000);bad[0,0,300]=float('nan')
assert not module(bad).valid.any()
try: PhaseAmplitudeMeasurement([6.,40.],sample_rate=100,passband=(.5,75.))
except ValueError: pass
else: raise AssertionError('accepted passband above Nyquist')
for mode in ['constant','ramp']:
    assert max(r['mean_depth'] or 0 for r in records if r['mode']==mode)<.03

# A notch can erase a modulation sideband even though the carrier survives.
t=np.arange(2200)/200;phi=2*np.pi*6*t+.3
raw=np.cos(phi)+(1+.6*np.cos(phi-.8))*np.cos(2*np.pi*54*t+.7)
b,a=iirnotch(60,30,fs=200);notched=filtfilt(b,a,raw).copy()
notch_rows=[]
for name,x,exclude in [('clean',raw,()),('notched_unprotected',notched,()),('notched_guarded',notched,((56,64),))]:
    m=PhaseAmplitudeMeasurement([6.,54.],excluded_intervals=exclude)
    r=m(torch.tensor(x,dtype=torch.float32)[None,None]);valid=r.valid[0,0,:,0,1]
    notch_rows.append(dict(mode=name,valid_fraction=float(valid.float().mean()),depth=float(r.depth[0,0,:,0,1][valid].mean()) if valid.any() else None))
assert abs(notch_rows[0]['depth']-.6)<.02 and notch_rows[1]['depth']<.5 and notch_rows[2]['valid_fraction']==0
transfer=[]
for rate in [200,250,256,512,1000]:
    b,a=iirnotch(60,30,fs=rate);f=np.linspace(1,99,20000);_,h=freqz(b,a,worN=f,fs=rate)
    gain=float((abs(h[(f<56)|(f>64)])**2).min())
    assert gain>.9
    transfer.append(dict(native_fs=rate,minimum_notch_zero_phase_gain_outside_guard=gain))
(out/'notch-validation.json').write_text(json.dumps(dict(recovery=notch_rows,transfer=transfer),indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
print(json.dumps(records,indent=2,allow_nan=False))
