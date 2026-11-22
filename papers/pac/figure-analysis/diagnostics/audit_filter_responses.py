"""Reconstruct finite FIR responses from exported checkpoint cutoff parameters."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--bands',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bands=np.load(a.bands);fs=200;kernel=201;nfft=32768
n=2*np.pi*np.arange(-100,0)[None,:]/fs
window=0.54-0.46*np.cos(2*np.pi*np.arange(100)/kernel)
freq=np.fft.rfftfreq(nfft,1/fs);rows=[];arrays={'frequency_hz':freq}
for stage in ['initial','trained']:
 low,high=bands[stage].T;high=np.clip(high,0,fs/2);width=high-low
 left=(np.sin(high[:,None]*n)-np.sin(low[:,None]*n))/(n/2)*window
 fir=np.concatenate([left,2*width[:,None],left[:,::-1]],1)/(2*width[:,None])
 mag=np.abs(np.fft.rfft(fir,nfft));norm=mag/mag.max(1,keepdims=True)
 arrays[stage+'_magnitude']=mag
 for i in range(len(low)):
  within=np.flatnonzero(norm[i]>=1/np.sqrt(2))
  rows.append(dict(stage=stage,band=i,low_parameter_hz=float(low[i]),high_parameter_hz=float(high[i]),
    midpoint_hz=float((low[i]+high[i])/2),peak_hz=float(freq[np.argmax(mag[i])]),
    dc_relative_to_peak=float(norm[i,0]),half_power_low_hz=float(freq[within[0]]),half_power_high_hz=float(freq[within[-1]])))
with (out/'responses.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(out/'responses.npz',**arrays)
(out/'provenance.json').write_text(json.dumps(dict(source=str(Path(a.bands).resolve()),sample_rate=fs,kernel_size=kernel,nfft=nfft,
 scope='One fixed TUEV seed-0 measured checkpoint; NumPy reconstruction of SincBandpass.filters, not a dataset-wide audit.',
 interpretation='Nominal cutoff ordering is distinct from spectral separation. Magnitude-response peaks on broad bands need not equal their midpoint.'),indent=2))
print(json.dumps(rows[16:20],indent=2))
