"""Additional physical-pair fixtures; original analytic fixture stays unchanged.

Reuse its analytic phase/envelope construction, population targets, balanced
families and disjoint split RNGs. Retune oscillator frequencies only inside the
generator. Estimators never receive the per-record parameters below.
"""
import argparse
from pathlib import Path
import numpy as np
from paclock_bench.probing.synthetic import generate_dataset, _phase

PAIRS=((3.,24.),(3.,40.),(4.,32.),(5.,40.))


def generate_pair(source, target, *, seed, n_train=1024, n_val=256, n_test=512):
    data=generate_dataset(n_train,n_val,n_test,seed=seed)
    t=np.arange(1000)/200
    rng=np.random.default_rng(np.random.SeedSequence([seed,7391]))
    source_hz=rng.uniform(source-.12,source+.12,len(data['signals']))
    independent_hz=rng.uniform(source-.12,source+.12,len(data['signals']))
    target_hz=rng.uniform(target-.3,target+.3,len(data['signals']))
    for i in range(len(source_hz)):
        old_phi=_phase(t,data['source_hz'][i],data['source_phase0'][i],data['source_drift_hz'][i],data['source_drift_phase0'][i])
        old_psi=_phase(t,data['envelope_hz'][i],data['envelope_phase0'][i],data['envelope_drift_hz'][i],data['envelope_drift_phase0'][i])
        c=data['envelope_coefficients'][i]
        def envelope(phase):
            return 1+c[0]*np.cos(phase)+c[1]*np.sin(phase)+c[2]*np.cos(2*phase)+c[3]*np.sin(2*phase)
        old_low=data['low_amplitude'][i]*np.cos(old_phi)
        old_high=data['high_gain'][i]*envelope(old_psi)*np.cos(2*np.pi*data['target_hz'][i]*t+data['carrier_phase0'][i])
        noise=data['signals'][i,0].astype(np.float64)-old_low-old_high
        phi=_phase(t,source_hz[i],data['source_phase0'][i],data['source_drift_hz'][i],data['source_drift_phase0'][i])
        null=data['family'][i]=='null'
        ehz=independent_hz[i] if null else source_hz[i]
        psi=_phase(t,ehz,data['envelope_phase0'][i],data['envelope_drift_hz'][i],data['envelope_drift_phase0'][i])
        low=data['low_amplitude'][i]*np.cos(phi)
        high=data['high_gain'][i]*envelope(psi)*np.cos(2*np.pi*target_hz[i]*t+data['carrier_phase0'][i])
        clean=low+high
        noise*=np.sqrt(np.mean(clean**2)/10**(data['snr_db'][i]/10)/np.mean(noise**2))
        data['signals'][i,0]=(clean+noise).astype(np.float32)
        data['envelope_hz'][i]=ehz
        for key,value in [('low_power',np.mean(low**2)),('high_power',np.mean(high**2)),
                          ('clean_power',np.mean(clean**2)),('noise_power',np.mean(noise**2)),
                          ('minimum_envelope',envelope(psi).min())]:
            data[key][i]=value
    data['source_hz']=source_hz;data['target_hz']=target_hz
    data['source_guard_hz']=source_hz+6*data['source_drift_hz']
    data['guarded_sideband_low']=target_hz-2*data['envelope_hz']-6*data['envelope_drift_hz']
    data['guarded_sideband_high']=target_hz+2*data['envelope_hz']+6*data['envelope_drift_hz']
    data['format_version']=np.asarray('analytic-pac-physical-pair-v2')
    data['query_pair']=np.array([source,target])
    data['null_warning']=np.asarray('Narrow frequency jitter increases finite-window null identifiability difficulty; report reference AUROC, not a perfect-detection gate.')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=float,required=True);p.add_argument('--target',type=float,required=True)
    p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:np.savez_compressed(f,**generate_pair(args.source,args.target,seed=args.seed))
