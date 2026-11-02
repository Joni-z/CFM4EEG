"""Signal-derived metric checks: gain, polarity, noise and context duration."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from paclock_bench.probing.synthetic import generate_dataset,curve_from_coefficients
from paclock_bench.probing.scoring import score_curves
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator
from scripts.experiments.pac_metric_calibration import signal_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report={};operator=CFCOperator().eval();saved={}
    for name,seconds,gain,noise in [('base',5.,1.,0.),('gain_low',5.,.1,0.),('gain_high',5.,10.,0.),
            ('polarity',5.,-1.,0.),('strong_noise',5.,1.,1.),('context10',10.,1.,0.),('context20',20.,1.,0.)]:
        data=generate_dataset(256,0,0,seed=7103,seconds=seconds)
        x=data['signals']*gain
        if noise:
            rng=np.random.default_rng(8103)
            x=x+rng.normal(size=x.shape).astype(np.float32)*noise
        target=data['curves'].copy()
        if gain<0:
            coef=data['coefficients'].copy();coef[:,:2]*=-1
            target=curve_from_coefficients(coef)
        raw=signal_reference(x)
        estimates=[];valid=[]
        with torch.no_grad():
            for start in range(0,len(x),32):
                packet=operator(torch.from_numpy(x[start:start+32])).relation_features(stabilized=False)
                estimates.append(packet['features'][:,0,0,2,2].flatten(-2).numpy())
                valid.append(packet['harmonic_valid'][:,0,0,2,2].numpy())
        est=np.concatenate(estimates)
        report[name]=dict(signal_reference=score_curves(target,curve_from_coefficients(raw),data['coupled']),
            operator=score_curves(target,curve_from_coefficients(est),data['coupled']),
            coverage=np.concatenate(valid).mean(0).tolist(),seconds=seconds)
        saved[name+'_reference']=raw;saved[name+'_operator']=est;saved[name+'_target']=target
        print(name,report[name],flush=True)
    with (args.output/'scores.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    np.savez_compressed(args.output/'predictions.npz',**saved)


if __name__=='__main__':main()
