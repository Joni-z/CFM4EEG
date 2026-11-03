"""Locate PAC recovery failures in the original frozen rotation tokenizer."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from paclock_bench.probing.features import load_project_tokenizer,linear_token_budget
from paclock_bench.models.paclock.frontend.analytic import hilbert,phase_amplitude
from paclock_bench.models.paclock.frontend.triaxial import patch_pac_vector
from scripts.experiments.pac_candidate_search import probe,dump,cfft_sequence


def internal_checks(frontend):
    torch.manual_seed(7301)
    p=torch.randn(2,1,3,16,64,dtype=torch.complex64)
    a=torch.randn(2,1,3,16,64);w=torch.randn_like(a)
    z=torch.randn(2,1,3,16,16,dtype=torch.complex64)
    out=frontend._pac_interaction(p,a,z,w)
    scaled=frontend._pac_interaction(p,a,z*3,w)
    delta=torch.linspace(-1.2,1.2,16)
    shifted=frontend._pac_interaction(p*torch.exp(1j*delta)[None,None,None,:,None],a,
        z*torch.exp(1j*delta)[None,None,None,:,None],w)
    ph=torch.exp(1j*torch.arange(1000)/200*2*torch.pi*4)
    h2_amp=1+.4*(ph**2).real
    h2moment=(ph*(h2_amp-h2_amp.mean())).mean()
    return dict(scale_pac_only_max_token_error=float(abs(out-scaled).max()),
        phase_reference_nonroot_max_error=float(abs(out[...,1:,:]-shifted[...,1:,:]).max()),
        h2_population_example_first_moment_abs=float(abs(h2moment)),
        scope='Conditional internal-path identities with content held fixed, not absence of information from the complete raw-signal representation')


def sideband_checks(frontend,data):
    kernel=frontend.sinc.filters().squeeze(1).double()
    t=torch.arange(kernel.shape[1],dtype=torch.float64)/200
    def gain(f):
        f=torch.as_tensor(f,dtype=torch.float64)
        arg=2*torch.pi*f[:,None]*t
        return ((arg.cos()@kernel.T)**2+(arg.sin()@kernel.T)**2).sqrt()
    center=gain(data['target_hz']);band=center.argmax(1);index=torch.arange(len(band))
    reference=center[index,band]
    report={}
    for h in (1,2):
        lo=gain(data['target_hz']-h*data['source_hz'])[index,band]
        hi=gain(data['target_hz']+h*data['source_hz'])[index,band]
        ratio=torch.minimum(lo,hi)/reference.clamp_min(1e-12)
        report['h'+str(h)]=dict(min_sideband_to_carrier_gain_quantiles=torch.quantile(ratio,torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist(),
            both_sidebands_at_least_90pct_fraction=float((ratio>=.9).double().mean()))
    report['definition']='Pick the strongest learned filter at each carrier; inspect its actual response at carrier +/- h*source. Ground-truth frequencies used for diagnostic response evaluation only.'
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    source=dict(np.load(args.dataset,allow_pickle=False));ids=np.concatenate([np.flatnonzero(source['split']==s)[:n]for s,n in [('train',1024),('val',256),('test',512)]])
    data={k:v[ids]if v.ndim and len(v)==len(source['signals'])else v for k,v in source.items()}
    np.savez_compressed(args.output/'data.npz',**data)
    m,meta=load_project_tokenizer(args.reference/'original-training.yaml',args.reference/'original-best.pt',(16,1000),normalization='div100')
    f=m.frontend;mask=torch.triu(torch.ones(16,16,dtype=torch.bool),diagonal=1)
    report=dict(checkpoint=meta,internal=internal_checks(f),sidebands=sideband_checks(f,data),rows={})
    dump(args.output/'structural-checks.json',{k:v for k,v in report.items()if k!='rows'})
    arrays={k:[]for k in ('original','identity','local_moments','global_moments')}
    begin=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),32):
            x=torch.from_numpy(data['signals'][start:start+32])*.2;b,c,length=x.shape
            band=f.sinc(x.reshape(b*c,1,length)).reshape(b,c,16,length)
            phase,amp=phase_amplitude(hilbert(band));z=patch_pac_vector(phase,amp,20,f.normalize)
            base=f._interaction_tokens(phase,amp,z,filtered=band)
            f.pac_token_mode='identity';identity=f._interaction_tokens(phase,amp,z,filtered=band);f.pac_token_mode='measured'
            arrays['original'].append(base.numpy().reshape(b,-1));arrays['identity'].append(identity.numpy().reshape(b,-1))
            arrays['local_moments'].append(torch.view_as_real(z[...,mask]).numpy().reshape(b,-1))
            zg=patch_pac_vector(phase,amp,1,f.normalize)
            arrays['global_moments'].append(torch.view_as_real(zg[...,mask]).numpy().reshape(b,-1))
    for name,parts in arrays.items():
        features=np.concatenate(parts);np.save(args.output/(name+'.npy'),features,allow_pickle=False)
        records=[];predictions=[]
        for degree in (1,2):
            r,pred=probe(features,data,degree);records.append(r);predictions.append(pred)
        chosen=min(records,key=lambda v:v['validation_mse'])
        # Radius readout is a separate diagnostic, not a replacement curve score.
        strength=dict(data);strength['curves']=np.linalg.norm(data['coefficients'],axis=1,keepdims=True)
        sr=[]
        for degree in (1,2):sr.append(probe(features,strength,degree)[0])
        report['rows'][name]=dict(curve_choices=records,selected_degree=chosen['degree'],strength_choices=sr)
        dump(args.output/(name+'.json'),report['rows'][name]);np.savez_compressed(args.output/(name+'-predictions.npz'),predictions=predictions)
        print(name,'curve',chosen['overall'],'strength',min(sr,key=lambda v:v['validation_mse'])['overall'],flush=True)
    report['seconds']=time.monotonic()-begin
    dump(args.output/'scores.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
