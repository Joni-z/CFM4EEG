"""Read-only full-test diagnostic of a fixed validation-selected checkpoint.

No retraining, checkpoint selection, or changes to the production model files.
All interventions use the same windows and deterministic batch permutation.
"""
import argparse, csv, hashlib, json, os, time, types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix
from paclock_bench.models.build import build_model
from paclock_bench.data.datasets import WindowDataset
from paclock_bench.training.train import set_seed, amp_context
from paclock_bench.training.metrics import compute_metrics

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def bounds(f):
    lo=f.min_low_hz+f.low_hz_.detach().abs().flatten()
    hi=(lo+f.min_band_hz+f.band_hz_.detach().abs().flatten()).clamp(f.min_low_hz,f.sample_rate/2)
    return torch.stack([lo,hi],1).cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists(), 'Do not overwrite a completed analysis'
    run=Path(args.run); ck=run/'best.pt'; result=json.loads((run/'result.json').read_text())
    state=torch.load(ck,map_location='cpu',weights_only=False)
    cfg=state['config'];set_seed(int(cfg['seed']));torch.set_num_threads(1)
    ds=WindowDataset(args.data,'test',preload=True)
    model=build_model(cfg,ds.shape)
    fe=next(m for m in model.modules() if hasattr(m,'_pac_interaction'))
    initial=bounds(fe.sinc);model.load_state_dict(state['model'],strict=True)
    trained=bounds(fe.sinc);model=model.cuda().eval();nb=fe.n_bands
    assert nb==16 and fe.pac_token_mode=='measured'
    assert fe.waveform_modulation=='quadrature'
    nclasses=int(cfg['num_classes']);batch_size=32
    # An analysis-only permutation avoids contiguous class/record order as the
    # definition of cross-window pairing. No labels enter this permutation.
    perm=np.random.default_rng(20260916).permutation(len(ds))
    if args.smoke: perm=perm[:64]
    assert len(perm)%batch_size!=1,'A singleton cannot be cyclically shuffled'
    loader=DataLoader(Subset(ds,perm),batch_size=batch_size,num_workers=4,
                      pin_memory=True,shuffle=False,drop_last=False)
    original=fe._pac_interaction
    current={'mode':'measured','collect':False,'labels':None}
    acc={'alpha':np.zeros((nb,nb)), 'alpha_interior':np.zeros((nb,nb)),
         'class_alpha':np.zeros((nclasses,nb,nb)), 'class_count':np.zeros(nclasses),
         'a2':np.zeros(nb),'w2':np.zeros(nb),'n':0,'fallback':0,'columns':0}
    def hooked(self,phase_feat,amplitude_feat,pac_vector,waveform_feat=None):
        mode=current['mode']
        if current['collect']:
            edge=pac_vector.transpose(-2,-1)
            valid=torch.tril(torch.ones(nb,nb,device=edge.device,dtype=torch.bool),-1)
            mag=edge.abs()*valid
            denom=mag.sum(-1,keepdim=True)
            fallback=valid/valid.sum(-1,keepdim=True).clamp_min(1)
            alpha=torch.where(denom>1e-8,mag/denom.clamp_min(1e-8),fallback)
            # Mean per window first; every window has equal weight.
            aw=alpha.mean((1,2)).double().cpu().numpy()
            ai=alpha[:,:,2:-2].mean((1,2)).double().cpu().numpy()
            acc['alpha']+=aw.sum(0);acc['alpha_interior']+=ai.sum(0)
            for c in range(nclasses):
                mask=current['labels']==c
                acc['class_alpha'][c]+=aw[mask].sum(0)
                acc['class_count'][c]+=int(mask.sum())
            acc['a2']+=amplitude_feat.square().mean((1,2,4)).double().cpu().numpy().sum(0)
            acc['w2']+=waveform_feat.square().mean((1,2,4)).double().cpu().numpy().sum(0)
            acc['n']+=len(aw)
            acc['fallback']+=int((denom[:,:,:,1:]<=1e-8).sum().item())
            acc['columns']+=denom[:,:,:,1:].numel()
        self.pac_token_mode={'own_phase':'own','weights_only':'magnitude','identity':'identity'}.get(mode,'measured')
        if mode=='shuffled_Z':
            pac_vector=pac_vector.roll(1,0)
        if mode=='envelope_only': waveform_feat=waveform_feat*0
        return original(phase_feat,amplitude_feat,pac_vector,waveform_feat=waveform_feat)
    # Prove interception is identity before interpreting any intervention.
    X,_=next(iter(loader));X=X.cuda()
    with torch.inference_mode(),amp_context(cfg,'cuda'):
        ref=model(X)
        fe._pac_interaction=types.MethodType(hooked,fe)
        observed=model(X)
        torch.testing.assert_close(ref,observed,rtol=0,atol=0)
        current['mode']='weights_only';other=model(X)
        assert torch.isfinite(other).all() and not torch.equal(ref,other)
    print('STRICT RELOAD + HOOK IDENTITY + INTERVENTION SMOKE PASS',flush=True)
    np.savez(out/'bands.npz',initial=initial,trained=trained)
    output={};all_logits={};labels=None
    modes=['measured','identity','own_phase','weights_only','shuffled_Z','envelope_only']
    for mode in modes:
        current['mode']=mode;current['collect']=mode=='measured'
        logits=[];ys=[];t0=time.time()
        with torch.inference_mode():
            for bi,(X,y) in enumerate(loader):
                current['labels']=y.numpy();ys.append(y.numpy())
                with amp_context(cfg,'cuda'): pred=model(X.cuda(non_blocking=True))
                assert torch.isfinite(pred).all()
                logits.append(pred.float().cpu().numpy())
                if bi%200==0:print(mode,bi,'/',len(loader),'sec',round(time.time()-t0,1),flush=True)
        ll=np.concatenate(logits);yy=np.concatenate(ys)
        if labels is not None: assert np.array_equal(labels,yy)
        labels=yy;all_logits[mode]=ll
        metrics=compute_metrics(yy,ll,nclasses)
        output[mode]={'metrics':metrics,'class_f1':f1_score(yy,ll.argmax(-1),labels=range(nclasses),average=None,zero_division=0).tolist(),
                      'confusion':confusion_matrix(yy,ll.argmax(-1),labels=range(nclasses)).tolist(),'seconds':time.time()-t0}
        if mode=='measured' and not args.smoke:
            for k,v in result['test'].items():
                assert abs(metrics[k]-v)<1e-6, ('BASELINE MISMATCH',k,metrics[k],v)
        (out/'progress.json').write_text(json.dumps(output,indent=2))
        print(mode,metrics,flush=True)
    fe._pac_interaction=original;fe.pac_token_mode='measured'
    alpha=acc['alpha']/acc['n'];alpha_i=acc['alpha_interior']/acc['n']
    ca=acc['class_alpha']/acc['class_count'][:,None,None]
    wr=np.sqrt(acc['w2']/np.maximum(acc['a2'],1e-30))
    np.testing.assert_allclose(alpha[1:].sum(-1),1,atol=1e-6)
    np.savez_compressed(out/'diagnostics.npz',alpha=alpha,alpha_interior=alpha_i,
                        class_alpha=ca,counts=acc['class_count'],waveform_ratio=wr,
                        initial=initial,trained=trained,labels=labels,indices=perm,**all_logits)
    with (out/'bands.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['band','initial_low_hz','initial_high_hz','learned_low_hz','learned_high_hz','rms_w_over_a'])
        for i in range(nb): w.writerow([i+1,*initial[i],*trained[i],wr[i]])
    report={'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'result_sha256':sha(run/'result.json'),
            'analysis_sha256':sha(__file__),'manifest_sha256':sha(Path(args.data)/'manifest.json'),
            'dataset':cfg['dataset'],'seed':cfg['seed'],'n_test':len(labels),'class_counts':acc['class_count'].tolist(),
            'parameters':sum(p.numel() for p in model.parameters()),'parameter_groups':{name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()},'config':cfg,'interventions':output,
            'ascending_centers':bool(np.all(np.diff(trained.mean(-1))>0)),
            'max_center_shift_hz':float(abs(trained.mean(-1)-initial.mean(-1)).max()),
            'fallback_fraction':acc['fallback']/acc['columns'],
            'alpha_max_edge_exclusion_change':float(abs(alpha_i-alpha).max()),
            'waveform_ratio_range':[float(wr.min()),float(wr.max())],
            'sampling':'Full test; window permutation seed 20260916; batch 32; Z rolled one window within each batch; not subject-level shuffle',
            'interpretation':'Single fixed checkpoint, test-time interventions, not retrained ablations. No significance test.',
            'smoke_only':args.smoke}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
