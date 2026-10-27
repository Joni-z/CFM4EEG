"""Paired actual CfFT tokenizer/encoder learning on spectrum-matched signals."""
import argparse,copy,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,roc_auc_score
from paclock_bench.models.build import build_model

def cross_pair_dataset(seed,pairs,frange):
    """Each LF and HF waveform appears once in each class, with equal spectra.

    Quadruplets are (L,H), (shift L,shift H), (L,shift H), (shift L,H).
    Low/high signals occupy disjoint Fourier bins. Positive pairs retain their
    joint timing; negative pairs change timing between nonstationary bands.
    A shifted stationary sinusoid would not be an adequate negative control.
    """
    rng=np.random.default_rng(seed);t=np.arange(1000)/200
    frequency=np.fft.rfftfreq(1000,1/200);data=[];labels=[];maxerr=0.
    for group in range(pairs//2):
        f=rng.uniform(*frange);rate=rng.uniform(.55,.85)
        phi=2*np.pi*f*t+rng.uniform(-np.pi,np.pi)+np.sin(2*np.pi*rate*t+rng.uniform(-np.pi,np.pi))/rate
        low=.8*np.cos(phi)+.02*rng.normal(size=len(t))
        high=(1+rng.uniform(.5,.8)*np.cos(phi-rng.uniform(-np.pi,np.pi)))*np.cos(2*np.pi*rng.uniform(28,32)*t+rng.uniform(-np.pi,np.pi))
        high+=.02*rng.normal(size=len(t))
        low=np.fft.irfft(np.fft.rfft(low)*(frequency<=12),n=len(t))
        high=np.fft.irfft(np.fft.rfft(high)*(frequency>=16),n=len(t))
        lag=round(.5/rate*200);origin=int(rng.integers(len(t)))
        shifted_low,shifted_high=np.roll(low,lag),np.roll(high,lag)
        group_x=np.array([low+high,shifted_low+shifted_high,low+shifted_high,shifted_low+high])
        group_x=np.roll(group_x,origin,axis=-1)
        spectrum=abs(np.fft.rfft(group_x,axis=-1))
        maxerr=max(maxerr,float(abs(spectrum-spectrum[:1]).max()))
        data.extend(group_x*20);labels.extend([1,1,0,0])
    return torch.tensor(np.array(data),dtype=torch.float32)[:,None],torch.tensor(labels),maxerr

def dataset(seed,pairs,frange,task='phase_scramble'):
    if task=='cross_pair':return cross_pair_dataset(seed,pairs,frange)
    rng=np.random.default_rng(seed);t=np.arange(1000)/200
    data=[];labels=[];maxerr=0.
    for k in range(pairs):
        f=rng.uniform(*frange);carrier=rng.uniform(28,32)
        phi=2*np.pi*f*t+rng.uniform(-np.pi,np.pi)
        rate=rng.uniform(.2,.6)
        phi=phi+.65/rate*np.sin(2*np.pi*rate*t+rng.uniform(-np.pi,np.pi))
        envelope=1+rng.uniform(.5,.8)*np.cos(phi-rng.uniform(-np.pi,np.pi))
        raw=.8*np.cos(phi)+envelope*np.cos(2*np.pi*carrier*t+rng.uniform(-np.pi,np.pi))
        raw+=.04*rng.normal(size=len(t))
        spectrum=np.fft.rfft(raw)
        phases=rng.uniform(-np.pi,np.pi,len(spectrum));phases[0]=np.angle(spectrum[0]);phases[-1]=np.angle(spectrum[-1])
        null=np.fft.irfft(abs(spectrum)*np.exp(1j*phases),n=len(t))
        maxerr=max(maxerr,float(abs(abs(np.fft.rfft(null))-abs(spectrum)).max()))
        data.extend([raw*20,null*20]);labels.extend([1,0])
    return torch.tensor(np.array(data),dtype=torch.float32)[:,None],torch.tensor(labels),maxerr

@torch.no_grad()
def evaluate(model,x,y):
    model.eval();logits=[]
    for chunk in x.split(32):logits.append(model(chunk))
    logits=torch.cat(logits)
    return float(torch.nn.functional.cross_entropy(logits,y)),logits



def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['phase_scramble','cross_pair'],default='phase_scramble')
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--seed-offset',type=int,default=0);args=parser.parse_args()
    root=args.output
    root.mkdir(parents=True,exist_ok=False)

    seeds=(32001,42001,52001) if args.task=='phase_scramble' else (62001,72001,82001)
    seeds=tuple(seed+args.seed_offset for seed in seeds)
    ranges=[(4.3,5.0),(4.8,5.3),(5.2,5.7)] if args.task=='phase_scramble' else [(4.3,5.0)]*3
    train_x,train_y,_=dataset(seeds[0],128,ranges[0],args.task)
    val_x,val_y,_=dataset(seeds[1],48,ranges[1],args.task)
    test_x,test_y,spectral_error=dataset(seeds[2],96,ranges[2],args.task)
    config=dict(model='paclock',dataset='synthetic',sample_rate=200,num_classes=2,
        model_kwargs=dict(arch='triaxial',n_bands=8,d_model=64,depth=1,n_heads=4,
            dropout=.1,kernel_size=201,patch_len=50,filter_range=[.5,75.],
            tokenizer_mode='measured_pac',freq_mixer='none',band_pe='index',
            spatial_pe='index',space_over_bands=True,augmentations=[]))

    report={'config':config,'task':args.task,'steps_per_arm':args.epochs*(len(train_x)//32),'generator_seeds':seeds,
            'source_frequency_ranges':ranges,'test_fourier_magnitude_max_error':spectral_error,'arms':{}}
    for arm in ['identity','measured']:
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        cfg=copy.deepcopy(config);cfg['model_kwargs']['measurement_mode']=arm
        model=build_model(cfg,(1,1000))
        state_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for k,v in model.state_dict().items() if k.startswith('encoder.') or k.startswith('head.'))).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        history=[];best=float('inf');best_state=None;start=time.monotonic();step=0
        for epoch in range(args.epochs):
            model.train()
            order=torch.randperm(len(train_x))
            for indices in order.split(32):
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(train_x[indices]),train_y[indices])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
                assert torch.isfinite(loss)
            val_loss,logits=evaluate(model,val_x,val_y)
            history.append({'epoch':epoch,'validation_loss':val_loss,
                'validation_accuracy':accuracy_score(val_y,logits.argmax(-1)),'training_loss':float(loss.detach())})
            if val_loss<best:
                best=val_loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            print(arm,history[-1],flush=True)
        model.load_state_dict(best_state)
        test_loss,logits=evaluate(model,test_x,test_y)
        probability=logits.softmax(-1)[:,1].numpy()
        torch.save(dict(state_dict=best_state,config=cfg,best_epoch=best_epoch),root/(arm+'-best.pt'))
        np.savez_compressed(root/(arm+'-predictions.npz'),labels=test_y.numpy(),logits=logits.numpy(),probability=probability)
        report['arms'][arm]=dict(seconds=time.monotonic()-start,best_epoch=best_epoch,
            validation_loss=best,test_loss=test_loss,test_accuracy=accuracy_score(test_y,logits.argmax(-1)),
            test_auroc=roc_auc_score(test_y,probability),initial_encoder_head_sha256=state_hash,history=history)
        (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{m:v for m,v in a.items() if m!='history'} for k,a in report['arms'].items()},indent=2),flush=True)


if __name__ == "__main__":
    main()
