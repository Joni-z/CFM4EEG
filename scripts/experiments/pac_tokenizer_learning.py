"""Paired actual CfFT tokenizer/encoder learning on spectrum-matched signals."""
import argparse,copy,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,roc_auc_score
from paclock_bench.models.build import build_model

torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=args.output
root.mkdir(parents=True,exist_ok=False)

def dataset(seed,pairs,frange):
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

train_x,train_y,_=dataset(32001,128,(4.3,5.0))
val_x,val_y,_=dataset(42001,48,(4.8,5.3))
test_x,test_y,spectral_error=dataset(52001,96,(5.2,5.7))
config=dict(model='paclock',dataset='synthetic',sample_rate=200,num_classes=2,
    model_kwargs=dict(arch='triaxial',n_bands=8,d_model=64,depth=1,n_heads=4,
        dropout=.1,kernel_size=201,patch_len=50,filter_range=[.5,75.],
        tokenizer_mode='measured_pac',freq_mixer='none',band_pe='index',
        spatial_pe='index',space_over_bands=True,augmentations=[]))

@torch.no_grad()
def evaluate(model,x,y):
    model.eval();logits=[]
    for chunk in x.split(32):logits.append(model(chunk))
    logits=torch.cat(logits)
    return float(torch.nn.functional.cross_entropy(logits,y)),logits

report={'config':config,'steps_per_arm':160,'test_fourier_magnitude_max_error':spectral_error,'arms':{}}
for arm in ['identity','measured']:
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    cfg=copy.deepcopy(config);cfg['model_kwargs']['measurement_mode']=arm
    model=build_model(cfg,(1,1000))
    state_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for k,v in model.state_dict().items() if k.startswith('encoder.') or k.startswith('head.'))).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[];best=float('inf');best_state=None;start=time.monotonic();step=0
    for epoch in range(20):
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
