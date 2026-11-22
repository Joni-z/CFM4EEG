"""Contract, invariance, gradient, data, checkpoint and steady-state GPU smoke."""
from pathlib import Path
import argparse, copy, hashlib, json, os, statistics, time
import numpy as np
import torch
import yaml
from paclock_bench.models.paclock.frontend.geometric_fusion import GeometricContentFusion
from paclock_bench.models.build import build_model
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.training.train import set_seed
from paclock_bench.training.losses import build_loss

ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
report={'checks': [], 'torch':torch.__version__, 'device':torch.cuda.get_device_name(0)}
def passed(s):
 report['checks'].append(s);print('PASS',s,flush=True)

set_seed(123)
shape=(2,2,3,4,8)
a=torch.randn(shape,requires_grad=True);w=torch.randn(shape,requires_grad=True)
p=torch.complex(torch.randn(shape),torch.randn(shape)).requires_grad_()
z=torch.complex(torch.randn(2,2,3,4,4),torch.randn(2,2,3,4,4)).requires_grad_()
modules=[GeometricContentFusion(16,4,mode=m) for m in ['aligned','no_alignment','ordinary']]
assert len({sum(t.numel() for t in m.parameters()) for m in modules})==1
passed('all future controls have identical parameter counts')
f=modules[0];y=f(p,a,w,z)
assert torch.equal(y[...,0,:],torch.zeros_like(y[...,0,:]))
assert torch.equal(f(p,a,w,torch.zeros_like(z)),torch.zeros_like(y))
passed('root band and absent relations yield exactly zero residual')
delta=torch.randn(2,2,3,4);shift=torch.polar(torch.ones_like(delta),delta)
yshift=f(p*shift.unsqueeze(-1),a,w,z*shift.unsqueeze(-1))
torch.testing.assert_close(y,yshift,atol=2e-8,rtol=2e-5)
passed('source analytic-reference invariance with content fixed')
# Explicit loop verifies source/target orientation and the conjugate sign.
m=f.aggregate(p,a,w,z)
gate=f.content_gate(torch.cat((a,w),-1));value=gate*torch.complex(a,w)*(p/p.abs().clamp_min(1e-6))
manual=torch.zeros_like(m)
for j in range(1,4):
 edges=z[...,:j,j];alpha=edges.abs()/edges.abs().sum(-1,keepdim=True)
 coef=alpha*(edges/edges.abs()).conj()
 manual[...,j,:]=(coef.unsqueeze(-1)*value[...,:j,:]).sum(-2)
torch.testing.assert_close(m,manual,atol=1e-6,rtol=1e-5)
passed('direct loop matches directed transport and conjugate alignment')
changed=a.detach().clone();changed[...,0,:]+=1
assert (f(p,changed,w,z)[...,-1,:]-y[...,-1,:]).abs().max()>1e-6
passed('source content changes target token with phase and relations fixed')
y.abs().square().sum().backward()
for name,param in f.named_parameters():
 assert param.grad is not None and torch.isfinite(param.grad).all(),name
 assert param.grad.abs().sum()>0,name
assert all(torch.isfinite(t.grad).all() for t in [a,w,p,z])
passed('all new parameter groups receive finite nonzero gradients')
for v in [0.,1e-12]:
 pp=torch.full(shape,complex(v,v),dtype=torch.complex64,requires_grad=True)
 zz=torch.full((2,2,3,4,4),complex(v,v),dtype=torch.complex64,requires_grad=True)
 yy=f(pp,torch.zeros(shape),torch.zeros(shape),zz);yy.real.sum().backward()
 assert torch.isfinite(yy).all() and torch.isfinite(pp.grad).all() and torch.isfinite(zz.grad).all()
passed('flat and near-zero inputs have finite output and backward')

root=Path(os.path.expandvars(cfg['data_root']))
for split in ['train','val','test']:
 for kind in ['signals','labels']:
  path=root/f'{split}_{kind}.npy';v=np.load(path,mmap_mode='r')
  assert path.stat().st_size>=v.offset+v.nbytes,(path,'truncated')
 if split=='train':input_shape=tuple(np.load(root/'train_signals.npy',mmap_mode='r').shape[1:])
passed('all three dataset splits exist with complete array payloads')
basecfg=copy.deepcopy(cfg);basecfg['model_kwargs'].pop('geometric_fusion')
set_seed(cfg['seed']);base=build_model(basecfg,input_shape)
set_seed(cfg['seed']);model=build_model(cfg,input_shape)
for name,v in base.state_dict().items(): assert torch.equal(v,model.state_dict()[name]),name
report['base_parameters']=sum(t.numel() for t in base.parameters())
report['new_parameters']=sum(t.numel() for t in model.parameters())
passed('all shared parameters initialize bit-identically to original model')
base=base.cuda().eval();model=model.cuda().eval()
x=torch.randn(2,*input_shape,device='cuda')
with torch.no_grad():
 saved=model.frontend.geometric_content_fusion.scale_logit.clone()
 model.frontend.geometric_content_fusion.scale_logit.zero_()
 torch.testing.assert_close(model(x),base(x),atol=0,rtol=0)
 model.frontend.geometric_content_fusion.scale_logit.copy_(saved)
passed('zero residual recovers original full-model logits exactly')
del base;torch.cuda.empty_cache()
train,val,test,info=build_dataloaders(cfg)
X,Y=next(iter(train));X=X.cuda();Y=Y.cuda().long()
assert tuple(X.shape[1:])==input_shape
report['data']={'input_shape':input_shape,'n_samples':info['n_samples'],'class_counts':info.get('class_counts'),'manifest_created':info.get('manifest',{}).get('created_utc')}
criterion=build_loss(cfg);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
model.train();times=[];torch.cuda.reset_peak_memory_stats()
for step in range(8):
 torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
 logits=model(X);loss=criterion(logits,Y);assert torch.isfinite(loss)
 loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);opt.step()
 torch.cuda.synchronize()
 if step>=3:times.append(time.perf_counter()-t)
passed('real TUEV batch, eight optimizer steps, warm-up excluded from timing')
report['batch_size']=len(X);report['step_seconds_median']=statistics.median(times)
report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
report['steps_per_epoch']=len(train)
model.eval()
with torch.no_grad(): expected=model(X[:2]).cpu()
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
ckpt=out.parent/'smoke-reload.pt';torch.save({'config':cfg,'model':model.state_dict()},ckpt)
reload=build_model(cfg,input_shape).cuda().eval();reload.load_state_dict(torch.load(ckpt,weights_only=False)['model'],strict=True)
with torch.no_grad():torch.testing.assert_close(expected,reload(X[:2]).cpu(),atol=0,rtol=0)
passed('strict checkpoint reload preserves predictions')
report['status']='passed';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
