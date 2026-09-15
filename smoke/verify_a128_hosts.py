"""Host-specific real data, exact native tail, token grid, gradient and reload checks."""
import argparse,copy,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.models.build import build_model
from paclock_bench.training.train import set_seed
from paclock_bench.training.losses import build_loss
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text());host=cfg['model'].removesuffix('_a128')
run_id=os.environ.get('SLURM_JOB_ID') or os.environ.get('CFM_STANDALONE_RUN_ID')
assert torch.cuda.is_available() and run_id
assert os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') is not None
if cfg.get('loader')=='biot':
 from paclock_bench.data.biot_dataset import build_biot_dataloaders as loaders
else:
 from paclock_bench.data.datasets import build_dataloaders as loaders
tr,va,te,info=loaders(cfg)
native_cfg=copy.deepcopy(cfg);native_cfg['model']=host
set_seed(cfg['seed']);native=build_model(native_cfg,info['input_shape'])
set_seed(cfg['seed']);model=build_model(cfg,info['input_shape'])
# The complete native tail must retain all original weights, including task head.
a=native.state_dict();b=model.state_dict();compared=[]
for k,v in a.items():
 if host=='cbramod':
  if k.startswith('backbone.patch_embedding.') and 'positional_encoding' not in k:continue
  dest=k
 else:
  if k.startswith('biot.patch_embedding.'):continue
  dest='biot.native.'+k[len('biot.'):] if k.startswith('biot.') else k
 assert dest in b and torch.equal(v,b[dest]),(k,dest)
 compared.append(k)
model=model.cuda();native=native.cuda().eval()
x,y=next(iter(tr))
if cfg['num_classes']==2 and y.unique().numel()<2:
 import numpy as np
 labels=np.asarray(tr.dataset.labels).reshape(-1)
 missing=1-int(y[0]); idx=int(np.flatnonzero(labels==missing)[0])
 x[0],y[0]=tr.dataset[idx]
x,y=x.cuda(),y.cuda().long()
print('smoke class counts',torch.bincount(y,minlength=cfg['num_classes']).tolist(),flush=True)
q=model.backbone.patch_embedding.quad if host=='cbramod' else model.biot.quad
hop=200 if host=='cbramod' else native.biot.hop_length
width=200 if host=='cbramod' else 256
with torch.no_grad():
 q.eval();t=q(x[:2]);assert t.shape==(min(2,x.shape[0]),x.shape[1],(x.shape[-1]-200)//hop+1,width)
 if host=='cbramod':
  n=native.backbone.patch_embedding(x[:2].reshape(2,x.shape[1],-1,200));assert n.shape==t.shape
 else:
  n=native.biot.patch_embedding(native.biot.stft(x[:2,:1]));assert n.shape==t[:,0].shape
# Verify real upstream tail behaves identically when fed the SAME synthetic tokens.
with torch.no_grad():
 model.eval()
 if host=='cbramod':
  z=torch.randn_like(t);u=native.classifier(native.backbone.encoder(z));v=model.classifier(model.backbone.encoder(z))
 else:
  z=torch.randn(2,t.shape[1]*t.shape[2],256,device='cuda')
  u=native.classifier(native.biot.transformer(z).mean(1));v=model.classifier(model.biot.native.transformer(z).mean(1))
 assert torch.allclose(u,v,rtol=1e-5,atol=1e-6)
del native
opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
loss_fn=build_loss(cfg);times=[];model.train()
for i in range(5):
 torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
 pred=model(x);assert pred.shape==(x.shape[0],cfg['num_classes'])
 loss=loss_fn(pred,y);assert torch.isfinite(loss);loss.backward()
 grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
 assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
 if i==0:assert any(g is not None and g.abs().sum()>0 for n,g in grads.items() if 'encoder' in n or 'transformer' in n)
 assert q.projection[0].weight.grad is not None
 if i==0:assert q.frontend.fusion_beta.grad is not None and q.frontend.fusion_beta.grad.abs().sum()>0
 if cfg.get('grad_clip'):torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
 opt.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
model.eval()
with torch.no_grad(): before=model(x[:2])
ck=Path('results')/f'host-roundtrip-{run_id}-{cfg["name"]}.pt';ck.parent.mkdir(exist_ok=True)
torch.save(model.state_dict(),ck);model.load_state_dict(torch.load(ck,weights_only=True),strict=True)
with torch.no_grad(): assert torch.equal(before,model(x[:2]))
ck.unlink()
result=dict(ok=True,host=host,config=cfg,source_tail_tensors=len(compared),shape=list(t.shape),steady_step_seconds=sum(times[2:])/3,
 projected_full_train_hours=sum(times[2:])/3*len(tr)*cfg['epochs']/3600,params=sum(p.numel() for p in model.parameters()),peak_gib=torch.cuda.max_memory_allocated()/2**30,
 note='Projection excludes validation and may overestimate patience-limited training')
p=Path('results')/f'a128-host-smoke-{run_id}-{cfg["name"]}.json';p.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if not cfg.get("patience") and cfg.get("max_hours"):
 assert result["projected_full_train_hours"] < cfg["max_hours"] * .9, "No-patience host schedule does not fit this runtime budget"
