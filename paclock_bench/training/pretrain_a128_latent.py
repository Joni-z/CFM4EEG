"""One Duplex-A128 masked latent-prediction pretrain; raw masking before all filters.
EMA targets, fixed numbered snapshots, atomic optimizer resume, bounded runtime.
No labels/validation/test data participate in this objective.
"""
import argparse,copy,json,math,os,random,signal,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from ..models.build import build_model
from ..paths import expand
from .train import set_seed

class A128Latent(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        model_cfg=dict(cfg['model_config'])
        self.student=build_model(model_cfg,(cfg['channels'],cfg['crop_samples']))
        self.teacher=copy.deepcopy(self.student).requires_grad_(False).eval()
        for p in self.student.head.parameters(): p.requires_grad_(False)
        for p in self.student.spatial_pe.parameters(): p.requires_grad_(False)
        d=model_cfg['model_kwargs']['d_model']
        self.mask_token=nn.Parameter(torch.zeros(d))
        self.predictor=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
        self.patch=50
        self.mask_ratio=cfg.get('mask_ratio',.5)

    def mask(self,x):
        b,c,t=x.shape;p=t//self.patch
        # Contiguous spans, same across electrodes; prevents an unmasked
        # electrode handing a masked electrode an identical time interval.
        span=max(1,p//4); mask=torch.zeros(b,p,device=x.device,dtype=torch.bool)
        for i in range(b):
            while mask[i].sum()<int(p*self.mask_ratio):
                start=torch.randint(p-span+1,(),device=x.device)
                mask[i,start:start+span]=True
        return mask

    def encode(self,model,x,mask=None):
        if mask is not None:
            # All nonlocal Sinc/Hilbert/PAC operations see only masked raw data.
            rawmask=mask.repeat_interleave(self.patch,1)[:,None,:]
            x=x.masked_fill(rawmask,0)
        t,coupling,hz=model.frontend(x)
        b,c,n,p,d=t.shape
        if mask is not None:
            t=torch.where(mask[:,None,None,:,None],self.mask_token.view(1,1,1,1,d),t)
        t=t+model.band_pe(hz).view(1,1,n,1,d)
        # No arbitrary channel-index identity in pretraining. Downstream
        # electrode positional embeddings are deliberately initialized fresh.
        return model.encoder(t,coupling,None)

    def forward(self,x,mask=None):
        mask=self.mask(x) if mask is None else mask
        self.teacher.eval()
        with torch.no_grad():
            target=self.encode(self.teacher,x).float()
            # Remove the batch-shared positional/DC component before feature
            # normalization: predicting only a band identity must not solve SSL.
            target=F.layer_norm(target-target.mean(0,keepdim=True),(128,))
        h=self.encode(self.student,x,mask)
        pred=F.layer_norm(self.predictor(h).float(),(128,))
        weights=mask[:,None,None,:].expand(h.shape[:-1])
        prediction=F.smooth_l1_loss(pred[weights],target[weights])
        # Sample-level variance, not variation of positional embeddings.
        pooled=F.layer_norm(h.float().mean((1,2,3)),(128,))
        std=pooled.std(dim=0,unbiased=False)
        varloss=F.relu(.5-std).mean()
        loss=prediction+varloss
        stats={'loss':float(loss.detach()),'prediction':float(prediction.detach()),
               'student_sample_std':float(std.detach().mean()),
               'teacher_sample_std':float(target.mean((1,2,3)).std(0,unbiased=False).mean()),
               'mask_fraction':float(mask.float().mean())}
        return loss,stats

    @torch.no_grad()
    def update_teacher(self,momentum):
        for t,s in zip(self.teacher.parameters(),self.student.parameters()): t.lerp_(s,1-momentum)
        for t,s in zip(self.teacher.buffers(),self.student.buffers()): t.copy_(s)

class CroppedEEG(torch.utils.data.Dataset):
    def __init__(self,root,channels,crop):
        self.path=str(Path(root)/'train_signals.npy');self.data=None
        a=np.load(self.path,mmap_mode='r');self.n,self.c,self.t=a.shape
        assert self.c>=channels and self.t>=crop
        self.channels,self.crop=channels,crop
    def __len__(self): return self.n
    def __getitem__(self,i):
        if self.data is None:self.data=np.load(self.path,mmap_mode='r')
        inds=np.random.choice(self.c,self.channels,replace=False)
        start=np.random.randint(self.t-self.crop+1)
        return torch.from_numpy(np.array(self.data[i,inds,start:start+self.crop],copy=True))

def save_atomic(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());set_seed(cfg['seed'])
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() if Path('.git').exists() else 'standalone-smoke'
    manifest=Path(expand(cfg['data_root']))/'manifest.json'
    manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert torch.cuda.is_available()
    out=Path('pretrain_runs')/cfg['name'];out.mkdir(parents=True,exist_ok=True)
    path=out/'checkpoint.pt'
    if path.exists() and not args.resume and not args.smoke: raise FileExistsError(path)
    net=A128Latent(cfg).cuda()
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=.05)
    start=0
    if path.exists() and args.resume:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['cfg']==cfg
        assert ck['source_sha256']==source_sha, 'Code changed across resume'
        net.load_state_dict(ck['full']);opt.load_state_dict(ck['opt']);start=ck['step']
        torch.set_rng_state(ck['rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    ds=CroppedEEG(expand(cfg['data_root']),cfg['channels'],cfg['crop_samples'])
    loader=torch.utils.data.DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    it=iter(loader);stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    steps=5 if args.smoke else cfg['steps'];t0=time.monotonic();times=[]
    for step in range(start+1,steps+1):
        try:x=next(it)
        except StopIteration:it=iter(loader);x=next(it)
        x=x.cuda(non_blocking=True)
        torch.cuda.synchronize();t=time.monotonic()
        warm=min(1.,step/cfg['warmup_steps']);progress=max(0,step-cfg['warmup_steps'])/max(1,cfg['steps']-cfg['warmup_steps'])
        for g in opt.param_groups:g['lr']=cfg['lr']*warm*(.1+.9*(1+math.cos(math.pi*progress))/2)
        opt.zero_grad(set_to_none=True);loss,stats=net(x)
        if not torch.isfinite(loss):raise FloatingPointError(stats)
        loss.backward();norm=nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradients')
        opt.step();net.update_teacher(.996+(.9998-.996)*step/cfg['steps'])
        torch.cuda.synchronize();times.append(time.monotonic()-t)
        if args.smoke or step%100==0:
            print(json.dumps(dict(step=step,**stats,step_seconds=float(np.mean(times[-98:])),peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        if args.smoke:continue
        stop_now=stop[0] or time.monotonic()-t0>cfg['max_hours']*3600
        if step%cfg['save_every']==0 or step==steps or stop_now:
            ck=dict(model=net.student.state_dict(),full=net.state_dict(),opt=opt.state_dict(),cfg=cfg,step=step,
                    source_sha256=source_sha,git_commit=revision,data_manifest_sha256=manifest_sha,
                    rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all(),
                    stopped_by='signal_or_budget' if stop_now else ('steps' if step==steps else 'snapshot'))
            save_atomic(path,ck)
            if step in cfg['snapshots'] or step==steps:save_atomic(out/f'checkpoint_{step}.pt',ck)
            (out/'progress.json').write_text(json.dumps(dict(step=step,**stats,stopped_by=ck['stopped_by'])))
        if stop_now:break
    if args.smoke:
        assert all(p.grad is None for p in net.teacher.parameters())
        assert net.student.frontend.fusion_beta.grad is not None
        net.eval();m=net.mask(x)
        other=x.clone();other.masked_fill_(m.repeat_interleave(50,1)[:,None,:],99.)
        with torch.no_grad():assert torch.equal(net.encode(net.student,x,m),net.encode(net.student,other,m))
        # Strict backbone transfer at the downstream native patch50 geometry.
        from ..models.paclock.build import _BACKBONE_PREFIXES
        target=build_model(cfg['model_config'],(16,1000))
        keys=[k for k in target.state_dict() if k.startswith(_BACKBONE_PREFIXES)]
        src=net.student.state_dict();assert all(k in src and src[k].shape==target.state_dict()[k].shape for k in keys)
        temp=out/'smoke-transfer.pt'
        save_atomic(temp,{'model':src})
        ftcfg=dict(cfg['model_config'],checkpoint=str(temp.resolve()),checkpoint_require_full=True)
        transferred=build_model(ftcfg,(16,1000))
        assert all(torch.equal(transferred.state_dict()[k],src[k].cpu()) for k in keys)
        broken=dict(src);broken.pop(keys[0]);save_atomic(temp,{'model':broken})
        try:build_model(ftcfg,(16,1000))
        except ValueError:pass
        else:raise AssertionError('Missing transfer tensor was silently accepted')
        temp.unlink()
        print(json.dumps({'smoke_ok':True,'mask_leak_check':True,'transfer_tensors':len(keys),'steady_step_seconds':float(np.mean(times[2:]))}),flush=True)

if __name__=='__main__':main()
