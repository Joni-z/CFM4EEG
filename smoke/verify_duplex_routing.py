"""Allocated GPU checks for shared initialization, source isolation and training."""
import copy,json,os,sys,gc,subprocess,shutil
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.duplex_routing import encode_sources
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
records=[]
for ds,shape in [('tuev',(16,1000)),('chbmit',(16,2000))]:
    cfg=yaml.safe_load(Path(f'configs/duplex_routing/{ds}_routed_s0.yaml').read_text())
    control=copy.deepcopy(cfg); control['model_kwargs']['duplex_routing']=False
    torch.manual_seed(0); baseline=build_model(control,shape).cuda().eval()
    torch.manual_seed(0); model=build_model(cfg,shape).cuda().eval()
    a,b=baseline.state_dict(),model.state_dict()
    shared=[k for k in a if not k.startswith('head.')]
    assert all(k in b and torch.equal(a[k],b[k]) for k in shared)
    x=torch.randn(2,*shape,device='cuda')
    with torch.no_grad():
        tokens,cpl,hz=model.frontend(x)
        n=tokens.shape[2];c=tokens.shape[1];d=tokens.shape[-1]
        tokens=tokens+model.band_pe(hz).view(1,1,n,1,d)+model.spatial_pe(c,x.device).view(1,c,1,1,d)
        w,h=encode_sources(model.encoder,tokens,cpl)
        mutated=tokens.clone();mutated[:,:,:n//2]+=torch.randn_like(mutated[:,:,:n//2])
        w2,h2=encode_sources(model.encoder,mutated,cpl)
        torch.testing.assert_close(h,h2,rtol=0,atol=0)
        assert not torch.allclose(w,w2)
        old_w=w.clone();old_h=h.clone();out=model.head(w,h)
        assert torch.equal(w,old_w) and torch.equal(h,old_h)
        assert out.shape==(2,cfg['num_classes']) and torch.isfinite(out).all()
        assert torch.isfinite(model(torch.zeros_like(x))).all()
    model.train(); logits=model(x); loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device='cuda'))
    loss.backward()
    for prefix in ['frontend.tokenizer','frontend.interaction_gate','encoder.','head.cross.','head.proj.']:
        grads=[p.grad for k,p in model.named_parameters() if k.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0,prefix
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    model.eval();clone=build_model(cfg,shape).cuda().eval();clone.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(model(x),clone(x),rtol=1e-5,atol=1e-6)
    records.append(dict(dataset=ds,shared_initial_state_exact=True,encoded_coupling_isolated=True,
                        original_streams_not_overwritten=True,gradients_finite_nonzero=True,checkpoint_roundtrip=True,
                        parameters=sum(p.numel() for p in model.parameters()),baseline_parameters=sum(p.numel() for p in baseline.parameters())))
    del baseline,model,clone,a,b,x,tokens,cpl,hz,w,h,w2,h2,mutated,old_w,old_h,out,logits,loss,grads
    gc.collect();torch.cuda.empty_cache()
p=Path('results')/f'duplex-routing-contract-{os.environ["SLURM_JOB_ID"]}.json';p.write_text(json.dumps(records,indent=2)+'\n');print(p.read_text(),flush=True)
for ds in ['tuev','chbmit']:
    for variant in ['control','routed']:
        cfg=f'configs/duplex_routing/{ds}_{variant}_s0.yaml'
        subprocess.run([sys.executable,'-u','smoke/smoke_amd_partition.py','--config',cfg],check=True)
        src=Path('results')/f'amd-partition-smoke-{os.environ["SLURM_JOB_ID"]}.json'
        shutil.copy2(src,Path('results')/f'duplex-routing-timing-{ds}-{variant}-{os.environ["SLURM_JOB_ID"]}.json')
