"""GPU-only A128 architecture, R1 identity, adapter ordering and latent masking."""
import os,sys,copy,json,tempfile
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.models.build import build_model
from paclock_bench.training.train import set_seed
from paclock_bench.training.pretrain_a128_latent import A128Latent
from paclock_bench.models.foundation.a128_adapter import A128HostTokens
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
c=yaml.safe_load(Path('configs/a128_r1/tuev_s0.yaml').read_text())
r=copy.deepcopy(c);r['model_kwargs']['augmentations']=[]
set_seed(0);a=build_model(c,(4,400)).cuda()
set_seed(0);b=build_model(r,(4,400)).cuda()
assert a.state_dict().keys()==b.state_dict().keys()
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
x=torch.randn(2,4,400,device='cuda');a.eval();b.eval()
with torch.no_grad():assert torch.equal(a(x),b(x))
assert a.frontend.n_output_bands==16 if hasattr(a.frontend,'n_output_bands') else True
a.train();loss=a(x).square().mean();loss.backward()
assert torch.isfinite(loss)
assert a.frontend.fusion_beta.grad is not None
assert a.frontend.interaction_gate.grad is not None
assert all(torch.isfinite(v.grad).all() for v in a.parameters() if v.grad is not None)
q=A128HostTokens(256,100).cuda().eval()
with torch.no_grad():
 t=q.frontend(x[:,:1])[0]
 assert t.shape==(2,1,16,8,128)
 assert torch.equal(q.frontend.fusion_beta,torch.zeros_like(q.frontend.fusion_beta))
 z=t[:,0].permute(0,2,1,3).flatten(2)
 manual=torch.stack([z[:,i:i+4].mean(1) for i in range(0,5,2)],dim=1)
 assert torch.allclose(q(x[:,:1])[:,0],q.projection(manual),rtol=1e-5,atol=1e-6)
pc=yaml.safe_load(Path('configs/pretrain/a128_latent_tueg.yaml').read_text())
pc.update(channels=4,crop_samples=400,batch_size=2)
net=A128Latent(pc).cuda();loss,stats=net(x);loss.backward()
assert torch.isfinite(loss) and all(p.grad is None for p in net.teacher.parameters())
assert net.student.frontend.fusion_beta.grad is not None
net.eval();m=net.mask(x);other=x.masked_fill(m.repeat_interleave(50,1)[:,None,:],99.)
with torch.no_grad():assert torch.equal(net.encode(net.student,x,m),net.encode(net.student,other,m))
from paclock_bench.models.paclock.build import _BACKBONE_PREFIXES
src=net.student.state_dict();path=Path('results/a128-contract-transfer.pt');path.parent.mkdir(exist_ok=True)
torch.save({'model':src},path)
ft=copy.deepcopy(c);ft.update(checkpoint=str(path.resolve()),checkpoint_require_full=True)
loaded=build_model(ft,(16,400));keys=[k for k in loaded.state_dict() if k.startswith(_BACKBONE_PREFIXES)]
assert all(torch.equal(loaded.state_dict()[k],src[k].cpu()) for k in keys)
path.unlink()
receipt=dict(ok=True,architecture_identity=True,eval_identity=True,duplex_rows=16,adapter_order=True,mask_leak_check=True,transfer_tensors=len(keys),latent_stats=stats)
Path('results/a128-contract-smoke.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
