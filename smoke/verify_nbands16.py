"""Check the existing residual operator under the actual 16-band recipe."""
import copy,json,os,sys
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.models.build import build_model
from paclock_bench.training.train import set_seed
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
base=yaml.safe_load(Path('configs/design/tuev_crofremo_n5.yaml').read_text())
alt=copy.deepcopy(base);alt['model_kwargs']['local_residual']=True
set_seed(0);a=build_model(base,(16,1000)).cuda()
set_seed(0);b=build_model(alt,(16,1000)).cuda()
sa,sb=a.state_dict(),b.state_dict()
assert set(sb)-set(sa)=={'frontend.local_projection.weight'}
assert all(torch.equal(v,sb[k]) for k,v in sa.items())
x=torch.randn(2,16,1000,device='cuda')
a.eval();b.eval()
with torch.no_grad():
 assert torch.equal(a(x),b(x))
 assert torch.isfinite(b(torch.zeros_like(x))).all()
a.train();b.train()
set_seed(7);ya=a(x)
set_seed(7);yb=b(x)
assert torch.equal(ya,yb)
yb.square().mean().backward()
w=b.frontend.local_projection.weight
assert torch.isfinite(w.grad).all() and w.grad.abs().sum()>0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in b.parameters())
result={'ok':True,'n_bands':16,'shared_initial_state_exact':True,'initial_logits_exact':True,'augmented_train_logits_exact':True,'residual_gradient_nonzero':True}
Path('results').mkdir(exist_ok=True)
p=Path('results')/('nbands16-contract-'+os.environ['SLURM_JOB_ID']+'.json')
p.write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
