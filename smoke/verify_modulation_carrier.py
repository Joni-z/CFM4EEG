"""GPU checks for waveform-preserving coordinates under PAC rotation."""
import copy,json,os,sys
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.models.build import build_model
from paclock_bench.training.train import set_seed
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
base=yaml.safe_load(Path('configs/design/tuev_crofremo_n5.yaml').read_text())
set_seed(0);baseline=build_model(base,(16,1000)).cuda()
records=[]
for arm in ['quadrature','carrier']:
 cfg=copy.deepcopy(base);cfg['model_kwargs']['waveform_modulation']=arm
 set_seed(0);model=build_model(cfg,(16,1000)).cuda()
 shared,changed=baseline.state_dict(),model.state_dict()
 assert set(changed)-set(shared)=={'frontend.waveform_quadrature.weight'}
 assert all(torch.equal(v,changed[k]) for k,v in shared.items())
 assert model.frontend.n_token_bands==16
 extra=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in baseline.parameters())
 assert extra==3200
 x=torch.randn(2,16,1000,device='cuda')
 model.eval();baseline.eval()
 with torch.no_grad():
  actual=model(x)
  assert actual.shape==(2,6) and torch.isfinite(actual).all()
  assert torch.isfinite(model(torch.zeros_like(x))).all()
  if arm=='quadrature':assert torch.equal(actual,baseline(x))
 if arm=='quadrature':
  model.train();baseline.train()
  set_seed(7);a=baseline(x)
  set_seed(7);b=model(x)
  assert torch.equal(a,b)
 else:
  model.train();b=model(x)
 torch.nn.functional.cross_entropy(b,torch.tensor([0,3],device='cuda')).backward()
 for name in ['phase_tokenizer','amplitude_tokenizer','waveform_quadrature']:
  g=getattr(model.frontend,name).weight.grad
  assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0,name
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 # Direct operator check separates conditional rotation from end-to-end
 # invertibility, which is NOT claimed. Cover ordinary and cancellation input.
 f=model.frontend
 phase=torch.randn(2,2,3,16,64,device='cuda',dtype=torch.complex64)
 pac=torch.randn(2,2,3,16,16,device='cuda',dtype=torch.complex64)
 amp=torch.randn(2,2,3,16,64,device='cuda')
 wave=torch.randn_like(amp)
 unit=f._pac_interaction(phase,torch.ones_like(amp),pac)
 out=f._pac_interaction(phase,amp,pac,waveform_feat=wave)
 expected=torch.complex(amp,wave)
 assert torch.allclose(out.abs().square(),expected.abs().square(),rtol=3e-5,atol=3e-5)
 assert torch.allclose(out*unit.conj(),expected,rtol=3e-5,atol=3e-5)
 left=f._pac_interaction(phase,amp,pac)
 right=f._pac_interaction(phase,torch.zeros_like(amp),pac,waveform_feat=wave)
 assert (left*right.conj()).real.abs().max()<3e-5
 zero=f._pac_interaction(torch.zeros_like(phase),amp,pac,waveform_feat=wave)
 assert torch.isfinite(zero).all() and torch.count_nonzero(zero)==0
 # A measured-alignment intervention must affect these same carriers.
 shifted=pac*torch.exp(1j*torch.randn_like(pac.real))
 assert not torch.allclose(out,f._pac_interaction(phase,amp,shifted,waveform_feat=wave))
 rank=None
 if arm=='carrier':
  matrix=torch.cat([f.amplitude_tokenizer.weight.reshape(64,50)*f.amplitude_scale[:,None],f.waveform_quadrature.weight.reshape(64,50)])
  rank=int(torch.linalg.matrix_rank(matrix).item());assert rank==50
 torch.optim.AdamW(model.parameters(),lr=1e-4).step()
 model.eval()
 checkpoint=Path('results')/f'carrier-roundtrip-{os.environ["SLURM_JOB_ID"]}-{arm}.pt'
 torch.save(model.state_dict(),checkpoint)
 clone=build_model(cfg,(16,1000)).cuda().eval()
 clone.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
 with torch.no_grad():assert torch.equal(clone(x),model(x))
 checkpoint.unlink()
 records.append(dict(arm=arm,ok=True,extra_parameters=extra,rows=16,shared_initial_state_exact=True,
                     conditional_rotation_recovery=True,orthogonal_coordinates=True,
                     finite_flat_forward=True,nonzero_pac_and_waveform_gradients=True,
                     checkpoint_roundtrip_exact=True,initial_carrier_projection_rank=rank))
 del model,clone,shared,changed,x,actual,b,out,expected,left,right,zero,phase,pac,amp,wave,unit
 if arm=='quadrature':del a
 torch.cuda.empty_cache()
p=Path('results')/f'modulation-contract-{os.environ["SLURM_JOB_ID"]}.json'
p.write_text(json.dumps(records,indent=2)+'\n');print(p.read_text(),flush=True)
