"""GPU contract and both real-config smokes. No training on login nodes."""
import io,json,os,sys,subprocess
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.models.build import build_model
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
cfg=yaml.safe_load(Path('configs/anchored_watch/tuev_s0.yaml').read_text())
torch.manual_seed(0);m=build_model(cfg,(16,1000)).cuda().eval();fe=m.frontend
assert fe.n_token_bands==16 and torch.count_nonzero(fe.modulation_gate)==0
# The actual method must return the waveform carrier unchanged at g=0.
phase=torch.complex(torch.randn(2,1,2,16,64,device='cuda'),torch.randn(2,1,2,16,64,device='cuda'))
a=torch.randn_like(phase.real);w=torch.randn_like(a)
z=torch.complex(torch.randn(2,1,2,16,16,device='cuda'),torch.randn(2,1,2,16,16,device='cuda'))
h=fe._pac_interaction(phase,a,z,waveform_feat=w);carrier=torch.complex(a,w)
assert torch.equal(h,carrier)
h.real.square().mean().backward();assert fe.modulation_gate.grad is not None and fe.modulation_gate.grad.abs().sum()>0
with torch.no_grad():fe.modulation_gate.fill_(.7)
h=fe._pac_interaction(phase,a,z,waveform_feat=w)
assert torch.allclose(h.abs(),carrier.abs(),atol=2e-6,rtol=2e-6)
assert torch.max((h/carrier).angle().abs()).item()<.524
m.zero_grad();x=torch.randn(2,16,1000,device='cuda');logits=m(x);assert torch.isfinite(logits).all();logits.square().mean().backward()
assert fe.phase_tokenizer.weight.grad is not None and fe.phase_tokenizer.weight.grad.abs().sum()>0
assert fe.waveform_quadrature.weight.grad.abs().sum()>0
with torch.no_grad():assert torch.isfinite(m(torch.zeros_like(x))).all()
state=io.BytesIO();torch.save(m.state_dict(),state);state.seek(0)
m2=build_model(cfg,(16,1000)).cuda().eval();m2.load_state_dict(torch.load(state,weights_only=True),strict=True)
with torch.no_grad():assert torch.equal(m(x),m2(x))
del m,m2,fe,h,logits;torch.cuda.empty_cache()
Path('results').mkdir(exist_ok=True)
Path('results/anchored-contract.json').write_text(json.dumps({'ok':True,'job':os.environ['SLURM_JOB_ID'],'identity_init':True,'bounded_rotation':True,'gradients':True,'checkpoint_roundtrip':True}))
if '--contract-only' in sys.argv: raise SystemExit(0)
for ds in ['tuev','chbmit']:
 subprocess.run([sys.executable,'smoke/smoke_amd_partition.py','--config',f'configs/anchored_watch/{ds}_s0.yaml','--output',f'results/anchored-smoke-{ds}.json'],check=True)
