"""Short hardware/real-batch check; run only through smoke_gpu.slurm."""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import set_seed
run_id=os.environ.get('SLURM_JOB_ID') or os.environ.get('CFM_STANDALONE_RUN_ID')
assert run_id and torch.cuda.is_available()
assert os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') is not None
ap=argparse.ArgumentParser()
ap.add_argument('--config',default='configs/selfcoup/tuev_crofremo_s2.yaml')
ap.add_argument('--require-budget-fit',action='store_true')
ap.add_argument('--output',help='Distinct receipt path for concurrent smoke checks')
args=ap.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text());cfg['num_workers']=2
set_seed(int(cfg.get('seed',0)))
assert not cfg.get('disabled_reason'),cfg.get('disabled_reason')
tr,va,te,info=build_dataloaders(cfg)
x,y=next(iter(tr));x,y=x.cuda(),y.cuda().long()
m=build_model(cfg,info['input_shape']).cuda().train()
opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
loss_fn=build_loss(cfg);times=[]
path_warmups=[]
augmentation=getattr(m,'augment',None)
if augmentation is not None and hasattr(augmentation,'augs') and len(augmentation.augs):
 # A stochastic branch can first compile after the initial two steps. Warm
 # every configured path, including the taken/untaken flip, before timing.
 for op in [torch.nn.Identity()]+list(augmentation.augs):
  durations=[]
  force_flip=patch.object(op,'prob',1.0) if hasattr(op,'prob') else nullcontext()
  with force_flip,patch.object(augmentation,'forward',side_effect=op):
   for _ in range(2):
    torch.cuda.synchronize();t=time.monotonic()
    opt.zero_grad(set_to_none=True);loss=loss_fn(m(x),y)
    assert torch.isfinite(loss)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.get('grad_clip',1.0))
    assert torch.isfinite(norm)
    opt.step();torch.cuda.synchronize();durations.append(time.monotonic()-t)
  path_warmups.append(dict(path=type(op).__name__,seconds=durations))
steps=14 if path_warmups else 5
for i in range(steps):
 torch.cuda.synchronize();t=time.monotonic()
 opt.zero_grad(set_to_none=True);loss=loss_fn(m(x),y)
 assert torch.isfinite(loss)
 loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.get('grad_clip',1.0))
 assert torch.isfinite(norm)
 opt.step()
 torch.cuda.synchronize();times.append(time.monotonic()-t)
 assert all(torch.isfinite(p).all() for p in m.parameters())
projected_train_seconds=statistics.mean(times[2:])*len(tr)*cfg['epochs']
if args.require_budget_fit:
 assert cfg.get('max_hours') and projected_train_seconds<cfg['max_hours']*3600*.8, \
  ('Training alone would consume over 80% of the budget',projected_train_seconds,cfg.get('max_hours'))
p=Path(args.output) if args.output else Path('results')/('amd-partition-smoke-'+run_id+'.json')
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(dict(ok=True,partition=os.environ.get('SLURM_JOB_PARTITION'),
 job=run_id,config=args.config,seed=int(cfg.get('seed',0)),host=os.uname().nodename,torch=torch.__version__,rocm=torch.version.hip,
 device=torch.cuda.get_device_name(),visible_gpus=torch.cuda.device_count(),
 input_shape=list(x.shape),steps=steps,warmup_excluded=2,steady_step_seconds=times[2:],
 augmentation_path_warmups=path_warmups,
 peak_GiB=torch.cuda.max_memory_allocated()/2**30,
 projected_train_seconds=projected_train_seconds,train_batches=len(tr),val_batches=len(va),
 val_batch_size=va.batch_size,train_samples=info.get('n_samples'),
 projection_note='Excludes data loading and evaluation; not a completion guarantee.',
 config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
 smoke_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
print(p.read_text())
