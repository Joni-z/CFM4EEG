"""Run contract, real-batch smoke, then fresh training; never reuse smoke weights."""
import argparse,json,os,pathlib,signal,subprocess,sys,time
import yaml
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];os.chdir(root)
cfg=yaml.safe_load(pathlib.Path(a.config).read_text());assert cfg['seed']==a.seed
out=root/'runs'/cfg['name']/f'seed{a.seed}';key=pathlib.Path(a.config).stem
assert not any((out/f).exists() for f in ['best.pt','progress.json','result.json','STOP']), 'Run already exists or stopped'
receipt=root/'results'/f'{key}-pipeline.json';receipt.parent.mkdir(exist_ok=True)
state=dict(config=a.config,seed=a.seed,host=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),pid=os.getpid());child=None
# All stages run as children of this GPU owner. Training has its own seed reset.
def save(**kw):
 state.update(kw,updated=time.time());t=receipt.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(receipt)
def stop(sig,frame):
 if child and child.poll() is None:
  child.send_signal(signal.SIGTERM)
  try:child.wait(timeout=180)
  except subprocess.TimeoutExpired:child.kill()
 save(phase='stopped');raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
cmds=[['smoke/verify_range_controls.py','--config',a.config,'--output',f'results/{key}-contract.json'],['smoke/smoke_amd_partition.py','--config',a.config,'--require-budget-fit','--output',f'results/{key}-smoke.json'],['-m','paclock_bench.training.train','--config',a.config,'--seed',str(a.seed)]]
try:
 for i,c in enumerate(cmds):
  assert not (root/'STOP_ALL_RANGE').exists() and not (out/'STOP').exists()
  save(phase=['contract','smoke','training'][i]);child=subprocess.Popen([sys.executable,'-u']+c);save(child_pid=child.pid)
  try:rc=child.wait(timeout=[1200,1800,(float(cfg['max_hours'])+1)*3600][i])
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=180)
   except subprocess.TimeoutExpired:child.kill()
   raise
  if rc:raise RuntimeError(f'stage {i} exited {rc}')
 r=json.loads((out/'result.json').read_text());assert r['seed']==a.seed
 save(phase='finished',result=str(out/'result.json'),stopped_by=r.get('stopped_by'))
except Exception as e:save(phase='failed',error=repr(e));raise
