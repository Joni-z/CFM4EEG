"""Attach two admitted pilots to known free GPUs in allocation 419232.
Hold only the old dispatcher (SIGSTOP), never its trainer; resume it after
new children finish. The batch shell therefore retains the existing allocation.
"""
import atexit,json,os,signal,subprocess,sys,time
from pathlib import Path
JOB='419232';RUNNER=50534;OLDTRAIN=50602
OLD=Path('/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-nbands-20260915')
ROOT=Path('/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-anchored-watch-20260915')
STATE=ROOT/'results'/'amd-handover-419232.json';ALLOW=ROOT/'results'/'amd-handover-419232.allow'
def identity(pid):
 p=Path(f'/proc/{pid}')
 try:return (p/'stat').read_text().split()[21]
 except FileNotFoundError:return None

def atomic(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def resume(token):
 if identity(RUNNER)==token:os.kill(RUNNER,signal.SIGCONT)

if '--rescue' in sys.argv:
 _,_,parent,parent_token,runner_token=sys.argv
 # Separate process survives an unexpected death of the Python controller.
 while identity(int(parent))==parent_token:
  if STATE.exists() and time.time()-STATE.stat().st_mtime>240:
   resume(runner_token);raise SystemExit(0)
  time.sleep(10)
 resume(runner_token);raise SystemExit(0)

assert os.environ.get('SLURM_JOB_ID')==JOB
assert 'k002-005' in os.uname().nodename
os.chdir(ROOT)
assert not STATE.exists(),'handover already initialized'
runner_token=identity(RUNNER);assert runner_token
assert 'scripts/slurm/run_packed.py' in Path(f'/proc/{RUNNER}/cmdline').read_bytes().decode().replace('\0',' ')
assert 'job_'+JOB in Path(f'/proc/{RUNNER}/cgroup').read_text()
status=Path(f'/proc/{OLDTRAIN}/status').read_text();assert f'PPid:\t{RUNNER}' in status
pack=json.loads((OLD/'logs/F_modcarrier_s0-419232-pack.json').read_text())
assert [(r['gpu'],r['pid']) for r in pack['running']]==[(0,OLDTRAIN)],pack['running']
for ds in ['chbmit','tuev']:
 assert not (ROOT/'runs'/f'{ds}-nb16_anchored_20260915'/'seed0').exists()
 assert json.loads((ROOT/f'results/anchored-smoke-{ds}.json').read_text())['ok']

remaining=int(os.environ['HANDOVER_END_EPOCH'])-time.time()
assert remaining>8*3600,'not enough allocation time for planned pilot budgets'
state=dict(job=JOB,controller_pid=os.getpid(),runner_pid=RUNNER,runner_start=runner_token,
           old_trainer_pid=OLDTRAIN,phase='holding_dispatcher',started_at=time.time(),runs={})
children={};stopping=[False];logs=[]
def snapshot():state['heartbeat']=time.time();atomic(STATE,state)
def stop(sig,frame):stopping[0]=True;state['stop_signal']=sig
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
atexit.register(resume,runner_token)
os.kill(RUNNER,signal.SIGSTOP);snapshot()
rescue=subprocess.Popen([sys.executable,__file__,'--rescue',str(os.getpid()),identity(os.getpid()),runner_token])
try:
 import yaml
 for ds,gpu in [('chbmit',1),('tuev',2)]:
  cfg=yaml.safe_load((ROOT/f'configs/anchored_watch/{ds}_s0.yaml').read_text())
  cfg['max_hours']=min(10.5 if ds=='chbmit' else 6.5,(remaining-3600)/3600)
  cp=ROOT/f'configs/runtime/amd_{ds}_{JOB}.yaml';cp.parent.mkdir(exist_ok=True);cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
  env=os.environ.copy();env['HIP_VISIBLE_DEVICES']=str(gpu)
  for k in ['ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:env.pop(k,None)
  for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
  env['MIOPEN_USER_DB_PATH']=f'/tmp/miopen_anchor_{JOB}_{gpu}';env['MIOPEN_CUSTOM_CACHE_DIR']=env['MIOPEN_USER_DB_PATH']
  logfile=ROOT/f'logs/F_anchor_adopt-{JOB}-{ds}-smoke.out';fh=logfile.open('x');logs.append(fh)
  proc=subprocess.Popen([sys.executable,'-u','smoke/smoke_amd_partition.py','--config',str(cp),'--output',f'results/adopt-smoke-{ds}-{JOB}.json'],env=env,stdout=fh,stderr=subprocess.STDOUT)
  children[ds]=(proc,env,cp);state['runs'][ds]=dict(gpu=gpu,smoke_pid=proc.pid,config=str(cp),max_hours=cfg['max_hours'])
 state['phase']='smoking';snapshot();limit=time.time()+1200
 while any(p.poll() is None for p,_,_ in children.values()):
  if stopping[0] or time.time()>limit:raise RuntimeError('smoke stopped or timed out')
  snapshot();time.sleep(5)
 assert all(p.returncode==0 for p,_,_ in children.values()),'real-config smoke failed'
 for ds in children:
  receipt=json.loads((ROOT/f'results/adopt-smoke-{ds}-{JOB}.json').read_text());assert receipt['ok'] and receipt['warmup_excluded']>=2
 state['phase']='ready_for_torch_cancellation';snapshot();limit=time.time()+600
 while not ALLOW.exists():
  if stopping[0] or time.time()>limit:raise RuntimeError('handover authorization timed out')
  snapshot();time.sleep(2)
 permit=json.loads(ALLOW.read_text());assert permit['cancelled_torch_jobs']==['17828676','17828677']
 for ds,(_,env,cp) in list(children.items()):
  logfile=ROOT/f'logs/F_anchor_adopt-{JOB}-{ds}.out';fh=logfile.open('x');logs.append(fh)
  proc=subprocess.Popen([sys.executable,'-u','-m','paclock_bench.training.train','--config',str(cp),'--seed','0'],env=env,stdout=fh,stderr=subprocess.STDOUT)
  children[ds]=(proc,env,cp);state['runs'][ds].update(pid=proc.pid,log=str(logfile),started_at=time.time())
 state['phase']='training';snapshot()
 while any(p.poll() is None for p,_,_ in children.values()):
  if stopping[0] or time.time()>int(os.environ['HANDOVER_END_EPOCH'])-240:
   for p,_,_ in children.values():
    if p.poll() is None:p.send_signal(signal.SIGTERM)
   state['phase']='stopping';snapshot()
  for ds,(p,_,_) in children.items():state['runs'][ds]['exit_code']=p.poll()
  snapshot();time.sleep(10)
 state['phase']='finished'
 for ds,(p,_,_) in children.items():state['runs'][ds]['exit_code']=p.returncode
except BaseException as e:
 state['phase']='failed';state['error']=str(e)
 for p,_,_ in children.values():
  if p.poll() is None:p.send_signal(signal.SIGTERM)
 for p,_,_ in children.values():
  try:p.wait(timeout=90)
  except subprocess.TimeoutExpired:p.kill()
 raise
finally:
 state['finished_at']=time.time();snapshot();resume(runner_token)
 rescue.terminate()
 for fh in logs:fh.close()
