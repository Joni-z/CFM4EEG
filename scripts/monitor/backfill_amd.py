"""Bounded residual-16 screen on free slots of existing allocation only."""
import atexit,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-anchored-watch-20260915')
STATE=ROOT/'results/backfill-419232.json'
SHELL=50528

def proc(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text().split()
  return s[2],s[21]
 except FileNotFoundError:return None

def alive(pid,token=None):
 s=proc(pid)
 return bool(s and s[0] not in ('Z','X') and (token is None or s[1]==token))

def resume(token):
 if alive(SHELL,token):os.kill(SHELL,signal.SIGCONT)

def free_slots(occupied):return [i for i in range(4) if i not in occupied]

if __name__=='__main__' and '--rescue' in sys.argv:
 parent,ptoken,stoken=sys.argv[2:]
 while alive(int(parent),ptoken):
  if STATE.exists() and time.time()-STATE.stat().st_mtime>240:break
  time.sleep(10)
 resume(stoken);sys.exit(0)

def main():
 import yaml
 from watch import decision
 assert os.environ['SLURM_JOB_ID']=='419232' and 'k002-005' in os.uname().nodename
 os.chdir(ROOT)
 assert not STATE.exists(),'already admitted'
 assert 'job_419232' in Path(f'/proc/{SHELL}/cgroup').read_text()
 assert 'bash' in Path(f'/proc/{SHELL}/cmdline').read_text()
 shell_token=proc(SHELL)[1]
 end=int(os.environ['HANDOVER_END_EPOCH'])
 assert end-time.time()>10*3600
 state={'phase':'running','pid':os.getpid(),'shell_pid':SHELL,'shell_token':shell_token,'runs':{},'pending':['chbmit','tuev']}
 children={};envs={};stopping=[False];handles=[]
 def snapshot():
  state['heartbeat']=time.time();p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(STATE)
 def stop(sig,frame):stopping[0]=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 atexit.register(resume,shell_token)
 os.kill(SHELL,signal.SIGSTOP);snapshot()
 rescue=subprocess.Popen([sys.executable,__file__,'--rescue',str(os.getpid()),proc(os.getpid())[1],shell_token])
 try:
  while state['pending'] or any(p.poll() is None for p in children.values()):
   for ds,p in children.items():
    r=state['runs'][ds];r['exit_code']=p.poll()
    if r['phase']=='smoking' and p.poll() is None and time.time()-r['started_at']>1200:raise RuntimeError('smoke timeout')
    if r['phase']=='training' and p.poll() is None:
     directory=ROOT/'runs'/r['name']/'seed0';progress=directory/'progress.json'
     if progress.exists():
      try:
       d=json.loads(progress.read_text());reason=decision(r['name'],{'history':d.get('history',[])})
       if reason and (directory/'best.pt').exists():
        (directory/'STOP').write_text(reason);r['stop_reason']=reason
      except (ValueError,FileNotFoundError):pass
    if r['phase']=='smoking' and p.poll() is not None:
     receipt=ROOT/f'results/backfill-smoke-{ds}-419232.json'
     if p.returncode!=0 or not receipt.exists() or not json.loads(receipt.read_text()).get('ok'):
      r['phase']='smoke_failed';continue
     cfg=yaml.safe_load(Path(r['config']).read_text())
     cfg['max_hours']=min(10.5 if ds=='chbmit' else 6.5,(end-time.time()-900)/3600)
     Path(r['config']).write_text(yaml.safe_dump(cfg,sort_keys=False))
     log=ROOT/f'logs/F_residual16-419232-{ds}.out';fh=log.open('x');handles.append(fh)
     children[ds]=subprocess.Popen([sys.executable,'-u','-m','paclock_bench.training.train','--config',r['config'],'--seed','0'],env=envs[ds],stdout=fh,stderr=subprocess.STDOUT)
     r.update(phase='training',pid=children[ds].pid,exit_code=None,log=str(log),max_hours=cfg['max_hours'])
    elif r['phase']=='training' and p.poll() is not None:r['phase']='finished'
   if stopping[0] or time.time()>end-300:
    state['pending']=[]
    for p in children.values():
     if p.poll() is None:p.send_signal(signal.SIGTERM)
   else:
    # Known owners plus every GPU-masked training/smoke process in this allocation.
    occupied=set()
    for p in Path('/proc').iterdir():
     if not p.name.isdigit():continue
     try:
      if not alive(int(p.name)) or 'job_419232' not in (p/'cgroup').read_text():continue
      cmd=(p/'cmdline').read_bytes()
      if b'paclock_bench.training.train' not in cmd and b'smoke_amd_partition' not in cmd:continue
      env=(p/'environ').read_bytes().split(b'\0')
      values=[v.split(b'=',1)[1] for v in env if v.startswith(b'HIP_VISIBLE_DEVICES=')]
      if not values:raise RuntimeError('unmapped GPU trainer in allocation')
      occupied.update(int(v) for v in values[0].split(b','))
     except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    for gpu in free_slots(occupied):
     if not state['pending']:break
     ds=state['pending'][0]
     minimum=9 if ds=='chbmit' else 5
     if end-time.time()<(minimum*3600):
      state.setdefault('deferred',[]).append({'dataset':ds,'reason':'insufficient_remaining_allocation'});state['pending'].pop(0);continue
     state['pending'].pop(0)
     cfg=yaml.safe_load((ROOT/f'configs/anchored_watch/{ds}_s0.yaml').read_text())
     name=f'{ds}-nb16_residual_20260915';cfg.update(name=name,max_hours=10.5 if ds=='chbmit' else 6.5)
     cfg['model_kwargs'].pop('waveform_modulation');cfg['model_kwargs']['local_residual']=True
     assert not (ROOT/'runs'/name/'seed0').exists()
     cp=ROOT/f'configs/runtime/residual16_{ds}_419232.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
     env=os.environ.copy();env['HIP_VISIBLE_DEVICES']=str(gpu)
     for k in ['ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']:env.pop(k,None)
     for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
     env['MIOPEN_USER_DB_PATH']=f'/tmp/residual16_419232_{gpu}';env['MIOPEN_CUSTOM_CACHE_DIR']=env['MIOPEN_USER_DB_PATH']
     fh=(ROOT/f'logs/F_residual16-419232-{ds}-smoke.out').open('x');handles.append(fh)
     children[ds]=subprocess.Popen([sys.executable,'-u','smoke/smoke_amd_partition.py','--config',str(cp),'--output',f'results/backfill-smoke-{ds}-419232.json'],env=env,stdout=fh,stderr=subprocess.STDOUT)
     envs[ds]=env
     state['runs'][ds]={'name':name,'gpu':gpu,'pid':children[ds].pid,'phase':'smoking','config':str(cp),'started_at':time.time()}
   snapshot();time.sleep(10)
  state['phase']='finished'
 except BaseException as e:
  state['phase']='failed';state['error']=str(e)
  for p in children.values():
   if p.poll() is None:p.send_signal(signal.SIGTERM)
  for p in children.values():
   try:p.wait(timeout=90)
   except subprocess.TimeoutExpired:p.kill()
  raise
 finally:
  snapshot();resume(shell_token);rescue.terminate()
  for fh in handles:fh.close()

if __name__=='__main__':main()
