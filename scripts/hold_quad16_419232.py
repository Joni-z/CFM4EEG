import os,signal,time,json
from pathlib import Path
ROOT=Path('/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-quad16-breadth-20260915')
assert os.environ.get('SLURM_JOB_ID')=='419232'
controllers=[63604,64258,63597,64251]
def identity(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().split();return s[21] if s[2]!='Z' else None
 except FileNotFoundError:return None
tokens={pid:identity(pid) for pid in controllers}
old=Path('/proc/74929');assert b'retire_old_419232.py' in (old/'cmdline').read_bytes()
for pid in controllers:assert (Path('/proc')/str(pid)/'stat').read_text().split()[2]=='T'
# Replace the old narrow TUEV-only waiter, without waking lifecycle controllers.
os.kill(74929,signal.SIGKILL)
state={'phase':'holding_all_quad16','controllers':controllers}
def save():
 state['heartbeat']=time.time();(ROOT/'results/hold-quad16-419232.json').write_text(json.dumps(state,indent=2))
def stop(sig,frame):raise InterruptedError(sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 while True:
  active=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    cmd=(p/'cmdline').read_bytes()
    if b'configs/quad16_seeds/' in cmd and (b'paclock_bench.training.train' in cmd or b'smoke_amd_partition.py' in cmd or b'slurm/quad16_seed_amd.slurm' in cmd) and 'job_419232' in (p/'cgroup').read_text() and identity(int(p.name)):
     active.append(int(p.name))
   except (OSError,ValueError):pass
  state['active_pids']=active;save()
  if not active:break
  time.sleep(10)
finally:
 for pid in controllers:
  if identity(pid)==tokens[pid]:os.kill(pid,signal.SIGCONT)
 state['phase']='released';save()
