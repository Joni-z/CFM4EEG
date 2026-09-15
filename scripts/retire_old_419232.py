import os,signal,time,json
from pathlib import Path
ROOT=Path('/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-quad16-breadth-20260915')
assert os.environ.get('SLURM_JOB_ID')=='419232'
controllers=[63604,64258,63597,64251]
trainers=[63799,64400]
state={'phase':'starting','controllers':controllers,'retired_trainers':trainers}
receipt=ROOT/'results/retire-old-419232.json'
def save():
 state['heartbeat']=time.time();receipt.write_text(json.dumps(state,indent=2))
def ident(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().split();return s[21] if s[2]!='Z' else None
 except FileNotFoundError:return None
tokens={pid:ident(pid) for pid in controllers+trainers}
def send(pid,sig):
 if tokens[pid] and ident(pid)==tokens[pid]:os.kill(pid,sig)
def quad_active():
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=(p/'cmdline').read_bytes()
   if b'paclock_bench.training.train' in cmd and b'quad16_seeds/tuev_s' in cmd and 'job_419232' in (p/'cgroup').read_text() and ident(int(p.name)):
    active.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return active
assert quad_active(),'No Quad16 TUEV trainers found; refuse lifecycle change'
for pid in controllers+trainers:
 p=Path('/proc')/str(pid)
 assert tokens[pid] and 'job_419232' in (p/'cgroup').read_text()
 cmd=(p/'cmdline').read_bytes()
 assert (b'adopt_amd.py' in cmd or b'backfill_amd.py' in cmd) if pid in controllers else b'configs/runtime/' in cmd
try:
 # Pause only lifecycle controllers and their rescue processes; preserve all Quad16 children.
 for pid in controllers:send(pid,signal.SIGSTOP)
 state['phase']='controllers_paused';save()
 for pid in trainers:send(pid,signal.SIGTERM)
 deadline=time.time()+180
 while any(ident(pid)==tokens[pid] for pid in trainers) and time.time()<deadline:save();time.sleep(3)
 for pid in trainers:
  if ident(pid)==tokens[pid]:send(pid,signal.SIGKILL)
 state['phase']='old_stopped_waiting_quad16';save()
 while quad_active():
  state['quad16_pids']=quad_active();save();time.sleep(10)
 state['phase']='quad16_finished_restoring_controllers'
finally:
 for pid in controllers:send(pid,signal.SIGCONT)
 save()
