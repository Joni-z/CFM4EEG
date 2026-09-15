"""Finite, idempotent SLURM dispatcher; never performs model computation."""
import json,os,pathlib,subprocess,time,sys,fcntl
R=pathlib.Path(__file__).resolve().parents[1];os.chdir(R)
plan=json.loads((R/'configs/a128_execution.json').read_text());statepath=R/'results/a128-dispatch.json'
lock=open(R/'results/a128-dispatch.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads(statepath.read_text()) if statepath.exists() else {'submitted':{},'created_at':time.time()}
if state.get('submission_in_progress'):raise RuntimeError('Previous submission outcome needs reconciliation before restart')
def save():
 state['heartbeat']=time.time();t=statepath.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(statepath)
def cmd(a):return subprocess.check_output(a,text=True).strip()
while time.time()-state['created_at']<72*3600:
 if (R/'results/STOP_DISPATCH').exists():state['phase']='operator_stopped';save();break
 q=cmd(['squeue','-h','-u',os.environ['USER'],'-o','%i %T']);active={x.split()[0] for x in q.splitlines() if x.strip()}
 gate=R/plan['gate_receipt']
 if not gate.exists():
  state['phase']='waiting_contract_smoke';save()
  if plan['gate_job'] not in active:raise RuntimeError('Contract smoke ended without an admission receipt')
  time.sleep(20);continue
 assert json.loads(gate.read_text())['ok']
 outstanding=[v for v in state['submitted'].values() if v['job'] in active]
 todo=[x for x in plan['packs'] if x['name'] not in state['submitted']]
 if not todo:state['phase']='all_submitted';save();break
 if len(outstanding)>=plan['max_active'] or len(active)>=9:state['phase']='waiting_queue_capacity';save();time.sleep(20);continue
 pack=todo[0]
 # Receipt persisted immediately; no automatic retry after an ambiguous submission.
 state['phase']='submitting';state['submission_in_progress']=pack['name'];save()
 args=['sbatch','--parsable','-J',pack['name'],'-p',pack['partition'],'--time='+('12:00:00' if pack['partition']=='mi2101x' else '24:00:00')]
 if pack['partition']=='mi2101x':args+=['-c','16']
 job=cmd(args+['slurm/a128_packed.slurm']+pack['specs']).split(';')[0]
 assert job.isdigit(),job
 state['submitted'][pack['name']]={'job':job,'submitted_at':time.time(),**pack};state.pop('submission_in_progress',None);save();print(pack['name'],job,flush=True);time.sleep(2)
