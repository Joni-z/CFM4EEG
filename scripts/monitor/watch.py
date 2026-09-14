"""Bounded, idempotent watcher. Standard library only; never imports torch locally."""
from pathlib import Path
import datetime,fcntl,hashlib,json,math,os,re,shlex,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
AMD='/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-nbands-20260915'
PREP='/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-anchored-watch-20260915'
TORCH='/scratch/zz5070/CFM4EEG-anchored-watch-20260915'
NAMES=[f'{ds}-nb16_{op}_20260915' for ds in ['chbmit','tuev'] for op in ['quadrature','carrier']]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def ssh(host,code):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','LogLevel=ERROR',host,'python3 -'],input=code,text=True,capture_output=True,timeout=75)
 if p.returncode:raise RuntimeError(host+': '+p.stderr[-1200:])
 return json.loads(p.stdout)
def epoch_position(tag):
 m=re.search(r'epoch\s+(\d+)(?: step (\d+))?',tag)
 return int(m[1])+(int(m[2])/9882 if m[2] else 1) if m else 0

def decision(name,r):
 if r.get('read_error'):return None
 h=r.get('history',[])
 if not h:return 'technical_exit' if r.get('exit_code') not in (None,0) else None
 for row in h[-2:]:
  vals=[row.get('train_loss'),row.get('val_loss'),*row.get('metrics',{}).values()]
  if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in vals):return 'nonfinite'
 if r.get('exit_code') not in (None,0):return 'technical_exit'
 key='pr_auc' if name.startswith('chbmit') else 'cohen_kappa'
 rows=[x for x in h if key in x.get('metrics',{})]
 if not rows:return None
 selected=max(rows,key=lambda x:x['metrics'][key]);best=selected['metrics'][key]
 res=r.get('result')
 if res:
  if res.get('stopped_by') not in ('epochs','patience'):return 'budget_censored' if best < (.658727686 if key=='pr_auc' else .623040061) else None
  if key=='pr_auc' and best<.658727686:return 'completed_below_gate'
  if key=='cohen_kappa' and (best<.623040061 or selected['metrics'].get('balanced_acc',0)<.5045):return 'completed_below_gate'
 if r.get('stopped'):return None
 if key=='pr_auc' and len(rows)>20:
  pos=epoch_position(rows[-1]['tag']);old=max(x['metrics'][key] for x in rows[:-20])
  plateau=best-old<.01
  if plateau and pos>=3 and best<.30:return 'epoch3_low_plateau'
  if plateau and pos>=2 and best<.15:return 'epoch2_low_plateau'
 return None

AMD_READ=f'''from pathlib import Path
import json,subprocess,time
root=Path({AMD!r});prep=Path({PREP!r})
def read(p):return json.loads(p.read_text()) if p.exists() else None
pack=read(root/'logs/F_modcarrier_s0-419232-pack.json') or {{}}
active={{r['name']:r for r in pack.get('running',[])}};done={{r['name']:r for r in pack.get('done',[])}}
runs={{}}
for name in {NAMES!r}:
 p=root/'runs'/name/'seed0';r={{'active':name in active,'exit_code':done.get(name,{{}}).get('exit_code')}}
 try:
  prog=read(p/'progress.json') or {{}};r.update(history=prog.get('history',[]),pid=prog.get('pid'),result=read(p/'result.json'),stopped=read(p/'stopped.json'),progress_age=time.time()-(p/'progress.json').stat().st_mtime if (p/'progress.json').exists() else None)
 except Exception as e:r['read_error']=str(e)
 runs[name]=r
smoke={{'contract':read(prep/'results/anchored-contract.json'),'tuev':read(prep/'results/anchored-smoke-tuev.json'),'chbmit':read(prep/'results/anchored-smoke-chbmit.json')}}
smoke['accounting']=subprocess.check_output(['sacct','-n','-X','-j','419454','--format=JobIDRaw,State,ExitCode','-P'],universal_newlines=True)
queue=subprocess.check_output(['squeue','-h','-u',__import__('os').environ['USER'],'-o','%i|%j|%T|%M'],universal_newlines=True)
print(json.dumps(dict(runs=runs,smoke=smoke,queue=queue)))
'''
TORCH_READ=fr'''from pathlib import Path
import json,re,subprocess,os
root=Path({TORCH!r});old=Path('/scratch/zz5070/PACLock');out={{}}
for seed,job in [(1,17365676),(2,17365677)]:
 p=old/f'logs/seed_tusz_cbramod_crofremo_bands_s{{seed}}-{{job}}.out';s=p.read_text() if p.exists() else '';rows=[]
 for e,b,pr,auc in re.findall(r'epoch\s+(\d+) \| val balanced_acc=([\d.]+) pr_auc=([\d.]+) auroc=([\d.]+)',s):rows.append(dict(epoch=int(e),pr=float(pr),auroc=float(auc)))
 out[str(job)]={{'epochs':len(rows),'selected':max(rows,key=lambda x:x['auroc']) if rows else None,'last':rows[-1] if rows else None}}
new={{}}
for ds in ['chbmit','tuev']:
 p=root/'runs'/f'{{ds}}-nb16_anchored_20260915'/'seed0';d={{}}
 for file in ['result.json','stopped.json','progress.json']:
  q=p/file
  if q.exists():
   try:
    v=json.loads(q.read_text());d[file]=v if file!='progress.json' else {{'last':v.get('history',[])[-1:]}}
   except Exception as e:d['read_error']=str(e)
 receipt=root/'controller'/f'{{ds}}.json'
 if receipt.exists():d['submission']=json.loads(receipt.read_text())
 new[ds]=d
queue=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%M'],universal_newlines=True)
print(json.dumps(dict(legacy=out,fallback=new,queue=queue)))
'''
def smoke_ready(s):
 if '419454|COMPLETED|0:0' not in s.get('accounting',''):return False
 if not (s.get('contract') or {}).get('ok'):return False
 for ds in ['chbmit','tuev']:
  r=s.get(ds) or {};p=ROOT/f'configs/anchored_watch/{ds}_s0.yaml'
  if not r.get('ok') or r.get('config_sha256')!=hashlib.sha256(p.read_bytes()).hexdigest():return False
 return True

def stop(name,r,reason):
 assert name in NAMES
 code=f'''from pathlib import Path
import json,os
root=Path({AMD!r});name={name!r};p=root/'runs'/name/'seed0'
pack=json.loads((root/'logs/F_modcarrier_s0-419232-pack.json').read_text())
active=[x for x in pack.get('running',[]) if x['name']==name]
assert len(active)==1 and active[0]['pid']=={r.get('pid')!r},'ownership changed'
assert not (p/'result.json').exists() and (p/'best.pt').exists(),'finished or no saved checkpoint'
q=p/'STOP'
if not q.exists():q.write_text({reason!r})
print(json.dumps(dict(stop_requested=True,name=name,checkpoint_preserved=True)))
'''
 return ssh('amd',code)

def submit(ds):
 # Reservation written remotely BEFORE sbatch. An ambiguous outcome cannot be
 # retried automatically, preventing duplicates after an SSH/Slurm disruption.
 code=f'''from pathlib import Path
import fcntl,json,subprocess,time,os
root=Path({TORCH!r});os.chdir(str(root));p=root/'controller'/{(ds+'.json')!r}
lock=open(str(p)+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
if p.exists():print(p.read_text());raise SystemExit(0)
out=root/'runs'/{(ds+'-nb16_anchored_20260915')!r}/'seed0'
assert not out.exists(),'existing output'
reservation={{'state':'submitting','dataset':{ds!r},'at':time.time()}}
p.write_text(json.dumps(reservation))
r=subprocess.run(['sbatch','--parsable','-J',{('F_anchor_'+ds+'_s0')!r},'torch_standby.slurm',{ds!r}],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
if r.returncode==0 and r.stdout.strip().split(';')[0].isdigit():reservation.update(state='submitted',job_id=r.stdout.strip().split(';')[0])
else:reservation.update(state='submission_failed',stderr=r.stderr[-1500:])
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(reservation));tmp.replace(p);print(json.dumps(reservation))
'''
 return ssh('torch',code)

def main():
 lock=(ROOT/'watch.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();return
 if (ROOT/'STOP').exists():lock.close();return
 p=ROOT/'state.json';state=json.loads(p.read_text()) if p.exists() else dict(created=now(),deadline=time.time()+72*3600,decisions={},submissions={})
 if time.time()>state['deadline']:lock.close();return
 state['heartbeat']=now();state['errors']=[];state['alerts']=[]
 try:amd=ssh('amd',AMD_READ);dump(ROOT/'amd-latest.json',amd)
 except Exception as e:state['errors'].append(str(e));amd=None
 try:torch=ssh('torch',TORCH_READ);dump(ROOT/'torch-latest.json',torch)
 except Exception as e:state['errors'].append(str(e));torch=None
 if amd:
  for name,r in amd['runs'].items():
   age=r.get('progress_age')
   if r['active'] and age is not None and age>2400:state['alerts'].append(name+': stale progress; no automatic kill')
   reason=decision(name,r)
   if reason and name not in state['decisions']:
    if r['active'] and not r.get('result') and not r.get('stopped'):
     try:stop(name,r,reason)
     except Exception as e:state['errors'].append('stop '+name+': '+str(e));continue
    state['decisions'][name]={'reason':reason,'at':now(),'censored':reason not in ['completed_below_gate']}
    dump(p,state)
  ready=smoke_ready(amd['smoke']);state['smoke_ready']=ready
  # Source failures trigger one bounded pair; never replicate passing models.
  if ready and state['decisions'] and torch:
   active=sum(r['active'] for r in amd['runs'].values())
   active+=sum(1 for x in torch['queue'].splitlines() if '|F_anchor_' in x)
   for ds in ['chbmit','tuev']:
    if ds in state['submissions'] or active>=4:continue
    try:
     receipt=submit(ds);state['submissions'][ds]=receipt;active+=1;dump(p,state)
    except Exception as e:state['errors'].append('submit '+ds+': '+str(e))
 state['summary']={'amd_active':sum(r['active'] for r in amd['runs'].values()) if amd else None,'legacy_torch':torch['legacy'] if torch else None}
 dump(p,state);lock.close();print(json.dumps({'at':now(),'smoke_ready':state.get('smoke_ready'),'decisions':state['decisions'],'submissions':state['submissions'],'errors':state['errors'],'alerts':state['alerts']}),flush=True)
if __name__=='__main__':main()
