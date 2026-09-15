"""Finite A128 pretrain->FT handover. Uses cat on B2; no model compute locally."""
import pathlib,subprocess,json,time,hashlib,os,fcntl
BASE=pathlib.Path('/Users/mr.z/PACLock-monitor');STATE=BASE/'a128-followup-state.json'
AMD='/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-a128-r1-20260916'
B2='/ocean/projects/cis260249p/qren2/Zhizhe/CFM4EEG-a128-r1-20260916/pretrain_runs/a128_latent_tueg_20260916/checkpoint_20000.pt'
lock=(BASE/'a128-followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=json.loads(STATE.read_text()) if STATE.exists() else {'created':time.time(),'phase':'waiting_final_pretrain'}
def save(**kw):
 s.update(kw,heartbeat=time.time());t=STATE.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2));t.replace(STATE)
def b2_init():subprocess.run(['expect','/Users/mr.z/.claude/jobs/18b2571f/tmp/b2_master.exp'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=40)
def remote(host,command):return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,command],text=True,timeout=90).strip()
while time.time()-s['created']<7*24*3600:
 try:
  if s.get('ft_job'):break
  if (BASE/'STOP_A128_FOLLOWUP').exists():save(phase='operator_stopped');break
  if s.get('phase')=='submitting':raise RuntimeError('Ambiguous previous FT submission needs manual reconciliation')
  if not s.get('checkpoint_verified'):
   b2_init();exists=remote('b2','test -f '+B2+' && echo ready || echo waiting')
   if exists!='ready':save(phase='waiting_final_pretrain');time.sleep(180);continue
   b2_init();expected=remote('b2','sha256sum '+B2).split()[0]
   local=BASE/'a128_latent_tueg_20260916.pt';part=local.with_suffix('.part')
   b2_init()
   with part.open('wb') as f:subprocess.run(['ssh','-o','BatchMode=yes','b2','cat '+B2],stdout=f,check=True,timeout=1800)
   assert hashlib.sha256(part.read_bytes()).hexdigest()==expected;part.replace(local)
   remote('amd','mkdir -p '+AMD+'/ckpt')
   subprocess.run(['rsync','-a',str(local),'amd:'+AMD+'/ckpt/'],check=True,timeout=300)
   assert remote('amd','sha256sum '+AMD+'/ckpt/'+local.name).split()[0]==expected
   save(phase='waiting_ft_capacity',checkpoint_verified=True,sha256=expected)
  q=remote('amd','squeue -h -u "$USER" -o "%i"').splitlines()
  if len(q)>=9:save(phase='waiting_ft_capacity');time.sleep(120);continue
  # Two single-seed FT arms plus two reserved attribution controls fill one node.
  # Replication is deliberately held until seed0 breadth has been reviewed.
  specs=[f'configs/a128_latent_ft/{d}_s0.yaml:0' for d in ['chbmit','tuev']]
  specs += [f'configs/a128_own/{d}_s0.yaml:0' for d in ['chbmit','tuev']]
  dispatched=json.loads(remote('amd','cat '+AMD+'/results/a128-dispatch.json'))
  ours=[v for v in dispatched['submitted'].values() if v['job'] in q]
  if len(ours)>=4:save(phase='waiting_ft_capacity');time.sleep(120);continue
  if (BASE/'STOP_A128_FOLLOWUP').exists():save(phase='operator_stopped');break
  save(phase='submitting')
  command='cd '+AMD+' && PACLOCK_CKPT='+AMD+'/ckpt sbatch --parsable -J A128_latent_FT slurm/a128_packed.slurm '+' '.join(specs)
  job=remote('amd',command).split(';')[0];assert job.isdigit(),job;save(phase='ft_submitted',ft_job=job)
 except Exception as e:
  save(error=repr(e));print(repr(e),flush=True)
  if s.get('phase')=='submitting':break
  time.sleep(180)
