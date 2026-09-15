"""Finish two smoke-rejected configurations inside the same exclusive allocation."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/work1/chenyuyou/yifanwang/Zhizhe/CFM4EEG-transfer-pretrain-20260915');os.chdir(ROOT)
receipt=ROOT/'results/host-repair-420562.json'
x=json.loads((ROOT/'logs/Q16_hosts-420562-pack.json').read_text());assert not x['pending']
runners=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  s=p.read_bytes()
  if b'scripts/slurm/run_quad_hosts_packed.py' in s and b'python' in s.split(b'\0')[0]:runners.append(int(p.parent.name))
 except (OSError,PermissionError):pass
assert len(runners)==1,runners
runner=runners[0]
# Hold only our pack supervisor so it cannot release the node while recovery
# uses its finished slot. Existing training children continue unaffected.
os.kill(runner,signal.SIGSTOP)
def save(**s):receipt.write_text(json.dumps(dict(supervisor=runner,**s),indent=2))
try:
 slot=next(v for v in x['running'] if v['gpu']==1);save(phase='waiting_slot',old_pid=slot['pid'])
 while (Path('/proc')/str(slot['pid'])).exists():
  if (Path('/proc')/str(slot['pid'])/'stat').read_text().split(') ')[1].split()[0]=='Z':break
  time.sleep(10)
 # Data restoration must finish before this config can pass admission.
 data=Path('/work1/chenyuyou/yifanwang/Zhizhe/processed_biot/tuev')
 while not (data/'manifest.json').exists():time.sleep(10)
 env=os.environ.copy();env['HIP_VISIBLE_DEVICES']='1'
 for cfg in ['configs/quad_hosts/tuev_biot_s0.yaml','configs/quad_hosts/chbmit_cbramod_s0.yaml']:
  save(phase='smoke_then_train',config=cfg)
  with open('logs/repair-'+Path(cfg).stem+'.out','a') as f:
   subprocess.run(['bash','scripts/run_quad_host.sh',cfg,'0'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 save(phase='finished')
finally:os.kill(runner,signal.SIGCONT)
