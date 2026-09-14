"""Four real-batch smokes on four allocated MI210s; never on a login node."""
import json,os,subprocess,sys
from pathlib import Path
assert os.environ.get('SLURM_JOB_PARTITION')=='mi2104x'
job=os.environ['SLURM_JOB_ID'];Path('logs').mkdir(exist_ok=True);Path('results').mkdir(exist_ok=True)
configs=['chbmit_quadrature','chbmit_carrier','tuev_quadrature','tuev_carrier']
# Contract check on a single visible GPU, before the parallel memory-heavy jobs.
env=os.environ.copy();env['HIP_VISIBLE_DEVICES']='0'
subprocess.run([sys.executable,'-u','smoke/verify_modulation_carrier.py'],env=env,check=True)
children=[]
for gpu,name in enumerate(configs):
 cfg=f'configs/modulation_carrier/{name}_s0.yaml'
 env=os.environ.copy();env['HIP_VISIBLE_DEVICES']=str(gpu)
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
 cache=f'/tmp/miopen_nbands_smoke_{job}_{gpu}'
 env['MIOPEN_USER_DB_PATH']=cache;env['MIOPEN_CUSTOM_CACHE_DIR']=cache
 log=Path(f'logs/modulation-smoke-{job}-{name}.out').open('w')
 cmd=[sys.executable,'-u','smoke/smoke_amd_partition.py','--config',cfg,'--output',f'results/modulation-smoke-{job}-{name}.json']
 children.append((name,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
results={}
for name,proc,log in children:
 results[name]=proc.wait();log.close()
print(json.dumps(results),flush=True)
assert all(code==0 for code in results.values()),results
