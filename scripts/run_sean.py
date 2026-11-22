"""Smoke-gated run, with shared GPU leasing on 145 and durable receipts."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time
import yaml
ROOT=pathlib.Path(__file__).resolve().parents[1];os.chdir(ROOT)
cfgpath=pathlib.Path(sys.argv[1]).resolve();cfg=yaml.safe_load(cfgpath.read_text())
name=cfg['name'];seed=cfg['seed'];out=ROOT/'runs'/name/f'seed{seed}'
res=ROOT/'results'/name/f'seed{seed}';res.mkdir(parents=True,exist_ok=True)
lock=open(res/'launcher.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not any((out/n).exists() for n in ['best.pt','progress.json','result.json','STOP'])
state={'pid':os.getpid(),'config':str(cfgpath),'seed':seed};child=None
def save(**kw):
    state.update(kw,updated=time.time());p=res/'pipeline.json';q=p.with_suffix('.tmp')
    q.write_text(json.dumps(state,indent=2));q.replace(p)
def stop(sig,frame):
    if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    save(phase='operator_stopped');raise SystemExit(128+sig)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
leases=[]
if not env.get('SLURM_JOB_ID'):
    directory=pathlib.Path('/data2/zz5070/CFM4EEG-quad16-breadth-20260915/results/shared-gpu-locks');directory.mkdir(parents=True,exist_ok=True)
    def free():
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
        ans={}
        for row in rows.splitlines():
            i,u,mem,util=map(str.strip,row.split(','))
            if u not in apps and float(mem)<100 and float(util)==0:ans[int(i)]=u
        return ans
    while not leases:
        save(phase='waiting_free_gpu')
        if (ROOT/'STOP_ALL_RANGE').exists():raise RuntimeError('Stop marker present')
        for gpu,uuid in free().items():
            lease=open(directory/f'gpu-{gpu}.lock','a')
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease.close();continue
            time.sleep(3)
            if free().get(gpu)==uuid:leases.append(lease);break
            lease.close()
        if not leases:time.sleep(20)
    env.update(CUDA_VISIBLE_DEVICES=uuid,CFM_STANDALONE_RUN_ID=name,
               PACLOCK_PROC='/data2/zz5070/CFM4EEG-data',
               PYTHONPATH=str(ROOT)+':/data2/zz5070/CFM4EEG-runtime')
    save(gpu=gpu,gpu_uuid=uuid)
try:
    for phase,command,timeout in [
        ('contract',[sys.executable,'-u','smoke/verify_sean.py','--config',str(cfgpath),'--output',str(res/'contract.json')],1200),
        ('timing',([sys.executable,'-u','smoke/verify_range_hosts.py','--config',str(cfgpath)] if cfg['model'].endswith('_quad_range') else [sys.executable,'-u','smoke/smoke_amd_partition.py','--config',str(cfgpath),'--output',str(res/'timing.json')]),1800),
        ('training',[sys.executable,'-u','-m','paclock_bench.training.train','--config',str(cfgpath),'--seed',str(seed)],None)]:
        save(phase=phase)
        child=subprocess.Popen(command,env=env,start_new_session=True);save(child_pid=child.pid)
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM);raise
        if rc:raise RuntimeError(f'{phase} returned {rc}')
    result=json.loads((out/'result.json').read_text())
    assert result['stopped_by'] in ['epochs','patience'],result['stopped_by']
    assert (out/'best.pt').is_file()
    save(phase='finished',test=result['test'],checkpoint=str(out/'best.pt'))
except Exception as e:save(phase='failed',error=repr(e));raise
