"""Run one authorized Quad16 seed on an observed vacant shared GPU."""
import fcntl, json, os, pathlib, signal, subprocess, sys, time
import yaml

ROOT = pathlib.Path('/data2/zz5070/CFM4EEG-quad16-breadth-20260915')
DATA = pathlib.Path('/data2/zz5070/CFM4EEG-data/processed')
PY = '/data2/zz5070/miniconda3/envs/py312/bin/python'
ds, seed = sys.argv[1], int(sys.argv[2])
assert ds in ['chbmit', 'sleepedf', 'tuar', 'siena'] and seed in [1, 2]
os.chdir(ROOT)
cfgpath = ROOT / f'configs/quad16_seeds/{ds}_s{seed}.yaml'
cfg = yaml.safe_load(cfgpath.read_text())
assert cfg['seed'] == seed and cfg['dataset'] == ds
out = ROOT / 'runs' / cfg['name'] / f'seed{seed}'
receipt = ROOT / f'results/shared145-{ds}-s{seed}.json'
locks = ROOT / 'results/shared-gpu-locks'
locks.mkdir(exist_ok=True)
lock = open(locks / f'run-{ds}-s{seed}.lock', 'a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit('Already queued/running')
state = dict(dataset=ds, seed=seed, pid=os.getpid(), phase='waiting_data')
def save(**kw):
    state.update(kw, heartbeat=time.time())
    temp = receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(receipt)

child = None
def stop(sig, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
    save(phase='operator_stopped')
    raise SystemExit(128 + sig)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def vacant():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    busy = set(apps.splitlines())
    answer = {}
    for row in rows.splitlines():
        idx, uuid, memory, util = [x.strip() for x in row.split(',')]
        if uuid not in busy and float(memory) < 100 and float(util) == 0:
            answer[int(idx)] = uuid
    return answer

try:
    if (out / 'result.json').exists():
        save(phase='already_complete'); sys.exit(0)
    deadline = time.time() + 48 * 3600
    while not (DATA / ds / '.cfm-sha256-verified').exists():
        save(phase='waiting_data')
        if time.time() > deadline:
            raise TimeoutError('Data readiness timeout')
        time.sleep(20)
    gpu_lock = None
    while gpu_lock is None:
        save(phase='waiting_vacant_gpu')
        for gpu, uuid in vacant().items():
            f = open(locks / f'gpu-{gpu}.lock', 'a')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close(); continue
            time.sleep(3)
            if vacant().get(gpu) == uuid:
                gpu_lock = f
                break
            f.close()
        if gpu_lock is None:
            if time.time() > deadline:
                raise TimeoutError('GPU admission timeout')
            time.sleep(20)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CFM_STANDALONE_RUN_ID=f'145-{ds}-s{seed}',
               PACLOCK_PROC=str(DATA.parent), PYTHONPATH='/data2/zz5070/CFM4EEG-runtime',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    save(phase='smoking', gpu=gpu, gpu_uuid=uuid, config=str(cfgpath))
    commands = [
        ['timeout', '--kill-after=60s', '30m', PY, '-u', 'smoke/verify_modulation_carrier.py'],
        ['timeout', '--kill-after=60s', '30m', PY, '-u', 'smoke/smoke_amd_partition.py', '--config', str(cfgpath), '--output', f'results/shared145-smoke-{ds}-s{seed}.json'],
        ['timeout', '--signal=TERM', '--kill-after=300s', '24h' if ds == 'chbmit' else '10h',
         PY, '-u', '-m', 'paclock_bench.training.train', '--config', str(cfgpath), '--seed', str(seed)]
    ]
    for i, cmd in enumerate(commands):
        save(phase='training' if i == 2 else 'smoking', stage=i)
        child = subprocess.Popen(cmd, env=env, start_new_session=True)
        save(child_pid=child.pid)
        rc = child.wait()
        if rc:
            raise RuntimeError(f'stage {i} exited {rc}')
    assert (out / 'result.json').exists(), 'No result despite normal exit'
    save(phase='finished', result=str(out / 'result.json'))
except Exception as e:
    save(phase='failed', error=repr(e))
    raise
