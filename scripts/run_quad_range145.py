"""One authorized Quad-Range run; lock and recheck a vacant shared GPU."""
import fcntl, json, os, pathlib, shutil, signal, subprocess, sys, time
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path('/data2/zz5070/CFM4EEG-data/processed')
LOCKS = pathlib.Path('/data2/zz5070/CFM4EEG-quad16-breadth-20260915/results/shared-gpu-locks')
PY = sys.executable
ds = sys.argv[1]
assert ds in ['tuar', 'sleepedf', 'adfd']
os.chdir(ROOT)
cfgpath = ROOT / f'configs/quad_range/{ds}.yaml'
cfg = yaml.safe_load(cfgpath.read_text())
seed = cfg['seed']
out = ROOT / 'runs' / cfg['name'] / f'seed{seed}'
receipt = ROOT / f'results/range145-{ds}.json'
receipt.parent.mkdir(parents=True, exist_ok=True)
LOCKS.mkdir(parents=True, exist_ok=True)
lock = open(LOCKS / f'quad-range-{ds}-s{seed}.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (out / 'progress.json').exists(), 'Existing run must not be overwritten'
assert not (out / 'result.json').exists(), 'Already complete'
assert (DATA / ds / '.cfm-sha256-verified').exists(), 'Dataset not verified'
assert shutil.disk_usage(ROOT).free > 5 * 2**30, 'Less than 5 GiB free'
state = dict(dataset=ds, seed=seed, pid=os.getpid())
child = None
def save(**kw):
    state.update(kw, heartbeat=time.time())
    temp = receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(receipt)
def stopped():
    return (ROOT / 'STOP_ALL_RANGE').exists() or (out / 'STOP').exists()
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
    apps = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())
    answer = {}
    for row in rows.splitlines():
        idx, uuid, memory, util = [x.strip() for x in row.split(',')]
        if uuid not in apps and float(memory) < 100 and float(util) == 0:
            answer[int(idx)] = uuid
    return answer
try:
    deadline = time.time() + 12 * 3600
    gpu_lock = None
    while gpu_lock is None:
        if stopped():
            raise RuntimeError('Stop marker present')
        save(phase='waiting_vacant_gpu')
        for gpu, uuid in vacant().items():
            f = open(LOCKS / f'gpu-{gpu}.lock', 'a')
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
                raise TimeoutError('No vacant shared GPU within 12 hours')
            time.sleep(20)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CFM_STANDALONE_RUN_ID=f'145-range-{ds}-s{seed}',
               PACLOCK_PROC=str(DATA.parent), PYTHONPATH='/data2/zz5070/CFM4EEG-runtime',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    save(gpu=gpu, gpu_uuid=uuid, config=str(cfgpath))
    commands = [
        ['timeout', '--kill-after=60s', '20m', PY, '-u', 'smoke/verify_quad_range.py', '--config', str(cfgpath), '--output', f'results/range-contract-{ds}.json'],
        ['timeout', '--kill-after=60s', '30m', PY, '-u', 'smoke/smoke_amd_partition.py', '--config', str(cfgpath), '--require-budget-fit', '--output', f'results/range-smoke-{ds}.json'],
        ['timeout', '--signal=TERM', '--kill-after=300s', str(float(cfg['max_hours'])+2)+'h', PY, '-u', '-m', 'paclock_bench.training.train', '--config', str(cfgpath), '--seed', str(seed)]
    ]
    for i, cmd in enumerate(commands):
        if stopped():
            raise RuntimeError('Stop marker present')
        save(phase='training' if i == 2 else 'smoking', stage=i)
        child = subprocess.Popen(cmd, env=env, start_new_session=True)
        save(child_pid=child.pid)
        rc = child.wait()
        if rc:
            raise RuntimeError(f'stage {i} exited {rc}')
    result = json.loads((out / 'result.json').read_text())
    assert result.get('stopped_by') in [None, 'epochs', 'patience'], 'Not normally completed'
    save(phase='finished', result=str(out / 'result.json'))
except Exception as exc:
    save(phase='failed', error=repr(exc))
    raise
