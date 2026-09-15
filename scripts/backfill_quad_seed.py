"""Run one queued seed in a completed packed slot without releasing its node early."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

cfg, seed, gpu = sys.argv[1:]
job = os.environ['SLURM_JOB_ID']
assert os.environ.get('SLURM_STEP_ID') not in (None, 'batch')
root = Path(os.environ['SLURM_SUBMIT_DIR'])
os.chdir(root)
assert Path(cfg).is_file() and seed in ('0', '1', '2') and gpu in ('0', '1', '2', '3')
receipt = root / 'results' / ('backfill-' + job + '.json')
lock = open(str(receipt) + '.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not receipt.exists(), 'Backfill already attempted; inspect receipt before retry'
state = dict(job=job, config=cfg, seed=int(seed), hip_gpu=int(gpu), pid=os.getpid())
def save(**kw):
    state.update(kw, time=time.time())
    temp = receipt.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(receipt)

supervisors = []
needle = ('/var/spool/slurmd/job' + job + '/slurm_script').encode()
for p in Path('/proc').glob('[0-9]*'):
    try:
        if p.stat().st_uid == os.getuid() and needle in (p / 'cmdline').read_bytes().split(b'\0'):
            supervisors.append(int(p.name))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
assert len(supervisors) == 1, supervisors
supervisor = supervisors[0]
child = None
paused = False
def terminate(sig, frame):
    raise SystemExit(128 + sig)
signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
try:
    os.kill(supervisor, signal.SIGSTOP)
    paused = True
    save(phase='checking_slot', supervisor=supervisor)
    env = os.environ.copy()
    env['QUAD_GPU'] = gpu
    env['HIP_VISIBLE_DEVICES'] = gpu
    # A separate process releases its probe context before smoke/training.
    probe = 'import torch; assert torch.cuda.device_count()==1; f,t=torch.cuda.mem_get_info(); print("slot_free_bytes",f,"total",t,flush=True); assert t-f < 1024**3, "GPU already occupied"'
    subprocess.run([sys.executable, '-c', probe], env=env, check=True, timeout=120)
    save(phase='smoke_then_train')
    child = subprocess.Popen(['bash', 'slurm/quad16_seed_amd.slurm', cfg, seed], env=env, start_new_session=True)
    save(child_pid=child.pid)
    # Current allocations have >21 hours left; leave time for orderly shutdown.
    rc = child.wait(timeout=21 * 3600)
    save(phase='finished' if rc == 0 else 'failed', returncode=rc)
    if rc:
        raise SystemExit(rc)
except BaseException as exc:
    save(phase='failed', error=repr(exc))
    raise
finally:
    try:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            except ProcessLookupError:
                pass
    finally:
        if paused:
            try:
                os.kill(supervisor, signal.SIGCONT)
            except ProcessLookupError:
                pass
