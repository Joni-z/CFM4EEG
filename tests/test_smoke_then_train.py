"""Exercise the real shell wrapper without allocating GPUs or running a model."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SmokeThenTrainTests(unittest.TestCase):
    def execute(self, seed_line, existing_seed=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('slurm', 'scripts/slurm', 'smoke'):
                (root / directory).mkdir(parents=True)
            (root / 'slurm/amd_env.sh').write_text('export PACLOCK_PYTHON=' + shlex.quote(sys.executable) + '\n')
            shutil.copy(ROOT / 'scripts/slurm/run_packed.py', root / 'scripts/slurm/run_packed.py')
            (root / 'slurm/configs_packed.slurm').write_text('printf "%s\\n" "$@" > train_args\n')
            (root / 'smoke/smoke_amd_partition.py').write_text(
                'import pathlib,sys,yaml\n'
                'cfg=yaml.safe_load(open(sys.argv[sys.argv.index("--config")+1]))\n'
                'pathlib.Path("smoke_seed").write_text(str(cfg.get("seed",0)))\n')
            (root / 'config with space.yaml').write_text('name: example\n' + seed_line)
            if existing_seed is not None:
                old = root / f'runs/example/seed{existing_seed}'
                old.mkdir(parents=True); (old / 'result.json').write_text('preserved')
            env = dict(os.environ, SLURM_SUBMIT_DIR=str(root), SLURM_JOB_PARTITION='mi2101x', SLURM_JOB_ID='test')
            out = subprocess.run(['bash', str(ROOT / 'slurm/smoke_then_train.slurm'), 'config with space.yaml'],
                                 env=env, capture_output=True, text=True)
            if existing_seed is not None:
                self.assertEqual((old / 'result.json').read_text(), 'preserved')
            return out, {p.name: p.read_text() for p in (root / 'smoke_seed', root / 'train_args') if p.exists()}

    def test_seed_one_does_not_target_existing_seed_zero(self):
        out, artifacts = self.execute('seed: 1\n', existing_seed=0)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)['runs'][0]['seed'], 1)
        self.assertEqual(artifacts, {'smoke_seed': '1', 'train_args': 'config with space.yaml:1\n'})

    def test_missing_seed_preserves_default_zero(self):
        out, artifacts = self.execute('')
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(artifacts['smoke_seed'], '0')
        self.assertEqual(artifacts['train_args'], 'config with space.yaml:0\n')

    def test_invalid_seed_or_existing_target_stops_before_smoke(self):
        for seed, existing in [('seed: -1\n', None), ('seed: true\n', None), ('seed: bad\n', None), ('seed: 1\n', 1)]:
            out, artifacts = self.execute(seed, existing)
            self.assertNotEqual(out.returncode, 0)
            self.assertEqual(artifacts, {})


if __name__ == '__main__':
    unittest.main()
