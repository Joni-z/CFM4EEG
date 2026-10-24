"""An unfinished baseline must never silently run the project's trainer."""
import os
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from paclock_bench.models.build import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('extra,message', [
    ({}, 'generic baseline training is disabled'),
    ({'disabled_reason': 'Unverified author entrypoint'}, 'Unverified author entrypoint'),
])
def test_unverified_recipe_cannot_reach_training(tmp_path, extra, message):
    cfg = dict(name='not-a-certified-run', model='biot', dataset='tuev',
               data_root=str(tmp_path / 'absent-data'), **extra)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / 'outputs'
    env = dict(os.environ, CFM_REBUILD_ROOT=str(output))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/rebuild/launch.py'),
                             str(path), '0', 'train'], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


def test_project_factory_cannot_construct_a_baseline():
    with pytest.raises(ValueError, match='original entrypoint'):
        build_model({'model': 'cbramod'}, (16, 1000))


@pytest.mark.parametrize('visible', [0, 1])
def test_two_rank_eegpt_refuses_insufficient_gpus(tmp_path, monkeypatch, visible):
    import torch

    spec = importlib.util.spec_from_file_location('native_launch', ROOT / 'scripts/rebuild/launch.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump({
        'runner': 'scripts/rebuild/run_eegpt_chb_official.py', 'name': 'resource-check',
    }))
    monkeypatch.setattr(sys, 'argv', ['launch.py', str(cfg), '0', 'train'])
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: visible)

    def forbidden(*args, **kwargs):
        pytest.fail('Insufficient allocation must not start any subprocess')

    monkeypatch.setattr(launcher.subprocess, 'run', forbidden)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_package_source_is_not_hidden_by_output_ignore_rules():
    if not (ROOT / '.git').exists():
        pytest.skip('Git ignore rules require checkout metadata')
    result = subprocess.run(['git', 'check-ignore', '--no-index',
                             'paclock_bench/data/contracts.py'], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 1, result.stdout + result.stderr
