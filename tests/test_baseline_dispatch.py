"""An unfinished baseline must never silently run the project's trainer."""
import os
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
