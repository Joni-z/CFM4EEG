"""Released-weight identity and channel capacity must not silently fall back."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('biot_pretrained_contract', ROOT/'scripts/rebuild/native_biot_entry.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(dataset='tuev'):
    return yaml.safe_load((ROOT/'configs/baselines/biot'/f'{dataset}-scratch.yaml').read_text())


def pretrained(tmp_path, monkeypatch):
    cfg = config()
    weight = tmp_path/'released.ckpt'
    weight.write_bytes(b'released checkpoint identity fixture')
    monkeypatch.setenv('BIOT_TEST_WEIGHT', str(weight))
    cfg.update(pretrained=True, checkpoint='$BIOT_TEST_WEIGHT',
               checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
    cfg['model_kwargs']['encoder_channels'] = 18
    return cfg, weight


def test_scratch_keeps_original_capacity_and_no_weights():
    for dataset in ['tuev', 'chbmit']:
        args = MODULE.native_args(config(dataset))
        assert args.in_channels == 16 and args.pretrain_model_path == ''


def test_pretrained_capacity_matches_released_constructor(tmp_path, monkeypatch):
    cfg, path = pretrained(tmp_path, monkeypatch)
    args = MODULE.native_args(cfg)
    assert args.in_channels == 18 and args.pretrain_model_path == str(path.resolve())


def test_missing_or_modified_pretrained_weight_is_rejected(tmp_path, monkeypatch):
    cfg, weight = pretrained(tmp_path, monkeypatch)
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        MODULE.native_args(cfg)
    weight.unlink()
    with pytest.raises(ValueError, match='missing'):
        MODULE.native_args(cfg)


def test_scratch_cannot_load_pretrained_and_capacity_cannot_shrink(tmp_path, monkeypatch):
    cfg, _ = pretrained(tmp_path, monkeypatch)
    bad = copy.deepcopy(cfg)
    bad['pretrained'] = False
    with pytest.raises(ValueError, match='scratch forbids'):
        MODULE.native_args(bad)
    cfg['model_kwargs']['encoder_channels'] = 16
    with pytest.raises(ValueError, match='capacity'):
        MODULE.native_args(cfg)
