"""IIIC uses its released raw-array transform, not the five-second TUH loader."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('biot_iiic_contract', ROOT/'scripts/rebuild/native_biot_entry.py')
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


@pytest.fixture
def native_utils():
    vendor = Path(os.environ.get('BIOT_TEST_VENDOR', ROOT/'vendor/biot'))
    if not vendor.is_dir():
        pytest.skip('Set BIOT_TEST_VENDOR to the pinned original BIOT checkout')
    path = vendor/'utils.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BRIDGE.SOURCE_SHA['utils.py']
    spec = importlib.util.spec_from_file_location('pinned_biot_iiic_utils_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def arrays(tmp_path):
    time = np.arange(2000)/200.
    physical = np.stack([np.stack([(ch+1)*np.sin(2*np.pi*8*time + i*.13)
                                  for ch in range(16)]) for i in range(6)]).astype(np.float32)
    stored = physical/100.
    y = np.arange(6, dtype=np.int64)
    for split in ['train','val','test']:
        np.save(tmp_path/(split+'_signals.npy'), stored)
        np.save(tmp_path/(split+'_labels.npy'), y)
    return tmp_path, stored, y


def test_native_iiic_transform_units_labels_and_spectrum(arrays, native_utils, monkeypatch):
    root, stored, labels = arrays
    # A TUEV resampling regression must fail rather than silently halve duration.
    monkeypatch.setattr(native_utils, 'resample', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('IIIC must not resample')))
    bridge = BRIDGE.NpyPickleBridge(root, 'IIIC')
    dataset = bridge.native_dataset(native_utils, 'train')
    assert len(dataset) == len(labels)
    for i in range(6):
        x, y = dataset[i]
        physical = torch.FloatTensor(stored[i]*100.)
        expected = physical/(torch.quantile(physical.abs(),.95,dim=-1,keepdim=True)+1e-8)
        assert x.shape == (16,2000) and x.dtype == torch.float32 and y == i
        torch.testing.assert_close(x, expected, rtol=0, atol=0)
        peak = np.argmax(np.abs(np.fft.rfft(x[0].numpy()))) * 200 / 2000
        assert peak == 8., '2000 samples must retain their original ten-second time scale'
    subset = bridge.native_dataset(native_utils, 'val', ['5','0','3'])
    assert [subset[i][1] for i in range(len(subset))] == [5,0,3]


def test_smoke_never_opens_test_arrays(arrays, native_utils, monkeypatch):
    root, _, _ = arrays
    original = np.load
    accessed = []
    def guarded(path, *args, **kwargs):
        accessed.append(Path(path).name)
        assert not Path(path).name.startswith('test_')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np,'load',guarded)
    bridge = BRIDGE.NpyPickleBridge(root,'IIIC',smoke=True)
    assert set(bridge.arrays) == {'train','val'}
    assert bridge.native_dataset(native_utils,'val')[5][1] == 5
    assert accessed == ['train_signals.npy','train_labels.npy','val_signals.npy','val_labels.npy']


def test_iiic_acceptance_without_invented_channel_names(arrays):
    root, _, _ = arrays
    manifest={'dataset':'iiic','protocol':{'sample_rate':200,'normalization':'div100'}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    files={f.name: {'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in root.glob('*.npy')}
    receipt={'status':'accepted','manifest_sha256':BRIDGE.sha(root/'manifest.json'),'files':files}
    (root/'ACCEPTED.json').write_text(json.dumps(receipt))
    assert BRIDGE.verify_accepted(root) == manifest


def test_iiic_reviewed_native_args():
    cfg=yaml.safe_load((ROOT/'configs/baselines/biot/iiic-scratch.yaml').read_text())
    args=BRIDGE.native_args(cfg)
    assert (args.dataset,args.in_channels,args.n_classes,args.sample_length,args.batch_size) == ('IIIC',16,6,10,128)
    assert args.pretrain_model_path == ''
    assert cfg['expected_manifest_sha256'] == '3c9d3e148c6cb3f07d4e7861d321bf5a47bbba9327897850b265b6f2d71ab715'
    assert cfg['expected_split_sizes'] == {'train':73637,'val':15778,'test':15780}
    bad=dict(cfg,loader_input_multiplier=1e-4)
    with pytest.raises(ValueError,match='input units'):
        BRIDGE.native_args(bad)
    bad=dict(cfg,model='ffcl')
    with pytest.raises(ValueError,match='scratch BIOT'):
        BRIDGE.native_args(bad)


def test_iiic_rejects_shifted_labels(arrays):
    root, _, y = arrays
    np.save(root/'train_labels.npy',y+1)
    with pytest.raises(ValueError,match='labels'):
        BRIDGE.NpyPickleBridge(root,'IIIC',smoke=True)
