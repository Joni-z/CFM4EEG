"""Released encoder coverage is mandatory; task heads must not be replaced."""
import hashlib

import unittest
import torch
from torch import nn

from scripts.rebuild.eegpt_pretrained import load_released_pretraining


class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_encoder = nn.Linear(2, 2)
        self.reconstructor = nn.Module()
        self.reconstructor.proj = nn.Linear(2, 2)
        self.reconstructor.cls_token = nn.Parameter(torch.randn(1, 2))
        self.head = nn.Linear(2, 1)


def checkpoint(tmp_path, state):
    p = tmp_path / 'released.ckpt'
    torch.save({'state_dict': state}, p)
    return {'pretrained_path': str(p), 'pretrained_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def check_complete_encoder_load_keeps_task_head_and_reports_unused_token(tmp_path):
    source, target = Task(), Task()
    head = target.head.weight.detach().clone()
    state = {k: v for k, v in source.state_dict().items()
             if k.startswith(('target_encoder.', 'reconstructor.')) and k != 'reconstructor.cls_token'}
    cfg = checkpoint(tmp_path, state)
    proof = load_released_pretraining(target, cfg)
    assert proof['strict_target_encoder'] and proof['loaded_tensor_equality']
    assert torch.equal(target.target_encoder.weight, source.target_encoder.weight)
    assert torch.equal(target.head.weight, head)


def check_partial_or_corrupted_pretraining_cannot_fall_back_to_random(tmp_path):
    source, target = Task(), Task()
    state = {k: v for k, v in source.state_dict().items() if k.startswith('target_encoder.')}
    del state['target_encoder.bias']
    cfg = checkpoint(tmp_path, state)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'complete target encoder'):
        load_released_pretraining(target, cfg)
    cfg['pretrained_sha256'] = '0' * 64
    with unittest.TestCase().assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        load_released_pretraining(target, cfg)


class ReleasedEEGPTTests(unittest.TestCase):
    def test_native_positional_autocast_decorator_accepts_bf16(self):
        from unittest.mock import patch
        from scripts.rebuild.eegpt_pretrained import configure_native_autocast
        with patch('torch.cuda.amp.autocast', side_effect=lambda **kw: kw):
            configure_native_autocast('bf16')
            result = torch.cuda.amp.autocast(True)
            self.assertTrue(result['enabled'])
            self.assertEqual(result['dtype'], torch.bfloat16)

    def test_shared_acceptance_binds_metadata_without_opening_test_arrays(self):
        import tempfile
        import json
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        from scripts.rebuild.run_eegpt_tuev_official import verify_shared_tuev
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'dataset': 'tuev', 'protocol': {'sample_rate': 200,
                        'normalization': 'div100', 'channels': list(range(16)), 'num_classes': 6}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            files = {}
            for split in ['train', 'val', 'test']:
                for kind in ['signals', 'labels']:
                    path = root/(split+'_'+kind+'.npy')
                    np.save(path, np.zeros((2, 16, 1000), np.float32) if kind == 'signals'
                            else np.zeros(2, np.int64))
                    st = path.stat()
                    files[path.name] = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
            accepted = {'status': 'accepted', 'dataset': 'tuev', 'files': files,
                        'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
            with patch('numpy.load', side_effect=AssertionError('Metadata guard must not open held-out arrays')):
                self.assertEqual(verify_shared_tuev(root, accepted), manifest)
            with open(root/'test_signals.npy', 'ab') as stream:
                stream.write(b'x')
            with self.assertRaisesRegex(RuntimeError, 'Accepted shared16 file changed'):
                verify_shared_tuev(root, accepted)

    def test_complete_encoder(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            check_complete_encoder_load_keeps_task_head_and_reports_unused_token(Path(tmp))

    def test_corruption_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            check_partial_or_corrupted_pretraining_cannot_fall_back_to_random(Path(tmp))


if __name__ == '__main__':
    unittest.main()
