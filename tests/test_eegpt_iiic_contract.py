"""IIIC is an explicit six-class adaptation, not a renamed CHB/TUAB dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.rebuild.run_eegpt_chb_official import verify_iiic_contract, verify_iiic_split


class IIICContractTests(unittest.TestCase):
    def fixture(self, root):
        splits = {}
        for name, count in [('train', 73637), ('val', 15778), ('test', 15780)]:
            counts = [count // 6] * 6
            counts[-1] += count - sum(counts)
            splits[name] = dict(n_windows=count, shape=[16, 2000],
                                class_counts={str(k): v for k, v in enumerate(counts)})
        manifest = dict(dataset='iiic', protocol=dict(sample_rate=200,
                        normalization='div100', num_classes=6), splits=splits)
        accepted = dict(status='accepted', dataset='iiic', files={})
        for split in splits:
            for kind in ['signals', 'labels']:
                path = root/(split+'_'+kind+'.npy')
                path.write_bytes(b'fixture metadata only: do not load payload')
                stat = path.stat()
                accepted['files'][path.name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        cfg = dict(pretrained=False, native_task='tuab', seed=0, num_classes=6,
                   epochs=50, batch_size=100, precision='bf16', lr=0.0005,
                   weight_decay=0.05, select_metric='accuracy', loader_signal_mode='uv_div100')
        self.write_metadata(root, manifest, accepted, cfg)
        return manifest, accepted, cfg

    def write_metadata(self, root, manifest, accepted, cfg):
        path = root/'manifest.json'
        path.write_text(json.dumps(manifest))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        accepted['manifest_sha256'] = cfg['expected_manifest_sha256'] = digest
        (root/'ACCEPTED.json').write_text(json.dumps(accepted))

    def test_acceptance_uses_metadata_only_and_does_not_require_chb_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, cfg = self.fixture(root)
            self.assertFalse((root/'READY.json').exists())
            with patch('numpy.load', side_effect=AssertionError('No test payload may be opened')):
                self.assertEqual(verify_iiic_contract(root, cfg), manifest)
            with open(root/'test_signals.npy', 'ab') as stream:
                stream.write(b'x')
            with self.assertRaisesRegex(RuntimeError, 'Accepted IIIC file changed'):
                verify_iiic_contract(root, cfg)

    def test_wrong_units_or_recipe_cannot_silently_enter_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, accepted, cfg = self.fixture(root)
            bad = dict(cfg, num_classes=2)
            with self.assertRaisesRegex(ValueError, 'num_classes'):
                verify_iiic_contract(root, bad)
            manifest['protocol']['normalization'] = 'q95'
            self.write_metadata(root, manifest, accepted, cfg)
            with self.assertRaisesRegex(RuntimeError, 'microvolts/100'):
                verify_iiic_contract(root, cfg)

    def test_opened_split_rejects_illegal_label_and_cohort_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = self.fixture(root)
            meta = manifest['splits']['val']
            labels = np.repeat(np.arange(6), list(meta['class_counts'].values()))
            dataset = SimpleNamespace(labels=labels,
                signals=SimpleNamespace(shape=(len(labels), 16, 2000), dtype=np.dtype('float32')))
            verify_iiic_split(dataset, 'val', manifest)
            dataset.labels[0] = 6
            with self.assertRaisesRegex(RuntimeError, 'layout/labels'):
                verify_iiic_split(dataset, 'val', manifest)
