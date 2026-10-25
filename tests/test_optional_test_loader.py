"""Smoke loaders must work without held-out signals, labels, or scale sidecars."""
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paclock_bench.data.datasets import build_dataloaders


class OptionalTestLoaderTests(unittest.TestCase):
    def fixture(self, root: Path, *, with_test: bool, q95: bool = False) -> dict:
        manifest = dict(dataset='synthetic', protocol=dict(sample_rate=200,
                        normalization='q95' if q95 else 'div100'),
                        splits={s: dict(n_windows=4, shape=[2, 8]) for s in ['train', 'val', 'test']})
        (root/'manifest.json').write_text(json.dumps(manifest))
        for split in ['train', 'val'] + (['test'] if with_test else []):
            signal = np.arange(64, dtype=np.float32).reshape(4, 2, 8) / 10
            if split == 'test':
                signal += 1000
            np.save(root/f'{split}_signals.npy', signal)
            np.save(root/f'{split}_labels.npy', np.array([0, 1, 1, 0], dtype=np.int64))
            if q95:
                np.save(root/f'{split}_q95_denominators.npy', np.full((4, 2, 1), 7.0, dtype=np.float64))
        return dict(data_root=str(root), sample_rate=200, batch_size=2, num_workers=0,
                    loader_signal_mode='uv_div100' if q95 else 'stored')

    def reject_test_access(self, root: Path):
        original_open, original_load = builtins.open, np.load

        def guard(path):
            if isinstance(path, (str, bytes, os.PathLike)):
                p = Path(os.fsdecode(path))
                if p.parent == root and p.name.startswith('test_'):
                    raise AssertionError('Held-out payload was accessed: '+str(p))

        def checked_open(path, *args, **kwargs):
            guard(path)
            return original_open(path, *args, **kwargs)

        def checked_load(path, *args, **kwargs):
            guard(path)
            return original_load(path, *args, **kwargs)

        return patch('builtins.open', checked_open), patch('numpy.load', checked_load)

    def test_false_works_with_missing_test_signals_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root, with_test=False)
            open_guard, load_guard = self.reject_test_access(root)
            with open_guard, load_guard:
                train, val, test, info = build_dataloaders(cfg, include_test=False)
                x, y = next(iter(val))
            self.assertIsNone(test)
            self.assertEqual(tuple(x.shape), (2, 2, 8))
            self.assertEqual(y.tolist(), [0, 1])
            self.assertEqual(info['n_samples'], {'train': 4, 'val': 4})
            self.assertEqual(set(info['class_counts']), {'train', 'val'})
            self.assertIn('test', info['manifest']['splits'])

    def test_false_never_reads_missing_q95_test_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root, with_test=False, q95=True)
            open_guard, load_guard = self.reject_test_access(root)
            with open_guard, load_guard:
                _, val, test, info = build_dataloaders(cfg, include_test=False)
                x, _ = val.dataset[0]
            self.assertIsNone(test)
            self.assertFalse((root/'test_q95_denominators.npy').exists())
            original = np.load(root/'val_signals.npy')[0]
            expected = (original.astype(np.float64) * 7.0 * 0.01).astype(np.float32)
            np.testing.assert_array_equal(x.numpy(), expected)

    def test_default_preserves_a_distinct_real_test_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root, with_test=True)
            _, val, test, info = build_dataloaders(cfg)
            self.assertEqual(test.dataset.split, 'test')
            self.assertEqual(info['n_samples']['test'], 4)
            np.testing.assert_array_equal(test.dataset[0][0].numpy(), np.load(root/'test_signals.npy')[0])
            self.assertFalse(np.array_equal(test.dataset[0][0].numpy(), val.dataset[0][0].numpy()))

    def test_default_still_rejects_missing_held_out_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.fixture(Path(tmp), with_test=False)
            for kwargs in [{}, {'include_test': True}]:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(FileNotFoundError, 'test_signals.npy'):
                        build_dataloaders(cfg, **kwargs)


if __name__ == '__main__':
    unittest.main()
