"""Reject wrong cohorts or changed cached payloads before native GPU training."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "labram_entry", Path(__file__).parents[1] / "scripts/rebuild/run_labram_original.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class SharedReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "manifest.json").write_text(json.dumps({"dataset": "iiic"}))
        files = {}
        for split in ["train", "val", "test"]:
            for kind in ["signals", "labels"]:
                p = self.root / f"{split}_{kind}.npy"
                p.write_bytes(b"fixture-payload")
                s = p.stat()
                files[p.name] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns,
                                     sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        self.digest = entry.sha(self.root / "manifest.json")
        self.accepted = dict(status="accepted", dataset="iiic",
                             manifest_sha256=self.digest, files=files)

    def verify(self, dataset="iiic", digest=None):
        entry.verify_shared_receipt(self.root, self.accepted, dataset, digest or self.digest)

    def test_complete_receipt_passes_but_missing_test_metadata_fails(self):
        self.verify()
        self.accepted["files"].pop("test_labels.npy")
        with self.assertRaisesRegex(ValueError, "all six"):
            self.verify()

    def test_wrong_dataset_or_manifest_fails(self):
        with self.assertRaisesRegex(ValueError, "Wrong accepted"):
            self.verify(dataset="tuev")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.verify(digest="0" * 64)
        (self.root / "manifest.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            self.verify()

    def test_changed_payload_stat_fails_without_reading_test_payload(self):
        (self.root / "test_signals.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "test_signals"):
            self.verify()


if __name__ == '__main__':
    unittest.main()
