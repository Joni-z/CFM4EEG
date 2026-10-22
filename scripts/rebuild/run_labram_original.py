"""Invoke the pinned LaBraM main/engine directly; scratch-only data bridges.

TUEV uses the independently accepted original23 pickle cohort. CHB-MIT is
explicitly a new-task adaptation: ordinal trainable channel slots have no
electrode identity, and stored q95 normalization is inverted before the
unchanged original engine divides physical microvolts by100.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

COMMIT = "c431221e6cfd23dbfa9950e0180682fb322b0548"
SOURCE = {
    "run_class_finetuning.py": "5aeeb1290ed5a1e2be0ba7244181864d86ebd8d3f67489c46f27979b50fd57dc",
    "engine_for_finetuning.py": "7e0e8dad3f208645ac3c966ef356433b84def3f371852daf2858778562726f92",
    "utils.py": "c5ad729bd57ab31e04572f426d94f7fb164ef0fbfd1f7076da6bf8dfbd833e32",
    "modeling_finetune.py": "7b3514c9a661ebd417a2dd9834a9f2fab0a2e518c8212c73ebebaaa165c29d88",
    "optim_factory.py": "7f82eeef697f6abc5a1c76e5e6be0087a42c95656d89a6b7d46b8216d246e1ab",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--vendor", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-proof", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if cfg["seed"] != 0 or cfg.get("pretrained") is not False:
        raise ValueError("Only scratch seed0 is authorized")
    root = Path(os.path.expandvars(os.path.expanduser(cfg["data_root"])))
    accepted = json.loads((root / "ACCEPTED.json").read_text())
    if accepted.get("status") != "accepted":
        raise ValueError("Unaccepted dataset")
    for name, expected in SOURCE.items():
        if sha(a.vendor / name) != expected:
            raise ValueError("Pinned source drift: " + name)
    identity = dict(config_sha256=sha(a.config), runner_sha256=sha(__file__),
                    accepted_sha256=sha(root / "ACCEPTED.json"), source_sha256=SOURCE,
                    vendor_commit=COMMIT)
    if not a.smoke:
        if a.smoke_proof is None:
            raise ValueError("Formal run requires successful smoke proof")
        proof = json.loads(a.smoke_proof.read_text())
        if proof.get("status") != "passed" or proof["identity"] != identity:
            raise ValueError("Smoke proof does not match this exact run")
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Preserving existing output " + str(a.out))
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.vendor))
    import numpy as np
    import torch
    from torch.utils.data import Dataset, Subset
    import run_class_finetuning as native
    import utils
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one physical GPU; official update_freq preserves global batch")
    if cfg["dataset"] == "TUEV":
        transfer = json.loads((root / "TRANSFER_VERIFIED.json").read_text())
        if transfer["accepted_sha256"] != identity["accepted_sha256"]:
            raise ValueError("Native transfer acceptance mismatch")
        original = utils.prepare_TUEV_dataset
        utils.prepare_TUEV_dataset = lambda ignored: original(str(root / "native/processed"))
        original_get = native.get_dataset
        def get_dataset(args):
            tr, te, va, names, metrics = original_get(args)
            if a.smoke:
                tr, va = Subset(tr, range(256)), Subset(va, range(96))
                te = va  # no held-out test payload is read by the smoke
            return tr, te, va, names, metrics
    elif cfg["dataset"] in ("CHBMIT", "ADFD"):
        manifest = json.loads((root / "manifest.json").read_text())
        expected_manifest = accepted["manifest_sha256"]
        is_chb = cfg["dataset"] == "CHBMIT"
        ready = json.loads((root / "READY.json").read_text()) if is_chb else {"manifest_sha256": expected_manifest}
        if sha(root / "manifest.json") != expected_manifest or ready["manifest_sha256"] != expected_manifest:
            raise ValueError("Manifest/build completion mismatch")
        norm = "q95" if is_chb else "div100"
        n_channels, n_classes = (16, 2) if is_chb else (19, 3)
        if manifest["protocol"]["sample_rate"] != 200 or manifest["protocol"]["normalization"] != norm:
            raise ValueError("Unexpected stored units or sample rate")
        class SharedCHB(Dataset):
            def __init__(self, split):
                self.x = np.load(root / (split + "_signals.npy"), mmap_mode="r")
                self.y = np.load(root / (split + "_labels.npy"), mmap_mode="r")
                self.scale = np.load(root / (split + "_q95_denominators.npy"), mmap_mode="r") if is_chb else None
                if self.x.shape[1:] != (n_channels, 2000) or len(self.y) != len(self.x):
                    raise ValueError("Array shape mismatch")
                if is_chb and self.scale.shape != (len(self.x), n_channels, 1):
                    raise ValueError("CHB scale mismatch")
            def __len__(self):
                return len(self.y)
            def __getitem__(self, index):
                uv = (self.x[index].astype(np.float64) * (self.scale[index] if is_chb else 100.)).astype(np.float32)
                if not np.isfinite(uv).all():
                    raise ValueError("Nonfinite recovered microvolts")
                return torch.from_numpy(uv), int(self.y[index])
        original_chans = utils.get_input_chans
        names = ["SCRATCH_CHANNEL_SLOT_" + str(i) for i in range(n_channels)]
        def slots(ch_names):
            return list(range(n_channels + 1)) if ch_names == names else original_chans(ch_names)
        utils.get_input_chans = slots
        def get_dataset(args):
            args.nb_classes = 1 if is_chb else 3
            tr, va = SharedCHB("train"), SharedCHB("val")
            if a.smoke:
                # Include both classes in validation to exercise native binary metrics.
                ids = [int(np.flatnonzero(va.y == label)[0]) for label in range(n_classes)]
                va = Subset(va, ids * (96 // n_classes))
                tr = Subset(tr, range(256))
                te = va
            else:
                te = SharedCHB("test")
            return tr, te, va, names, (["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"] if is_chb else ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"])
    else:
        raise ValueError("Unsupported native bridge")
    native.get_dataset = get_dataset
    argv = ["--dataset", "TUEV" if cfg["dataset"] == "TUEV" else "TUAB",
            "--output_dir", str(a.out), "--model", "labram_base_patch200_200",
            "--seed", "0", "--batch_size", str(cfg["batch_size"]),
            "--update_freq", "1" if a.smoke else str(cfg["update_freq"]),
            "--epochs", "1" if a.smoke else str(cfg["epochs"]),
            "--warmup_epochs", "0" if a.smoke else str(cfg["warmup_epochs"]),
            "--lr", str(cfg["lr"]), "--min_lr", "1e-6",
            "--layer_decay", str(cfg["layer_decay"]), "--weight_decay", "0.05",
            "--num_workers", "0" if a.smoke else str(cfg["num_workers"]),
            "--save_ckpt_freq", "5", "--drop_path", "0.1", "--smoothing", "0.1",
            "--disable_rel_pos_bias", "--abs_pos_emb", "--disable_qkv_bias", "--no_auto_resume"]
    sys.argv = ["run_class_finetuning.py"] + argv
    args, ds_init = native.get_args()
    if args.finetune or args.resume or ds_init:
        raise ValueError("External weights/DeepSpeed are not authorized")
    captured = {}
    original_models = native.get_models
    def capture(args):
        m = original_models(args)
        captured["model"] = m
        captured["head_before"] = m.head.weight.detach().cpu().clone()
        return m
    native.get_models = capture
    (a.out / "provenance.json").write_text(json.dumps(dict(identity=identity, config=cfg,
        args=vars(args), recipe_scope=cfg["recipe_scope"], input_scope=cfg["input_scope"],
        smoke=a.smoke, test_payload_read=not a.smoke), indent=2))
    native.main(args, ds_init)
    model = captured["model"]
    ckpt = torch.load(a.out / "checkpoint.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
        raise RuntimeError("Nonfinite trained state")
    if torch.equal(captured["head_before"], model.head.weight.detach().cpu()):
        raise RuntimeError("Native optimizer made no classifier update")
    proof = dict(status="passed", identity=identity, checkpoint_strict_reload=True,
                 classifier_updated=True, finite_model=True, smoke=a.smoke)
    (a.out / ("smoke_pass.json" if a.smoke else "native_completed.json")).write_text(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
