"""Run pinned CBraMod finetune_main.main; replace only its TUEV data reader.

No generic trainer, model adapter, optimizer, metric or checkpoint-selection code
is called. Shared-cohort input is an explicit departure from the paper's cohort.
The official argparse ``type=bool`` cannot express False; the scoped parse hook
sets that one boolean to the requested scratch/pretrained setting.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

COMMIT = "b9e961003214326972c567eff390e75b0287e32a"
PINNED = {
    "finetune_main.py": "6fb96bbc7a73b487c9fdc1f39e8e0ee10fb36c535b0975f8d255fc85eae717f9",
    "finetune_trainer.py": "077eace4ed4e24ba12c4e1c2a3c542c85a6b00d1255fef912544ac73839fc78a",
    "finetune_evaluator.py": "56646f96447384920216d84aa1189b4a96c2e16ced3b927a360e6bcc19711ad3",
    "datasets/tuev_dataset.py": "da78fb2caa0ddc462e59b172f00f2680009b455fef384c1e3bdf05825f8de475",
    "models/model_for_tuev.py": "29df3cad926996f17771bed3055f57899d61df23ba04533298f006ec04dfc1ab",
    "models/cbramod.py": "caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14",
    "models/criss_cross_transformer.py": "90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c",
    "utils/util.py": "acb07a4c862803a59d219e920a849eba86c968854f4f814b9dad8402aaf252ff",
}
CHANNELS = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4",
            "T4-T6", "T6-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_source(vendor):
    vendor = Path(vendor).resolve()
    for name, expected in PINNED.items():
        if sha256(vendor / name) != expected:
            raise ValueError(f"Pinned CBraMod source changed: {name}")
    return {str(p.relative_to(vendor)): sha256(p)
            for p in sorted(vendor.rglob("*.py")) if ".git" not in p.parts}


def verify_input(root, *, smoke=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    accepted = json.loads((root / "ACCEPTED.json").read_text())
    manifest_hash = sha256(root / "manifest.json")
    if accepted.get("status") != "accepted" or accepted.get("manifest_sha256") != manifest_hash:
        raise ValueError("Missing or stale accepted input receipt")
    protocol = manifest["protocol"]
    if (manifest["dataset"] != "tuev" or protocol["sample_rate"] != 200
            or protocol["normalization"] != "div100" or protocol["channels"] != CHANNELS):
        raise ValueError("Expected accepted TUEV 16-channel bipolar, 200 Hz, microvolts/100")
    shapes = {}
    for split in (("train", "val") if smoke else ("train", "val", "test")):
        for kind in ("signals", "labels"):
            name = f"{split}_{kind}.npy"
            path = root / name
            receipt = accepted["files"][name]
            stat = path.stat()
            if stat.st_size != receipt["size"] or stat.st_mtime_ns != receipt["mtime_ns"]:
                raise ValueError(f"Array changed since full-hash acceptance: {name}")
        x = np.load(root / f"{split}_signals.npy", mmap_mode="r")
        y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        if x.dtype != np.float32 or x.ndim != 3 or x.shape[1:] != (16, 1000):
            raise ValueError(f"Wrong native input layout: {split}: {x.shape}/{x.dtype}")
        if y.shape != (len(x),) or y.dtype.kind not in "iu" or not ((y >= 0) & (y < 6)).all():
            raise ValueError(f"Wrong native label layout/range: {split}")
        if not len(x) or (root / f"{split}_signals.npy").stat().st_size != x.offset + x.nbytes:
            raise ValueError(f"Empty or malformed signal array: {split}")
        # The unchanged official evaluator squeezes singleton batches to scalars.
        if split != "train" and len(x) % 64 == 1:
            raise ValueError(f"Native evaluator singleton-tail bug would affect {split}; do not silently drop rows")
        shapes[split] = list(x.shape)
    return {"manifest_sha256": manifest_hash, "accepted_sha256": sha256(root / "ACCEPTED.json"),
            "shapes": shapes, "units": "microvolts/100; no second division"}


class SmokeStepsFinished(Exception):
    """Intentional smoke-only exit after actual native optimizer updates."""


class StopAfterBatches:
    def __init__(self, loader, steps):
        self.loader, self.steps = loader, steps

    def __len__(self):
        # Native CosineAnnealingLR must retain the full training-loader length.
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.steps):
            yield next(iterator)
        raise SmokeStepsFinished()


class NoTestData:
    def __iter__(self):
        raise AssertionError("A smoke run must never iterate test data")


def make_array_loader(native_dataset, root, *, smoke_steps=0):
    """Preserve native collate and all actual DataLoader options, including workers=0."""
    from torch.utils.data import DataLoader, Subset

    class ArrayDataset(native_dataset.CustomDataset):
        def __init__(self, split):
            self.signals = np.load(Path(root) / f"{split}_signals.npy", mmap_mode="r")
            self.labels = np.load(Path(root) / f"{split}_labels.npy", mmap_mode="r")

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            # Native pickle path divides physical microvolts by100 then collate
            # converts to float32. The accepted array already stores that result.
            return self.signals[index].reshape(16, 5, 200), int(self.labels[index])

    class ArrayLoader:
        def __init__(self, params):
            self.params = params

        def get_data_loader(self):
            result = {}
            for split in (("train", "val") if smoke_steps else ("train", "val", "test")):
                dataset = ArrayDataset(split)
                if smoke_steps and split == "val":
                    # Validation labels only; no test access. Include each
                    # present class so tiny-smoke metrics remain well-defined.
                    indices = []
                    for label in np.unique(dataset.labels):
                        indices.extend(np.flatnonzero(dataset.labels == label)[:10].tolist())
                    dataset = Subset(dataset, indices)
                base = dataset.dataset if isinstance(dataset, Subset) else dataset
                result[split] = DataLoader(dataset, batch_size=self.params.batch_size,
                                           collate_fn=base.collate, shuffle=split == "train")
            if smoke_steps:
                if len(result["train"]) <= smoke_steps:
                    raise ValueError("Smoke requires more native batches than the requested step limit")
                result["train"] = StopAfterBatches(result["train"], smoke_steps)
                result["test"] = NoTestData()
            return result

    return ArrayLoader


def load_native(vendor):
    vendor = Path(vendor).resolve()
    # Upstream uses generic top-level names. Never reuse another vendor's cache.
    for name in ("datasets", "models", "utils", "finetune_trainer", "finetune_evaluator"):
        existing = sys.modules.get(name)
        if existing and (not getattr(existing, "__file__", None)
                         or not Path(existing.__file__).resolve().is_relative_to(vendor)):
            raise RuntimeError(f"Foreign module cached as {name}; start a fresh process")
    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("_cfft_official_cbramod_main", vendor / "finetune_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def native_arguments(argv, pretrained, captured):
    old_argv, old_parse = sys.argv, argparse.ArgumentParser.parse_args

    def parse(parser, *args, **kwargs):
        params = old_parse(parser, *args, **kwargs)
        params.use_pretrained_weights = bool(pretrained)
        captured.update(vars(params))
        return params

    try:
        sys.argv = argv
        argparse.ArgumentParser.parse_args = parse
        yield
    finally:
        sys.argv = old_argv
        argparse.ArgumentParser.parse_args = old_parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--vendor", type=Path, help="Optional explicit pinned source checkout")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", choices=(0,), type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--inspect", action="store_true", help="Validate source/data and print plan without CUDA or training")
    args = parser.parse_args()
    import yaml
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from paclock_bench.paths import vendored, expand
    from paclock_bench.training.recipe_runtime import provenance
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["model"] != "cbramod" or cfg["dataset"] != "tuev":
        raise ValueError("This wrapper has only reviewed the native TUEV clinical entry")
    expected = {"epochs": 50, "batch_size": 64, "lr": 1e-4, "weight_decay": .05,
                "multi_lr": True, "grad_clip": 1., "label_smoothing": .1,
                "num_classes": 6, "amp": False, "train_drop_last": False,
                "select_metric": "cohen_kappa", "patience": 0, "sample_rate": 200}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Config {key} disagrees with pinned native TUEV recipe: {cfg.get(key)} != {value}")
    if cfg.get("optimizer", "").lower() != "adamw" or cfg.get("scheduler") != "cosine_updates":
        raise ValueError("Config optimizer/schedule disagrees with native trainer")
    if cfg["model_kwargs"] != {"dropout": .1, "classifier_hidden_dim": 1000}:
        raise ValueError("Config model kwargs disagree with native TUEV head")
    if cfg.get("seed", 0) != 0 or args.smoke_steps < 0:
        raise ValueError("Only authorized seed0 and nonnegative smoke limits are supported")
    args.vendor = args.vendor or Path(vendored("cbramod"))
    args.data_root = Path(expand(cfg["data_root"]))
    args.setting = "pretrained" if cfg.get("pretrained", True) else "scratch"
    args.pretrained_path = (Path(expand(cfg.get("pretrained_path") or str(args.vendor / "pretrained_weights/pretrained_weights.pth")))
                            if args.setting == "pretrained" else None)
    source = verify_source(args.vendor)
    input_receipt = verify_input(args.data_root, smoke=bool(args.smoke_steps))
    if (args.setting == "pretrained") != (args.pretrained_path is not None):
        raise ValueError("pretrained setting requires exactly one explicit checkpoint; scratch forbids it")
    argv = ["finetune_main.py", "--seed", str(args.seed), "--cuda", str(args.cuda),
            "--downstream_dataset", "TUEV", "--num_of_classes", "6",
            "--datasets_dir", str(args.data_root.resolve()), "--model_dir", str(args.out.resolve())]
    if args.pretrained_path:
        argv += ["--foundation_dir", str(args.pretrained_path.resolve())]
    receipt = {"entry": "finetune_main.main", "vendor_commit": COMMIT, "source_sha256": source,
               "input": input_receipt, "argv": argv, "setting": args.setting,
               "pretrained_sha256": sha256(args.pretrained_path) if args.pretrained_path else None,
               "changes": ["Data reader: accepted arrays replace native pickle storage and cohort.",
                           "Single seed0 requested by user instead of CLI default3407.",
                           "Only use_pretrained_weights boolean is supplied via scoped parser bridge."],
               "checkpoint_rule": "Native strict validation-kappa maximum across50epochs; test after restoring best; save native pth.",
               "resume": "Official trainer has no loop-resume and writes selected checkpoint only after finishing; no resume is claimed."}
    if args.inspect:
        print(json.dumps(receipt, indent=2))
        return
    if args.vendor.resolve() != Path(vendored("cbramod")).resolve():
        raise ValueError("Execution vendor must match the release's provenance-bound vendor/cbramod path")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Output is not empty; native trainer cannot safely resume or overwrite it")
    native = load_native(args.vendor)
    native.tuev_dataset.LoadDataset = make_array_loader(native.tuev_dataset, args.data_root,
                                                        smoke_steps=args.smoke_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    captured = {}
    observed = {}
    if args.smoke_steps:
        import torch
        original_trainer = native.Trainer

        class ObservedTrainer(original_trainer):
            def __init__(self, *trainer_args, **kwargs):
                super().__init__(*trainer_args, **kwargs)
                observed["trainer"] = self
                observed["updates"] = 0
                observed["initial"] = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

                def after_step(optimizer, step_args, step_kwargs):
                    for parameter in self.model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("Nonfinite native smoke gradient")
                    observed["updates"] += 1

                def after_loss(module, inputs, output):
                    if not torch.isfinite(output).all():
                        raise RuntimeError("Nonfinite native smoke loss")

                self.optimizer.register_step_post_hook(after_step)
                self.criterion.register_forward_hook(after_loss)

        native.Trainer = ObservedTrainer
    (args.out / "native_launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with native_arguments(argv, args.setting == "pretrained", captured):
            native.main()
    except SmokeStepsFinished:
        if not args.smoke_steps or observed.get("updates") != args.smoke_steps:
            raise RuntimeError("Unexpected smoke stop or optimizer step count")
        trainer = observed["trainer"]
        if not any(not torch.equal(parameter.detach().cpu(), observed["initial"][name])
                   for name, parameter in trainer.model.named_parameters() if name in observed["initial"]):
            raise RuntimeError("Native smoke did not update model parameters")
        with torch.no_grad():
            metrics = trainer.val_eval.get_metrics_for_multiclass(trainer.model)
            if not all(np.isfinite(float(v)) for v in metrics[:3]):
                raise RuntimeError("Nonfinite smoke validation metric")
            x, _ = next(iter(trainer.data_loader["val"]))
            x = x.cuda()
            before = trainer.model(x).detach().cpu()
            checkpoint = args.out / "native_smoke.pth"
            state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
            torch.save(state, checkpoint)
            next(trainer.model.parameters()).add_(1.)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            trainer.model.load_state_dict(loaded, strict=True)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value.detach().cpu(), state[key], rtol=0, atol=0)
            after = trainer.model(x).detach().cpu()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        smoke = {"status": "passed", "config": cfg, "checkpoint_reload": "passed",
                 "test_evaluated": False, "native_training_steps": observed["updates"],
                 "checkpoint_sha256": sha256(checkpoint), "native_params": captured,
                 "source": receipt, "validation_metrics": {"balanced_acc": metrics[0], "cohen_kappa": metrics[1], "weighted_f1": metrics[2]},
                 "provenance": provenance(str(args.config), cfg, {"input_contract": input_receipt})}
        (args.out / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
        return
    if args.smoke_steps:
        raise RuntimeError("Smoke must stop before any native test evaluation")
    checkpoints = sorted(args.out.glob("*.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError("Native trainer did not save exactly one selected checkpoint")
    receipt.update(status="completed", native_params=captured,
                   checkpoints={p.name: sha256(p) for p in checkpoints})
    (args.out / "native_completed.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
