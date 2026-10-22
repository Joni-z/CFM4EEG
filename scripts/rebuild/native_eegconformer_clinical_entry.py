"""Pinned EEG-Conformer author model and optimizer loop, declared clinical adaptations.

The author release covers motor imagery, not CHB-MIT. This entry preserves its
model forwards and Adam/CE update block while adapting input/output dimensions,
storage, validation and stopping to the accepted clinical task. It never runs
the generic project trainer or selects a checkpoint on test data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import textwrap
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

COMMIT = "9ae149ba62487ceae723277d13adac27837113d2"
SOURCE_SHA = "866ea540a20bf158dc22450ff393c136def016bcf75ee3d3d646d24161f6954c"
ADAPTATIONS = [
    "New task: accepted patient-disjoint CHB-MIT 16-channel 200-Hz 10-s windows; seed 0.",
    "Only model dimensional changes: spatial kernel 22->16, flattened head 2440->5080, classes 4->2.",
    "Lazy mmap data and existing per-channel Q95 scaling replace MI MATLAB loading/global z-score.",
    "MI class-wise segmentation/reconstruction augmentation disabled: no seizure-label-preservation claim.",
    "Independent validation PR-AUC selects checkpoint; test is evaluated once after selection (author MI code selects on test accuracy).",
    "Batched validation/test replace whole-cohort GPU materialization; no rows dropped.",
    "New-task budget 100 epochs/patience 8 from prior CHB config instead of 2000 MI epochs.",
    "Author model forward, batch size 72, Adam lr 0.0002/betas(0.5,0.999), unweighted CrossEntropyLoss and optimization block unchanged.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verified_source(vendor, *, n_classes=2, n_times=2000):
    path = Path(vendor) / "conformer.py"
    if sha(path) != SOURCE_SHA:
        raise RuntimeError("Author source does not match pinned commit/source hash")
    source = path.read_text()
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name != "ExP"]
    scope = dict(globals())
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    model = scope["Conformer"]()
    # Exact shape-only replacements; all forward methods remain author code.
    old = model[0].shallownet[1]
    model[0].shallownet[1] = nn.Conv2d(40, 40, (16, 1), (1, 1), bias=old.bias is not None)
    feature_count = ((n_times - 24 - 75) // 15 + 1) * 40
    model[2].fc[0] = nn.Linear(feature_count, 256)
    model[2].fc[-1] = nn.Linear(32, n_classes)

    exp = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExP")
    method = next(node for node in exp.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    raw = textwrap.dedent(ast.get_source_segment(source, method, padded=True))
    start = raw.index("    # Optimizers")
    end = raw.index("    test_data = Variable(test_data.type(self.Tensor))")
    optimizer = raw[start:end]
    start = raw.index("    for e in range(self.n_epochs):")
    end = raw.index("        # out_epoch = time.time()")
    loop = raw[start:end]
    augmentation = """            aug_data, aug_label = self.interaug(self.allData, self.allLabel)
            img = torch.cat((img, aug_data))
            label = torch.cat((label, aug_label))"""
    if augmentation not in loop:
        raise RuntimeError("Pinned augmentation patch anchor changed")
    loop = loop.replace(augmentation, "            # MI-only S&R disabled for this declared clinical adaptation.")
    loop = loop.replace("            self.optimizer.step()", """            self.optimizer.step()
            self.steps += 1
            if self.smoke_steps:
                self.check_smoke_step(loss)
                if self.steps >= self.smoke_steps:
                    return self.finish_smoke()
            if self.steps % 100 == 0:
                print(json.dumps({'event': 'step', 'epoch': e, 'step': self.steps,
                                  'loss': float(loss.detach().cpu()),
                                  'elapsed_s': time.time()-self.started}), flush=True)""")
    generated = "def train(self):\n    self.prepare_loaders()\n" + optimizer + loop
    generated += "        if self.after_epoch(e):\n            break\n    return self.finish()\n"
    namespace = dict(globals())
    exec(compile(generated, str(path) + ":train:clinical-adaptations", "exec"), namespace)
    return model, namespace["train"], generated


class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, smoke_count=None):
        self.x = np.load(root / f"{split}_signals.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.indices = None
        if smoke_count is not None:
            # Smoke only: include every class, avoiding undefined binary AUROC
            # from the first all-background rows of chronologically ordered data.
            classes = np.unique(self.y)
            per_class = max(1, smoke_count // len(classes))
            self.indices = np.concatenate([np.flatnonzero(self.y == c)[:per_class] for c in classes])
        self.n = len(self.y) if self.indices is None else len(self.indices)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = idx if self.indices is None else self.indices[idx]
        return torch.from_numpy(np.array(self.x[idx], copy=True)[None]), int(self.y[idx])


class ClinicalExperiment:
    def __init__(self, args, root, model, identity):
        self.args, self.root, self.out = args, root, Path(args.out)
        self.identity = identity
        self.n_classes = 6 if args.dataset == "tuev" else 2
        self.primary = {"chbmit": "pr_auc", "tuev": "cohen_kappa", "tuab": "auroc"}[args.dataset]
        self.batch_size, self.n_epochs = 72, args.epochs
        self.lr, self.b1, self.b2 = 0.0002, 0.5, 0.999
        self.Tensor, self.LongTensor = torch.cuda.FloatTensor, torch.cuda.LongTensor
        self.model = model.cuda()
        self.criterion_cls = nn.CrossEntropyLoss().cuda()
        self.steps, self.smoke_steps = 0, args.smoke_steps
        self.started, self.best, self.bad = time.time(), -1.0, 0
        self.history = []

    def prepare_loaders(self):
        limit = self.smoke_steps * self.batch_size if self.smoke_steps else None
        train = MMapDataset(self.root, "train", limit)
        self.dataloader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True,
                            num_workers=0 if self.smoke_steps else 4, pin_memory=True,
                            persistent_workers=not bool(self.smoke_steps))

    @torch.no_grad()
    def evaluate(self, split, limit=None):
        from paclock_bench.training.metrics import compute_metrics
        self.model.eval()
        loader = torch.utils.data.DataLoader(MMapDataset(self.root, split, limit),
                            batch_size=self.batch_size, shuffle=False, num_workers=0)
        labels, logits = [], []
        for x, y in loader:
            _, z = self.model(x.cuda(non_blocking=True))
            if not torch.isfinite(z).all():
                raise RuntimeError("Non-finite evaluation logits")
            labels.append(y.numpy()); logits.append(z.cpu().numpy())
        labels, logits = np.concatenate(labels), np.concatenate(logits)
        return compute_metrics(labels, logits, self.n_classes), labels, logits

    def check_smoke_step(self, loss):
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite native CE loss")
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError("Non-finite native gradient")

    def checkpoint(self, name, epoch, metric):
        target = self.out / name
        tmp = target.with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch, "steps": self.steps, "val_primary": metric,
                    "identity": self.identity, "seed": 0}, tmp)
        tmp.replace(target)

    def finish_smoke(self):
        # Training/checkpoint round trip and finite validation; never open test.
        self.checkpoint("smoke.pth", 0, 0.0)
        state = torch.load(self.out / "smoke.pth", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        metrics, _, _ = self.evaluate("val", 144)
        save_json(self.out / "SMOKE_PASSED.json", {"status": "passed", "steps": self.steps,
                  "test_opened": False, "identity": self.identity, "val": metrics,
                  "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        print("SMOKE_PASSED", flush=True)

    def after_epoch(self, epoch):
        metrics, _, _ = self.evaluate("val")
        score = metrics[self.primary]
        improved = score > self.best
        if improved:
            self.best, self.bad = score, 0
            self.checkpoint("best.pth", epoch, score)
        else:
            self.bad += 1
        self.checkpoint("last.pth", epoch, score)
        row = {"epoch": epoch, "steps": self.steps, "val": metrics, "improved": improved,
               "bad_epochs": self.bad, "elapsed_s": time.time()-self.started}
        self.history.append(row)
        save_json(self.out / "history.json", self.history)
        print(json.dumps(row), flush=True)
        return self.bad >= self.args.patience

    def finish(self):
        selected = torch.load(self.out / "best.pth", map_location="cpu", weights_only=False)
        self.model.load_state_dict(selected["model"], strict=True)
        metrics, labels, logits = self.evaluate("test")
        np.savez_compressed(self.out / "test_predictions.npz", labels=labels, logits=logits)
        result = {"dataset": self.args.dataset, "model": "eegconformer", "seed": 0,
                  "test": metrics, "best_epoch": selected["epoch"], "best_val": selected["val_primary"],
                  "epochs_run": len(self.history), "steps": self.steps,
                  "stopped_by": "patience" if self.bad >= self.args.patience else "epochs",
                  "wall_hours": (time.time()-self.started)/3600, "identity": self.identity,
                  "adaptations": ADAPTATIONS}
        save_json(self.out / "result.json", result)
        print(json.dumps(result), flush=True)
        return result


def main():
    global ADAPTATIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", choices=("chbmit", "tuev", "tuab"), default="chbmit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-proof")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if args.dataset == "tuev":
        ADAPTATIONS = [
            "New task: accepted TUEV 16-channel 200-Hz 5-s windows; seed 0; shared cohort differs from other papers' native cohorts.",
            "Only model dimensional changes: spatial kernel 22->16, classes 4->6; native flattened head remains 2440.",
            "Lazy mmap data and existing uV/100 scaling replace MI MATLAB loading/global z-score.",
            "MI class-wise segmentation/reconstruction augmentation disabled: no event-label-preservation claim.",
            "Independent validation Cohen kappa selects checkpoint; test evaluated once after selection (author MI code selects on test accuracy).",
            "Batched validation/test replace whole-cohort GPU materialization; no rows dropped.",
            "Declared new-task budget 100 epochs/patience 8 instead of 2000 MI epochs.",
            "Author model forward, batch size 72, Adam lr 0.0002/betas(0.5,0.999), unweighted CrossEntropyLoss and optimization block unchanged.",
        ]
    elif args.dataset == "tuab":
        ADAPTATIONS = [
            "New task: accepted corrected TUAB16-channel200Hz10s, seed0.",
            "Author forward/Adam/CE update block; shape-only spatial kernel16 and binary head5080.",
            "Stored microvolts/100 used directly; no CHB inverse-Q95 operation.",
            "MI segmentation/reconstruction disabled; validation AUROC selects checkpoint.",
            "Declared100epoch/patience8 budget; test evaluated once after selection.",
        ]
    root, out = Path(args.data).resolve(), Path(args.out).resolve()
    if (out / "result.json").exists():
        raise RuntimeError("Completed run exists; refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    accepted = json.loads((root / "ACCEPTED.json").read_text())
    ready = json.loads((root / "READY.json").read_text()) if args.dataset in ("chbmit", "tuab") else accepted
    if args.dataset == "tuab":
        manifest = json.loads((root / "manifest.json").read_text())
        protocol = manifest["protocol"]
        if (manifest.get("dataset") != "tuab" or protocol.get("sample_rate") != 200
                or protocol.get("normalization") != "div100" or len(protocol.get("channels", [])) != 16):
            raise RuntimeError("Expected corrected TUAB16-channel200Hz microvolts/100")
    manifest_sha = sha(root / "manifest.json")
    if accepted.get("status") != "accepted" or any(x.get("manifest_sha256") != manifest_sha for x in (accepted, ready)):
        raise RuntimeError("Dataset lacks current ACCEPTED/READY receipts")
    for split in (("train", "val") if args.smoke_steps else ("train", "val", "test")):
        for kind in ("signals", "labels"):
            name = f"{split}_{kind}.npy"
            stat, receipt = (root / name).stat(), accepted["files"][name]
            if stat.st_size != receipt["size"] or stat.st_mtime_ns != receipt["mtime_ns"]:
                raise RuntimeError("Accepted data changed: " + name)
        shape = np.load(root / f"{split}_signals.npy", mmap_mode="r").shape
        n_times = 1000 if args.dataset == "tuev" else 2000
        if shape[1:] != (16, n_times):
            raise RuntimeError("Unexpected task dimensions: " + str(shape))
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    n_classes = 6 if args.dataset == "tuev" else 2
    model, native_train, generated = verified_source(args.vendor, n_classes=n_classes, n_times=n_times)
    identity = {"upstream_commit": COMMIT, "source_sha256": SOURCE_SHA,
                "runner_sha256": sha(__file__), "manifest_sha256": manifest_sha,
                "accepted_sha256": sha(root / "ACCEPTED.json"), "seed": 0,
                "dataset": args.dataset, "n_classes": n_classes, "n_times": n_times,
                "epochs": args.epochs, "patience": args.patience,
                "adapted_train_sha256": hashlib.sha256(generated.encode()).hexdigest()}
    if not args.smoke_steps:
        if not args.smoke_proof:
            raise RuntimeError("Formal training requires actual GPU smoke proof")
        proof = json.loads(Path(args.smoke_proof).read_text())
        if proof.get("status") != "passed" or proof.get("identity") != identity or proof.get("test_opened") is not False:
            raise RuntimeError("Stale or invalid smoke proof")
    save_json(out / "protocol.json", {"identity": identity, "adaptations": ADAPTATIONS,
              "parameters": sum(p.numel() for p in model.parameters()), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
    (out / "adapted_author_train.py").write_text(generated)
    print(json.dumps({"event": "start", "identity": identity, "adaptations": ADAPTATIONS}), flush=True)
    native_train(ClinicalExperiment(args, root, model, identity))


if __name__ == "__main__":
    main()
