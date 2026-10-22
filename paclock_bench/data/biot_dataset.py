"""Q95 array input bridge for TFM stages and contract checks.

The BIOT baseline itself executes the original dataset/Trainer through
scripts/rebuild/native_biot_entry.py. This bridge records its explicit input
multiplier and preserves already-normalized Q95 arrays.
"""

from __future__ import annotations

import os

from ..paths import expand
from .contracts import verify_arrays, describe_contract, labram_divisor, resolve_signal_transform
from .datasets import load_manifest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BIOTWindowDataset(Dataset):
    def __init__(self, root: str, split: str, *, flatten_sequences: bool = False,
                 input_multiplier: float = 1.0, already_q95: bool = False):
        sig = os.path.join(root, f"{split}_signals.npy")
        lab = os.path.join(root, f"{split}_labels.npy")
        for p in (sig, lab):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} missing -- use the accepted array cache described in docs/DATA.md")
        self.input_multiplier = float(input_multiplier)
        self.already_q95 = already_q95
        self.signals = np.load(sig, mmap_mode="r")
        self.labels = np.load(lab)
        verify_arrays(self.signals, self.labels, sig, lab)

        # ISRUC is stored as (n_seq, 20, C, T) sequences of sleep epochs. Models
        # that consume one epoch at a time -- which is every group-B model on the
        # cross-corpus path -- set flatten_sequences and see (n_seq*20, C, T).
        self.flatten = flatten_sequences and self.signals.ndim == 4
        if self.flatten:
            n, s = self.signals.shape[:2]
            self.signals = np.asarray(self.signals).reshape(n * s, *self.signals.shape[2:])
            self.labels = np.asarray(self.labels).reshape(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        X = np.array(self.signals[i], dtype=np.float32) * self.input_multiplier
        # BIOT utils.py, verbatim
        if not self.already_q95:
            X = X / (np.quantile(np.abs(X), q=0.95, method="linear",
                                 axis=-1, keepdims=True) + 1e-8)
        return torch.from_numpy(X.astype(np.float32)), int(self.labels[i])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.signals.shape[1:])

    def class_counts(self) -> np.ndarray:
        return np.bincount(np.asarray(self.labels).ravel())


def build_biot_dataloaders(cfg: dict):
    root = expand(cfg["data_root"])
    manifest = load_manifest(root)
    contract = describe_contract(cfg, manifest, root, "biot")
    flatten = bool(cfg.get("flatten_sequences", False))
    multiplier = float(cfg.get("loader_input_multiplier", 1.0))
    already_q95 = manifest.get("protocol",{}).get("normalization") == "q95"
    if already_q95 and multiplier != 1.0:
        raise ValueError("q95-normalized arrays must not be multiplied by a physical-unit scalar")
    contract["already_q95"] = already_q95
    contract["loader_input_multiplier"] = multiplier
    contract["model_normalization"] = "per-window per-channel q95 after multiplier"
    sets = {s: BIOTWindowDataset(root, s, flatten_sequences=flatten, input_multiplier=multiplier, already_q95=already_q95)
            for s in ("train", "val", "test")}
    bs = cfg.get("batch_size", 512)
    nw = cfg.get("num_workers", 16)
    common = dict(num_workers=nw, pin_memory=True,
                  persistent_workers=nw > 0)
    loaders = (
        DataLoader(sets["train"], batch_size=bs, shuffle=True,
                   drop_last=bool(cfg.get("train_drop_last", False)), **common),
        DataLoader(sets["val"], batch_size=bs, shuffle=False, drop_last=False, **common),
        DataLoader(sets["test"], batch_size=bs, shuffle=False, drop_last=False, **common),
    )
    info = {
        "manifest": manifest,
        "input_contract": contract,
        "input_shape": sets["train"].shape,
        "class_counts": {s: d.class_counts().tolist() for s, d in sets.items()},
        "n_samples": {s: len(d) for s, d in sets.items()},
    }
    return (*loaders, info)
