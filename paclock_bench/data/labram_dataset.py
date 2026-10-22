"""Array unit/shape bridge for explicit input contracts and CPU checks.

Actual LaBraM baseline training calls the original main/engine through
scripts/rebuild/run_labram_original.py. Stored units determine the divisor;
this utility does not define a LaBraM training recipe.
"""

from __future__ import annotations

import os

from ..paths import expand
from .contracts import verify_arrays, describe_contract, labram_divisor, resolve_signal_transform
from .datasets import load_manifest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LaBraMWindowDataset(Dataset):
    def __init__(self, root: str, split: str, *, flatten_sequences: bool = False,
                 divisor: float = 100.0, normalization_scales=None, input_multiplier: float = 1.0):
        sig = os.path.join(root, f"{split}_signals.npy")
        lab = os.path.join(root, f"{split}_labels.npy")
        for p in (sig, lab):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} missing -- use the accepted array cache described in docs/DATA.md")
        self.divisor = divisor
        self.normalization_scales = normalization_scales
        self.input_multiplier = input_multiplier
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
        X = np.array(self.signals[i], dtype=np.float32)
        if self.normalization_scales is not None:
            X = X * self.normalization_scales[i]
        X = (X * self.input_multiplier / self.divisor).astype(np.float32)
        return torch.from_numpy(X), int(self.labels[i])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.signals.shape[1:])

    def class_counts(self) -> np.ndarray:
        return np.bincount(np.asarray(self.labels).ravel())


def build_labram_dataloaders(cfg: dict):
    root = expand(cfg["data_root"])
    manifest = load_manifest(root)
    contract = describe_contract(cfg, manifest, root, "labram")
    flatten = bool(cfg.get("flatten_sequences", False))
    # LaBraM's loader divides raw microvolts by 100. That is correct for the
    # TUH rows, whose arrays labram_native.py stores in microvolts. The
    # cross-corpus rows instead read the group-A arrays, which the frozen
    # protocol has *already* normalised -- div100 for most corpora, so they
    # sit at exactly the scale LaBraM expects after its own division.
    # Dividing again put them at std ~1e-3 and the model collapsed to a
    # constant class (balanced_acc exactly 1/K, zero variance over seeds).
    transforms = {s: resolve_signal_transform(root,s,cfg,manifest) for s in ("train","val","test")}
    if cfg.get("loader_signal_mode") == "uv_div100":
        divisor = float(cfg.get("loader_divisor",1.0))
        if divisor != 1.0: raise ValueError("uv_div100 transformation already divides by100; loader_divisor must be1")
    else:
        divisor = labram_divisor(cfg, manifest)
    contract["loader_divisor"] = divisor
    contract["loader_signal_mode"] = cfg.get("loader_signal_mode","stored")
    contract["physical_scale_recovery"] = "q95_denominators then /100" if transforms["train"][0] is not None else "none"
    sets = {s: LaBraMWindowDataset(root, s, flatten_sequences=flatten,
                                   divisor=divisor,normalization_scales=transforms[s][0],
                                   input_multiplier=transforms[s][1])
            for s in ("train", "val", "test")}
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 16)
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
