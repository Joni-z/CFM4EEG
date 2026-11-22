"""Fit the PAC-only Closure classifier and its dimension-matched null controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from paclock_bench.probing.real_eeg import CFC_FEATURES, FEATURES, SUMMARIES
from paclock_bench.probing.real_eeg_models import sha256
from paclock_bench.training.metrics import compute_metrics
from scripts.experiments.pac_real_eeg_probe import dataset_identity, write_json


SPLITS = ("train", "val", "test")
CLASSES = {"tuev": 6, "chbmit": 2}
DIMENSIONS = len(FEATURES) * len(SUMMARIES)
NULL_SEEDS = tuple(range(4311, 4316))
L2 = 1e-4


def load_inputs(root: Path, dataset: str):
    data = root / "data/processed" / dataset
    cache = root / "cfc-released-40-20261115/targets" / f"{dataset}-X"
    receipt = json.loads((cache / "complete.json").read_text())
    identity = dataset_identity(dict(dataset=dataset, data_root=data,
                                     manifest_sha256=receipt["data_identity"]["manifest_sha256"]))
    if (receipt["status"] != "completed" or identity != receipt["data_identity"]
            or tuple(receipt["features"]) != CFC_FEATURES or tuple(receipt["summaries"]) != SUMMARIES):
        raise ValueError(f"{dataset}: PAC cache does not match the accepted cohort and feature order")
    features, labels = {}, {}
    for split in SPLITS:
        target = cache / f"{split}_targets.npy"
        label = data / f"{split}_labels.npy"
        if (sha256(target) != receipt["target_hashes"][split]
                or sha256(label) != identity["accepted_file_hashes"][label.name]):
            raise ValueError(f"{dataset}/{split}: target or label digest mismatch")
        x = np.load(target, allow_pickle=False)
        y = np.load(label, allow_pickle=False).reshape(-1)
        if x.shape != (len(y), len(CFC_FEATURES) * len(SUMMARIES)) or len(y) != receipt["rows"][split]:
            raise ValueError(f"{dataset}/{split}: feature rows do not match labels or the cache receipt")
        features[split], labels[split] = x[:, :DIMENSIONS], y
    return features, labels, identity, receipt["target_hashes"]


def fit_predict(features, labels, n_classes):
    arrays = {s: np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
              for s, x in features.items()}
    center = arrays["train"].mean(axis=0)
    scale = arrays["train"].std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = torch.from_numpy((arrays["train"] - center) / scale)
    target = torch.tensor(labels["train"], dtype=torch.long)
    counts = target.bincount(minlength=n_classes)
    if len(counts) != n_classes or (counts == 0).any():
        raise ValueError("Training labels do not cover the declared task classes")
    class_weights = counts.to(torch.float64).reciprocal()
    weight = torch.zeros((train.shape[1], n_classes), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=0.8, max_iter=120,
                                  line_search_fn="strong_wolfe")

    def loss_fn():
        optimizer.zero_grad()
        loss = cross_entropy(train @ weight + bias, target, weight=class_weights)
        loss = loss + L2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(loss_fn)
    final_loss = float(loss_fn().detach())
    gradient = max(float(p.grad.abs().max()) for p in (weight, bias))
    w, b = weight.detach().numpy(), bias.detach().numpy()
    predictions = {s: ((arrays[s] - center) / scale) @ w + b for s in SPLITS[1:]}
    if not all(np.isfinite(logits).all() for logits in predictions.values()):
        raise ValueError("Closure classifier produced nonfinite predictions")
    return predictions, dict(center=center.tolist(), scale=scale.tolist(), weight=w.tolist(),
                             bias=b.tolist(), iterations=optimizer.state[weight]["n_iter"],
                             training_loss=final_loss, max_abs_gradient=gradient,
                             class_counts=counts.tolist())


def run(root: Path, output: Path, dataset: str):
    start = time.monotonic()
    features, labels, identity, target_hashes = load_inputs(root, dataset)
    n_classes = CLASSES[dataset]
    scores, params = {}, {}
    predictions = {f"labels_{s}": labels[s] for s in SPLITS[1:]}
    for seed in (None, *NULL_SEEDS):
        if seed is None:
            arm, inputs = "pac", features
        else:
            arm = f"gaussian_{seed}"
            rng = np.random.default_rng(seed)
            inputs = {s: rng.standard_normal(features[s].shape) for s in SPLITS}
        logits, fitted = fit_predict(inputs, labels, n_classes)
        scores[arm] = {s: compute_metrics(labels[s], logits[s], n_classes) for s in SPLITS[1:]}
        params[arm] = fitted
        predictions.update({f"{arm}_{s}": val for s, val in logits.items()})
        print(f"{dataset} {arm}: {scores[arm]['test']} ({time.monotonic() - start:.1f}s)", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.npz"
    np.savez_compressed(output / "predictions.tmp.npz", **predictions)
    (output / "predictions.tmp.npz").replace(path)
    write_json(output / "closure-fit.json", dict(
        status="completed", dataset=dataset, data_identity=identity, target_hashes=target_hashes,
        features=FEATURES, summaries=SUMMARIES, columns=DIMENSIONS, null_seeds=NULL_SEEDS,
        rows={s: len(y) for s, y in labels.items()}, scores=scores, fits=params,
        optimizer=dict(name="LBFGS", l2=L2, lr=0.8, max_iter=120, line_search="strong_wolfe"),
        code_sha256=sha256(Path(__file__)), predictions_sha256=sha256(path),
        elapsed_sec=time.monotonic() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for dataset in CLASSES:
        run(args.root, args.out / dataset, dataset)


if __name__ == "__main__":
    main()
