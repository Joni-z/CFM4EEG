"""Dataset/cache contracts and held-out evaluation for frozen PAC readouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

from .scoring import fit_probe, score_curves


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz(path: str | Path, **arrays) -> None:
    """Never silently replace a dataset, cache or prediction record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    required = {"signals", "curves", "coupled", "split", "group_id", "sample_id", "family", "sample_rate"}
    if missing := required - data.keys():
        raise ValueError(f"dataset is missing {sorted(missing)}")
    x, y = data["signals"], data["curves"]
    if x.ndim != 3 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("signals [N,C,T] and curves [N,K] must have matching nonempty rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite signal/target")
    for key in ("coupled", "split", "group_id", "sample_id", "family"):
        if data[key].shape != (len(x),):
            raise ValueError(f"{key} must have one entry per sample")
    if not np.isin(data["coupled"], [0, 1]).all():
        raise ValueError("coupled must be binary")
    if len(np.unique(data["sample_id"])) != len(x):
        raise ValueError("duplicate sample IDs")
    if set(data["split"].tolist()) != {"train", "val", "test"}:
        raise ValueError("dataset needs nonempty train, val and test splits")
    groups = {split: set(data["group_id"][data["split"] == split].tolist())
              for split in ("train", "val", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("base-waveform group leakage across splits")
    if np.asarray(data["sample_rate"]).size != 1 or float(data["sample_rate"]) <= 0:
        raise ValueError("sample_rate must be a positive scalar")
    return data


def save_cache(path: str | Path, features: np.ndarray, data: dict,
               dataset_path: str | Path, metadata: dict) -> None:
    features = np.asarray(features)
    if features.ndim != 2 or len(features) != len(data["signals"]) or not np.isfinite(features).all():
        raise ValueError("finite feature matrix must match all dataset rows")
    meta = {**metadata, "dataset_sha256": file_sha256(dataset_path)}
    save_npz(path, features=features, sample_id=data["sample_id"],
             metadata=np.array(json.dumps(meta, allow_nan=False)))


def load_cache(path: str | Path, data: dict, dataset_hash: str) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"]
        ids = archive["sample_id"]
        meta = json.loads(str(archive["metadata"]))
    if meta.get("dataset_sha256") != dataset_hash:
        raise ValueError("feature cache was extracted from a different dataset")
    if not np.array_equal(ids, data["sample_id"]):
        raise ValueError("feature cache sample order/identity differs from targets")
    if features.ndim != 2 or len(features) != len(ids) or not np.isfinite(features).all():
        raise ValueError("invalid cached features")
    return features, meta


def evaluate_cache(data: dict, features: np.ndarray, *, alphas=(0., .01, .1, 1., 10., 100.)) -> tuple[dict, np.ndarray]:
    train, val, test = (data["split"] == split for split in ("train", "val", "test"))
    predictions, fit = fit_probe(features[train], data["curves"][train],
                                 features[val], data["curves"][val], features[test], alphas=alphas)
    truth, coupled = data["curves"][test], data["coupled"][test]
    report = {"overall": score_curves(truth, predictions, coupled), "fit": fit, "families": {}}
    for family in np.unique(data["family"][test]):
        rows = data["family"][test] == family
        report["families"][str(family)] = score_curves(truth[rows], predictions[rows], coupled[rows])
    return report, predictions


def paired_error_interval(data: dict, prediction_a: np.ndarray, prediction_b: np.ndarray,
                          *, resamples: int = 1000, seed: int = 0) -> dict:
    """Positive error(B)-error(A) favors A; resample base signals, not bins.

    Equal group weights; only non-null rows. scipy implements the percentile
    bootstrap. Pairing is shared across methods through group-level differences.
    """
    test = data["split"] == "test"
    truth, groups, coupled = data["curves"][test], data["group_id"][test], data["coupled"][test].astype(bool)
    difference = np.mean((truth - prediction_b) ** 2 - (truth - prediction_a) ** 2, axis=1)
    values = np.array([difference[(groups == group) & coupled].mean()
                       for group in np.unique(groups[coupled])])
    if len(values) < 2:
        return {"mean": None, "ci95": None, "reason": "fewer_than_two_coupled_groups"}
    interval = bootstrap((values,), np.mean, method="percentile", confidence_level=.95,
                         n_resamples=resamples, rng=np.random.default_rng(seed)).confidence_interval
    return {"mean": float(values.mean()), "ci95": [float(interval.low), float(interval.high)],
            "n_groups": len(values), "resamples": resamples,
            "definition": "group_mean_MSE_B_minus_MSE_A; positive favors A"}
