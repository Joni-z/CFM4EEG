import json

import numpy as np
import pytest

from paclock_bench.probing.benchmark import (
    evaluate_cache, file_sha256, load_cache, load_dataset, paired_error_interval,
    save_cache, save_npz,
)
from paclock_bench.probing.synthetic import generate_dataset
from paclock_bench.probing.validation import validate_protocol


def test_dataset_cache_and_oracle(tmp_path):
    path = tmp_path / "data.npz"
    save_npz(path, **generate_dataset(64, 32, 64))
    data = load_dataset(path)
    cache = tmp_path / "oracle.npz"
    save_cache(cache, data["coefficients"], data, path, {"purpose": "test_fixture"})
    features, metadata = load_cache(cache, data, file_sha256(path))
    report, predictions = evaluate_cache(data, features)
    assert report["overall"]["curve_r2"] > .99999999
    assert report["overall"]["detection_auroc"] == 1
    assert report["families"]["null"]["curve_r2"] is None
    assert metadata["purpose"] == "test_fixture"
    with pytest.raises(FileExistsError):
        save_npz(path, **data)
    with pytest.raises(ValueError, match="different dataset"):
        load_cache(cache, data, "wrong_hash")
    reversed_data = {**data, "sample_id": data["sample_id"][::-1]}
    with pytest.raises(ValueError, match="sample order"):
        load_cache(cache, reversed_data, file_sha256(path))
    interval = paired_error_interval(data, predictions, np.zeros_like(predictions), resamples=50)
    assert interval["ci95"][0] > 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["group", "duplicate", "nonfinite", "split", "labels"])
def test_dataset_contract_rejects_bad_inputs(tmp_path, failure):
    data = generate_dataset(8, 8, 8)
    if failure == "group":
        data["group_id"][8] = data["group_id"][0]
    elif failure == "duplicate":
        data["sample_id"][1] = data["sample_id"][0]
    elif failure == "nonfinite":
        data["signals"][0, 0, 0] = np.nan
    elif failure == "split":
        data["split"][:] = "train"
    else:
        data["coupled"] = data["coupled"].astype(float)
        data["coupled"][0] = 2
    path = tmp_path / "bad.npz"
    save_npz(path, **data)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_protocol_implementation_controls():
    report = validate_protocol(seed=0)
    assert report["passed"], report["checks"]
    assert len(report["label_permutation_r2"]) == 20
    json.dumps(report, allow_nan=False)
