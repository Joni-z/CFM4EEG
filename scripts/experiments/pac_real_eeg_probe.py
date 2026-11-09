"""Frozen real-EEG PAC readout; no clinical optimization or activation cache.

One job JSON binds data, checkpoint SHA, author source hashes and output path.
Run with --targets to create the shared FFT/Tort labels, or --smoke to verify
small-batch extraction before submitting the full job. See docs/PAC_PROBE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from paclock_bench.probing.real_eeg import (
    ALPHAS, FEATURES, SUMMARIES, R2Moments, RidgeMoments, pac_targets, robust_scale,
)
from paclock_bench.probing.real_eeg_models import FrozenReadout, sha256


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def headroom(out: Path, device: str) -> dict:
    """Wait on low shared headroom; only this evaluator can stop itself."""
    started = time.monotonic()
    while True:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(info["MemAvailable"].split()[0]) * 1024
        disk = shutil.disk_usage(out).free
        gpu_free = torch.cuda.mem_get_info()[0] if device.startswith("cuda") else 100 << 30
        if available >= 24 << 30 and disk >= 4 << 30 and gpu_free >= 12 << 30:
            return dict(host_available_GiB=available / 2**30, disk_free_GiB=disk / 2**30,
                        gpu_free_GiB=gpu_free / 2**30)
        if time.monotonic() - started > 1800:
            raise RuntimeError("Shared memory/disk/GPU headroom remained too low; stopping our evaluator")
        print("waiting for shared headroom", available, disk, gpu_free, flush=True)
        time.sleep(10)


class Rows:
    """Sequential NPY reads, without mapping tens of GiB into process RSS."""
    def __init__(self, path: Path):
        self.stream = path.open("rb")
        version = np.lib.format.read_magic(self.stream)
        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
        if reader is None:
            raise ValueError("Unsupported NPY header")
        self.shape, fortran, self.dtype = reader(self.stream)
        if fortran or self.dtype != np.dtype("float32") or len(self.shape) != 3:
            raise ValueError("Expected contiguous float32 [N,C,T] data")
        self.offset = self.stream.tell()
        self.row_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        if path.stat().st_size != self.offset + self.row_bytes * self.shape[0]:
            raise ValueError("Truncated or appended NPY payload")

    def read(self, start: int, stop: int) -> np.ndarray:
        self.stream.seek(self.offset + start * self.row_bytes)
        payload = self.stream.read((stop - start) * self.row_bytes)
        return np.frombuffer(payload, self.dtype).reshape(stop - start, *self.shape[1:]).copy()

    def close(self) -> None:
        self.stream.close()


def dataset_identity(job: dict) -> dict:
    root = Path(job["data_root"])
    accepted = json.loads((root / "ACCEPTED.json").read_text())
    if accepted["status"] != "accepted" or accepted["dataset"] != job["dataset"]:
        raise ValueError("Dataset not accepted")
    manifest_hash = sha256(root / "manifest.json")
    if manifest_hash != accepted["manifest_sha256"] or manifest_hash != job["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    for split in ("train", "val", "test"):
        names = [f"{split}_signals.npy", f"{split}_labels.npy"]
        if job["dataset"] == "chbmit":
            names.append(f"{split}_q95_denominators.npy")
        for name in names:
            recorded, current = accepted["files"][name], (root / name).stat()
            if current.st_size != recorded["size"] or current.st_mtime_ns != recorded["mtime_ns"]:
                raise ValueError("Accepted data file changed: " + name)
    return dict(manifest_sha256=manifest_hash, accepted_sha256=sha256(root / "ACCEPTED.json"),
                accepted_file_hashes={k: v["sha256"] for k, v in accepted["files"].items()})


def targets(job: dict, device: str, smoke: bool) -> None:
    out = Path(job["targets_root"])
    out.mkdir(parents=True, exist_ok=True)
    identity = dataset_identity(job)
    if not smoke and (out / "complete.json").exists():
        saved = json.loads((out / "complete.json").read_text())
        if saved["data_identity"] != identity:
            raise ValueError("Existing PAC labels refer to another data identity")
        print("verified existing PAC label identity", flush=True)
        return
    counts, quality = {}, {}
    for split in (("train",) if smoke else ("train", "val", "test")):
        rows = Rows(Path(job["data_root"]) / f"{split}_signals.npy")
        n = min(8, rows.shape[0]) if smoke else rows.shape[0]
        y = np.empty((n, 18), dtype=np.float32)
        qc = dict(empty_phase_bins=0, degenerate_channel_pairs=0)
        for start in range(0, n, 32):
            if start % 3200 == 0:
                headroom(out, device)
            stop = min(start + 32, n)
            x = torch.from_numpy(rows.read(start, stop)).to(device)
            value, diagnostic = pac_targets(x)
            y[start:stop] = value.cpu().numpy().reshape(-1, 18)
            for key in qc:
                qc[key] += diagnostic[key]
            if smoke:
                reference, _ = pac_targets(x.cpu())
                np.testing.assert_allclose(value.cpu().numpy(), reference.numpy(), rtol=1e-8, atol=1e-10)
            if start % 3200 == 0:
                write_json(out / ("smoke_progress.json" if smoke else "progress.json"),
                           dict(status="running", split=split, completed=stop, total=n))
        rows.close()
        if not smoke:
            np.save(out / f"{split}_targets.npy", y)
        counts[split], quality[split] = n, qc
    write_json(out / ("smoke.json" if smoke else "complete.json"),
               dict(status="passed" if smoke else "completed", features=FEATURES, summaries=SUMMARIES,
                    rows=counts, quality=quality, data_identity=identity,
                    feature_definition="Tang et al. Appendix A X001-X003, C.3; 18 phase bins; six summaries",
                    code_sha256=sha256(ROOT / "paclock_bench/probing/real_eeg.py"),
                    target_hashes={} if smoke else {s: sha256(out / f"{s}_targets.npy") for s in counts}))


def run_model(job: dict, device: str, smoke: bool) -> None:
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    if not smoke and (out / "result.json").exists():
        raise FileExistsError("Preserving completed probe " + str(out))
    data = dataset_identity(job)
    root = Path(job["data_root"])
    rows = Rows(root / "train_signals.npy")
    model = FrozenReadout(job, tuple(rows.shape[1:]), device)
    if smoke:
        x = torch.from_numpy(rows.read(0, min(2, rows.shape[0])))
        scales = np.load(root / "train_q95_denominators.npy", mmap_mode="r")[:len(x)] if job["dataset"] == "chbmit" else None
        first, second = model(x, scales), model(x, scales)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], rtol=1e-7, atol=1e-9)
        torch.cuda.synchronize()
        write_json(out / "smoke.json", dict(status="passed", strict_checkpoint_loading=True,
                   checkpoint_sha256=job["checkpoint_sha256"], data_identity=data,
                   endpoints={k: list(v.shape) for k, v in first.items()},
                   training_config=model.training_config, source_hashes=model.source_hashes,
                   cuda_peak_GiB=torch.cuda.max_memory_allocated() / 2**30))
        rows.close()
        return
    rows.close()
    start_time = time.monotonic()
    target_root = Path(job["targets_root"])
    while not (target_root / "complete.json").exists():
        headroom(out, device)
        write_json(out / "progress.json", dict(status="waiting_for_shared_PAC_targets"))
        time.sleep(10)
    receipt = json.loads((target_root / "complete.json").read_text())
    if receipt["data_identity"] != data or receipt["code_sha256"] != sha256(ROOT / "paclock_bench/probing/real_eeg.py"):
        raise ValueError("PAC labels have different data or estimator provenance")
    raw = {s: np.load(target_root / f"{s}_targets.npy").astype(np.float64) for s in ("train", "val", "test")}
    if any(sha256(target_root / f"{s}_targets.npy") != receipt["target_hashes"][s] for s in raw):
        raise ValueError("PAC target SHA mismatch")
    center, scale = robust_scale(raw["train"])
    true = {s: (v - center) / scale for s, v in raw.items()}
    rng = np.random.default_rng(4311)
    permutation = rng.permutation(len(true["train"]))
    gauss = {s: rng.normal(size=v.shape) for s, v in true.items()}
    target = {s: np.concatenate((v, true["train"][permutation] if s == "train" else v, gauss[s]), 1)
              for s, v in true.items()}
    moments, fits, validation, test = {}, {}, {}, {}
    selected = None
    saved_predictions = None
    for split in ("train", "val", "test"):
        rows = Rows(root / f"{split}_signals.npy")
        if rows.shape[0] != len(raw[split]):
            raise ValueError("PAC target row count mismatch")
        denominators = np.load(root / f"{split}_q95_denominators.npy", mmap_mode="r") if job["dataset"] == "chbmit" else None
        batch = job.get("batch_size", 4)
        for start in range(0, rows.shape[0], batch):
            if start % (batch * 100) == 0:
                resources = headroom(out, device)
            stop = min(start + batch, rows.shape[0])
            extracted = model(torch.from_numpy(rows.read(start, stop)),
                              None if denominators is None else denominators[start:stop])
            y = target[split][start:stop]
            for name, h in extracted.items():
                if split == "train":
                    if name not in moments:
                        moments[name] = RidgeMoments(h.shape[1], y.shape[1])
                    moments[name].update(h, y)
                elif split == "val":
                    if name not in validation:
                        validation[name] = [R2Moments(54) for _ in ALPHAS]
                    for score, (coef, intercept) in zip(validation[name], fits[name]):
                        score.update(y, h @ coef + intercept)
                else:
                    for entry in selected:
                        if entry["endpoint"] != name:
                            continue
                        feature = entry["feature_index"]
                        columns = slice(feature * 6, (feature + 1) * 6)
                        coef, intercept = fits[name][entry["alpha_index"]]
                        prediction = h @ coef[:, columns] + intercept[columns]
                        key = entry["label"]
                        test[key].update(true[split][start:stop, columns], prediction)
                        saved_predictions[key][start:stop] = prediction * scale[columns] + center[columns]
            if start % (batch * 100) == 0 or stop == rows.shape[0]:
                write_json(out / "progress.json", dict(status="running", split=split,
                    completed=stop, total=rows.shape[0], elapsed_sec=time.monotonic() - start_time,
                    cuda_peak_GiB=torch.cuda.max_memory_allocated() / 2**30, **resources))
        rows.close()
        if split == "train":
            fits = {name: [m.fit(alpha) for alpha in ALPHAS] for name, m in moments.items()}
            del moments
        if split == "val":
            grid = {name: np.stack([r.groups().reshape(3, 3) for r in values])
                    for name, values in validation.items()}
            selected = []
            encoder_names = model.endpoints[1:]
            for feature, feature_name in enumerate(FEATURES):
                for scope, names in (("tokenizer", ["tokenizer"]), ("encoder_peak", encoder_names)):
                    choices = [(float(grid[n][a, 0, feature]), n, a) for n in names for a in range(len(ALPHAS))]
                    # Stable first maximum; test has not been passed through the model.
                    score, name, index = max(choices, key=lambda item: item[0])
                    controls = {c: float(grid[name][:, i, feature].max())
                                for i, c in ((1, "shuffled"), (2, "gaussian"))}
                    entry = dict(label=scope + "/" + feature_name, scope=scope, feature=feature_name,
                        feature_index=feature, endpoint=name, alpha=ALPHAS[index], alpha_index=index,
                        validation_r2=score, validation_controls=controls)
                    selected.append(entry)
            write_json(out / "selection.json", dict(selection_split="val", selected=selected,
                       alphas=ALPHAS, validation_grid={k: v.tolist() for k, v in grid.items()}))
            test = {entry["label"]: R2Moments(6) for entry in selected}
            saved_predictions = {entry["label"]: np.empty((len(true["test"]), 6), np.float32) for entry in selected}
    for entry in selected:
        entry["test_r2"] = float(test[entry["label"]].groups()[0])
        entry["test_raw_r2_by_summary"] = test[entry["label"]].raw().tolist()
    np.savez_compressed(out / "test_predictions.npz", **saved_predictions)
    arrays = {"target_center": center, "target_scale": scale}
    for name, solutions in fits.items():
        for i, (coef, intercept) in enumerate(solutions):
            arrays[name + f"_a{i}_coef"] = coef
            arrays[name + f"_a{i}_intercept"] = intercept
    np.savez_compressed(out / "probes.npz", **arrays)
    write_json(out / "result.json", dict(status="completed", evidence_type="real_eeg_pac_readout",
        dataset=job["dataset"], model=job["kind"], checkpoint_sha256=job["checkpoint_sha256"],
        checkpoint_archive=job.get("archive"), source_hashes=model.source_hashes,
        job=job, data_identity=data, source_paper="https://arxiv.org/html/2605.11410v2",
        scope="PAC encoding only; task-specific scratch-trained weights, adapted datasets; no erasure",
        endpoints=selected, wall_time_sec=time.monotonic() - start_time,
        cuda_peak_GiB=torch.cuda.max_memory_allocated() / 2**30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--targets", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This launch expects one visible CUDA GPU")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((6 << 30) / total)
    out = Path(job["targets_root"] if args.targets else job["out"])
    out.mkdir(parents=True, exist_ok=True)
    headroom(out, "cuda:0")
    try:
        if args.targets:
            targets(job, "cuda:0", args.smoke)
        else:
            run_model(job, "cuda:0", args.smoke)
    except Exception as error:
        write_json(out / ("smoke_error.json" if args.smoke else "error.json"),
                   dict(status="failed", error=repr(error), pid=os.getpid()))
        raise


if __name__ == "__main__":
    main()
