"""Joint PAC erasure on a frozen EEG checkpoint and its original task head.

Fit a centered cross-covariance subspace to three PAC targets on training
windows. Subtract only the projected per-window token mean from the original
grid, retaining tokenwise differences. Refit a residual probe and evaluate
the original, erased, shuffled-target and five matched-rank random variants.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.pac_real_eeg_probe import Rows, dataset_identity, write_json
from paclock_bench.probing.real_eeg import ALPHAS, CFC_FEATURES, R2Moments, RidgeMoments, robust_scale
from paclock_bench.probing.real_eeg_models import FrozenReadout, sha256
from paclock_bench.training.metrics import compute_metrics


def token_grid(output):
    return output[0] if isinstance(output, tuple) else output


def pooled(x):
    if x.ndim < 3:
        raise ValueError(f"Expected [batch, token axes, hidden], received {x.shape}")
    return x.mean(tuple(range(1, x.ndim - 1)))


def erase(output, center, basis):
    x = token_grid(output)
    shift = ((pooled(x) - center) @ basis) @ basis.T
    patched = x - shift.reshape(x.shape[0], *((1,) * (x.ndim - 2)), x.shape[-1])
    return (patched, *output[1:]) if isinstance(output, tuple) else patched


def basis_from(cov, rank=None):
    directions, values, _ = np.linalg.svd(cov, full_matrices=False)
    if rank is None:
        rank = int((values > 1e-4 * max(values[0], 1.0)).sum())
    if rank < 1:
        raise ValueError("Training features give a rank-zero PAC subspace")
    return directions[:, :rank]


def projected_moments(fit, basis):
    """Transform saved statistics by the fitted affine eraser without new passes."""
    if basis is None:
        return fit
    project = np.eye(len(fit.sx)) - basis @ basis.T
    res = RidgeMoments(len(fit.sx), len(fit.sy))
    res.n, res.sx, res.sy = fit.n, fit.sx, fit.sy
    xx = np.outer(fit.sx, fit.sx) / fit.n
    xy = np.outer(fit.sx, fit.sy) / fit.n
    res.xx = project @ (fit.xx - xx) @ project + xx
    res.xy = project @ (fit.xy - xy) + xy
    return res


def covariance(fit):
    return (fit.xy - np.outer(fit.sx, fit.sy) / fit.n) / fit.n


class Activation:
    """One exact module hook on an otherwise unchanged task model."""

    def __init__(self, layer, residual=False):
        self.value = None
        self.projection = None
        self.residual = residual
        self.handle = layer.register_forward_hook(self.capture)

    def capture(self, _module, args, output):
        if self.value is not None:
            raise ValueError("Selected module executed more than once in a forward")
        if self.residual:
            # TFM adds the feed-forward delta outside the module. Project
            # that full state, then return its delta for the native addition.
            state = args[0] + output
            if self.projection is not None:
                output = erase(state, *self.projection) - args[0]
                state = args[0] + output
        else:
            if self.projection is not None:
                output = erase(output, *self.projection)
            state = token_grid(output)
        self.value = pooled(state)
        return output

    def take(self):
        if self.value is None:
            raise ValueError("Selected activation hook did not execute")
        value, self.value = self.value, None
        return value


def attach(readout, endpoint):
    if endpoint not in readout.endpoints:
        raise ValueError(f"Unknown intervention endpoint: {endpoint}")
    if readout.kind == "tfm":
        if endpoint == "tokenizer":
            return Activation(readout.model.eeg_token_embedding)
        if endpoint == "layer_04":
            return Activation(readout.model.LAT)
        layer = int(endpoint.split("_")[1]) - 1
        return Activation(readout.model.LAT.layers.layers[layer][1], residual=True)
    if endpoint == "tokenizer":
        return Activation(readout.stem)
    layer = int(endpoint.split("_")[1]) - 1
    return Activation(readout.encoder_layers[layer])


def available_host_memory():
    if os.name != "nt":
        amount = next(int(row.split()[1]) for row in Path("/proc/meminfo").read_text().splitlines()
                      if row.startswith("MemAvailable:"))
        return amount * 1024
    from ctypes import Structure, byref, c_ulong, c_ulonglong, sizeof, windll

    class MemoryStatus(Structure):
        _fields_ = [("length", c_ulong), ("load", c_ulong)] + [
            (field, c_ulonglong) for field in
            ("total", "available", "page_total", "page_available", "virtual_total",
             "virtual_available", "extended_available")
        ]

    status = MemoryStatus()
    status.length = sizeof(status)
    if not windll.kernel32.GlobalMemoryStatusEx(byref(status)):
        raise OSError("Cannot read Windows memory availability")
    return status.available


def resource_guard(out, device):
    if device.type == "cpu":
        return
    started = time.monotonic()
    while True:
        free_gpu = torch.cuda.mem_get_info(device)[0]
        disk = shutil.disk_usage(out).free
        host_floor = 6 if os.name == "nt" else 24
        available = available_host_memory()
        if free_gpu >= 1024 << 20 and disk >= 4 << 30 and available >= host_floor << 30:
            return
        if time.monotonic() - started > 1800:
            raise RuntimeError("Shared memory or disk headroom remains low")
        print(f"Waiting for shared headroom. GPU free {free_gpu / 2**20:.0f} MiB", flush=True)
        time.sleep(15)


def load_targets(job, identity):
    root = Path(job["targets_root"])
    receipt = json.loads((root / "complete.json").read_text())
    if receipt["data_identity"] != identity or tuple(receipt["features"]) != CFC_FEATURES:
        raise ValueError("PAC cache does not match the selected dataset identity")
    arrays = {}
    for split in ("train", "val", "test"):
        p = root / f"{split}_targets.npy"
        if sha256(p) != receipt["target_hashes"][split]:
            raise ValueError(f"PAC cache hash changed: {p}")
        arrays[split] = np.load(p)[:, :18].astype(np.float64)
    center, scale = robust_scale(arrays["train"])
    return {s: (x - center) / scale for s, x in arrays.items()}


def chunked(root, split, batch, expected, need_denominators=False):
    rows = Rows(root / f"{split}_signals.npy")
    try:
        if rows.shape[0] != expected:
            raise ValueError("Targets and input windows disagree")
        scales = np.load(root / f"{split}_q95_denominators.npy", mmap_mode="r") if need_denominators else None
        if scales is not None and scales.shape != (*rows.shape[:2], 1):
            raise ValueError("Expected one accepted q95 denominator per channel and window")
        for lo in range(0, expected, batch):
            hi = min(lo + batch, expected)
            yield lo, hi, expected, (torch.from_numpy(rows.read(lo, hi)),
                                     None if scales is None else scales[lo:hi])
    finally:
        rows.close()


def elapsed_progress(out, phase, arm, hi, total, device, started):
    write_json(out / "progress.json", dict(phase=phase, arm=arm, completed=hi, total=total,
               elapsed_sec=time.monotonic()-started,
               cuda_peak_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0))


@torch.inference_mode()
def forward(readout, hook, inputs, delay):
    start = time.monotonic()
    logits = readout.task_logits(*inputs).detach().float().cpu().numpy()
    hidden = hook.take().double().cpu().numpy()
    if not np.isfinite(logits).all() or not np.isfinite(hidden).all():
        raise ValueError("Nonfinite logits or representation")
    if delay:
        time.sleep(delay * (time.monotonic()-start))
    return hidden, logits


def smoke(readout, hook, inputs, device, out):
    hidden, reference = forward(readout, hook, inputs, 0)
    other_hidden, other = forward(readout, hook, inputs, 0)
    np.testing.assert_allclose(reference, other, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hidden, other_hidden, rtol=1e-6, atol=1e-7)
    width = hidden.shape[1]
    center = np.zeros(width, np.float32)
    basis = np.eye(width, 2, dtype=np.float32)
    hook.projection = (torch.from_numpy(center).to(device), torch.from_numpy(basis).to(device))
    edited, _ = forward(readout, hook, inputs, 0)
    actual = hidden.copy()
    actual[:, :2] = 0
    np.testing.assert_allclose(actual, edited, rtol=1e-5, atol=1e-6)
    hook.projection = None
    _, after = forward(readout, hook, inputs, 0)
    np.testing.assert_allclose(reference, after, rtol=1e-6, atol=1e-7)
    write_json(out / "smoke.json", dict(status="passed", samples=len(inputs[0]), width=width,
               allocated_peak_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0,
               gpu_free_mib=torch.cuda.mem_get_info(device)[0]/2**20 if device.type=="cuda" else None,
               logits_untouched_after_hook_removal=True, projection_mean_verified=True))


def run(job, device, smoke_only):
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out/"result.json").exists():
        raise FileExistsError(out/"result.json")
    root = Path(job["data_root"])
    identity = dataset_identity(job)
    targets = load_targets(job, identity)
    rows = Rows(root / "train_signals.npy")
    shape = tuple(rows.shape[1:])
    witness = torch.from_numpy(rows.read(0, job["batch_size"]))
    rows.close()
    need_denominators = job["dataset"] == "chbmit" and job["kind"] != "cfft"
    scale_path = root / "train_q95_denominators.npy"
    witness_scale = np.load(scale_path, mmap_mode="r")[:len(witness)] if need_denominators else None
    resource_guard(out, device)
    readout = FrozenReadout(job, shape, str(device))
    for h in readout.handles:
        h.remove()
    hook = attach(readout, job["endpoint"])
    smoke(readout, hook, (witness, witness_scale), device, out)
    if smoke_only:
        return
    rng = np.random.default_rng(4311)
    labels = {s:np.load(root/f"{s}_labels.npy").reshape(-1) for s in ("val","test")}
    if any(len(y)!=len(targets[s]) for s,y in labels.items()):
        raise ValueError("Task labels and PAC windows disagree")
    batch, delay = job["batch_size"], job["delay_ratio"]
    started = time.monotonic()
    clock = started
    train = targets["train"]
    random_rows = rng.permutation(len(train))
    fit = shuffled = None
    with torch.inference_mode():
        for lo,hi,total,inputs in chunked(root,"train",batch,len(train),need_denominators):
            if lo == 0 or time.monotonic()-clock>10:
                resource_guard(out,device)
                elapsed_progress(out,"fit","original",lo,total,device,started)
                clock = time.monotonic()
            h,_ = forward(readout,hook,inputs,delay)
            if fit is None:
                fit = RidgeMoments(h.shape[1],18)
                shuffled = RidgeMoments(h.shape[1],18)
            fit.update(h,train[lo:hi])
            shuffled.update(h,train[random_rows[lo:hi]])

        basis = basis_from(covariance(fit))
        rank = basis.shape[1]
        center = fit.sx / fit.n
        arms = {"original":None, "erase_pac":basis, "erase_shuffled":basis_from(covariance(shuffled),rank)}
        for i in range(5):
            arms[f"random_{i}"] = np.linalg.qr(rng.standard_normal(basis.shape),mode="reduced")[0]
        np.savez(out/"projectors.npz",center=center,**{k:v for k,v in arms.items() if v is not None})
        scores, predictions, sanity = {}, {}, {}
        primary = "pr_auc" if job["dataset"]=="chbmit" else "cohen_kappa"
        for arm,u in arms.items():
            statistic = projected_moments(fit,u)
            fits = [statistic.fit(alpha) for alpha in ALPHAS]
            hook.projection = None if u is None else tuple(torch.as_tensor(
                a,dtype=torch.float32,device=device) for a in (center,u))
            scores[arm], sanity[arm] = {}, {}
            selected = None
            for split in ("val","test"):
                ys = targets[split]
                metrics = [R2Moments(18) for _ in ALPHAS] if split=="val" else [R2Moments(18)]
                logits = []
                for lo,hi,total,inputs in chunked(root,split,batch,len(ys),need_denominators):
                    if lo == 0 or time.monotonic()-clock>10:
                        resource_guard(out,device)
                        elapsed_progress(out,split,arm,lo,total,device,started)
                        clock=time.monotonic()
                    h,raw = forward(readout,hook,inputs,delay)
                    logits.append(raw)
                    if split == "val":
                        for metric,(weights,offset) in zip(metrics,fits):
                            metric.update(ys[lo:hi],h @ weights+offset)
                    else:
                        prediction = np.column_stack([
                            h @ fits[idx][0][:,i*6:(i+1)*6]+fits[idx][1][i*6:(i+1)*6]
                            for i,idx in enumerate(selected)])
                        metrics[0].update(ys[lo:hi],prediction)
                logits = np.concatenate(logits)
                scores[arm][split] = compute_metrics(labels[split],logits,
                                                    2 if primary=="pr_auc" else 6)
                predictions[f"{arm}_{split}"] = logits
                if split == "val":
                    grid = np.stack([m.groups() for m in metrics])
                    selected = grid.argmax(axis=0)
                    sanity[arm].update(alpha=[ALPHAS[i] for i in selected],
                                       val_r2=grid[selected,np.arange(3)].tolist())
                else:
                    sanity[arm]["test_r2"] = metrics[0].groups().tolist()
                write_json(out/"partial.json",dict(scores=scores,pac_readout=sanity,rank=rank))
        np.savez_compressed(out/"predictions.npz",**predictions,**{f"labels_{k}":v for k,v in labels.items()})
        before = scores["original"]["test"][primary]
        drops = {k:before-v["test"][primary] for k,v in scores.items() if k!="original"}
        write_json(out/"result.json",dict(status="completed",job=job,data_identity=identity,
            method="centered cross-covariance projection of per-window mean, broadcast back to grid",
            primary=primary,test_drops=drops,residual_pac_readout=sanity,scores=scores,
            rank=rank,elapsed_sec=time.monotonic()-started))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job",type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    job = json.loads(args.job.read_text())
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if device.type=="cuda":
        if torch.cuda.device_count()!=1:
            raise ValueError("Exactly one permitted GPU must be visible")
        available = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(job["cuda_allocator_mib"]*2**20/available,device)
    run(job,device,args.smoke)


if __name__=="__main__":
    main()
