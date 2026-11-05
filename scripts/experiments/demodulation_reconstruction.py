"""Frozen-token W/E recovery; no model updates or reconstruction training loss.

The learned projections can be overcomplete. Report their ranks and a known-
weight inverse alongside a train/validation-fitted linear readout, so recovery
is a representation sanity check rather than a claim of clinical utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
import torch

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.analytic import hilbert


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y, predicted):
    residual = float(np.square(y - predicted).sum())
    variance = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    return {"r2": 1.0 - residual / max(variance, 1e-30),
            "nrmse": float(np.sqrt(residual / max(variance, 1e-30)))}


def projection(conv, scale=None):
    weight = conv.weight.detach().cpu().double().numpy()[:, 0, :]
    bias = (conv.bias.detach().cpu().double().numpy() if conv.bias is not None
            else np.zeros(weight.shape[0], dtype=np.float64))
    if scale is not None:
        scale = scale.detach().cpu().double().numpy()
        weight, bias = weight * scale[:, None], bias * scale
    singular = np.linalg.svd(weight, compute_uv=False)
    return weight, bias, {"shape": list(weight.shape),
                         "rank": int(np.linalg.matrix_rank(weight)),
                         "condition_number": float(singular[0] / singular[-1])}


@torch.no_grad()
def extract(frontend, path, count, rng, device, batch_size):
    signals = np.load(path, mmap_mode="r")
    if count > len(signals):
        raise ValueError("requested sample count exceeds split")
    ids = np.sort(rng.choice(len(signals), count, replace=False))
    channels, length = signals.shape[1:]
    patch = frontend.patch_len
    # Fixed interior patches reduce edge effects; every physical band is retained.
    patches = np.unique(np.linspace(2, length // patch - 3, 3).astype(int))
    electrodes = np.unique(np.linspace(0, channels - 1, min(channels, 4)).astype(int))
    out = {key: [] for key in ("tokens", "W", "E", "logE", "band")}
    for start in range(0, len(ids), batch_size):
        x = torch.from_numpy(np.array(signals[ids[start:start + batch_size]], dtype=np.float32)).to(device)
        b = len(x)
        tokens = frontend(x)[0]
        filtered = frontend.sinc(x.reshape(b * channels, 1, length)).reshape(b, channels, frontend.n_bands, length)
        envelope = hilbert(filtered).abs()
        selected = lambda z: z[:, electrodes][:, :, :, patches].reshape(-1, z.shape[-1]).cpu().numpy()
        w = filtered[..., :length // patch * patch].reshape(b, channels, frontend.n_bands, -1, patch)
        e = envelope[..., :length // patch * patch].reshape_as(w)
        out["tokens"].append(selected(tokens))
        out["W"].append(selected(w))
        out["E"].append(selected(e))
        out["logE"].append(selected(torch.log1p(e)))
        out["band"].append(np.broadcast_to(np.arange(frontend.n_bands)[None, None, :, None],
                           (b, len(electrodes), frontend.n_bands, len(patches))).reshape(-1))
        print(path.stem, min(start + b, len(ids)), "/", len(ids), flush=True)
    return {key: np.concatenate(value).astype(np.float64 if key != "band" else np.int64)
            for key, value in out.items()}, {"sample_indices": ids.tolist(),
            "electrodes": electrodes.tolist(), "patches": patches.tolist()}


def render(path, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.2), layout="constrained")
    # Predeclared examples: first test record, first sampled electrode,
    # physical band indices 3 and 11, middle of the three sampled patches.
    for row, band in enumerate((3, 11)):
        index = band * 3 + 1
        for col, target in enumerate(("W", "E")):
            ax = axes[row, col]
            t = np.arange(data[target].shape[1]) / 200
            ax.plot(t, data[target][index], color="#276C78", label="Reference", lw=1.4)
            ax.plot(t, data[target + "_decoded"][index], color="#C17746", label="Decoded",
                    linestyle="--", lw=1.1)
            ax.set(title=f"Band {band + 1}: {'waveform' if target == 'W' else 'envelope'}",
                   xlabel="Time within patch (s)", ylabel="Stored-input units")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(path / "recovery.pdf")
    fig.savefig(path / "recovery.png", dpi=240)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--counts", type=int, nargs=3, default=(128, 32, 64))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--no-plot", action="store_true", help="Render the saved examples on a plotting host instead")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    start = time.time()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    shape = np.load(args.data_root / "train_signals.npy", mmap_mode="r").shape[1:]
    model = build_model(cfg, shape)
    model.load_state_dict(saved.get("model", saved.get("state_dict")), strict=True)
    model.eval().requires_grad_(False).to(args.device)
    fe = model.frontend
    if fe.pac_token_mode != "identity" or fe.waveform_modulation != "quadrature":
        raise ValueError("recovery requires the actual unrotated W+E checkpoint")
    manifest = args.data_root / "manifest.json"
    protocol = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
                "training_config": cfg, "data_root": str(args.data_root),
                "manifest_sha256": sha(manifest), "sample_seed": args.seed,
                "normalization": "stored arrays; no additional scaling",
                "endpoint": "frontend before positional encoding",
                "readout_fit": "train only; alpha selected on validation; test used once",
                "purpose": "content recovery, not clinical utility or proof of novelty"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    sets, indices = {}, {}
    for name, count in zip(("train", "val", "test"), args.counts):
        sets[name], indices[name] = extract(fe, args.data_root / (name + "_signals.npy"),
                                         count, rng, args.device, args.batch_size)
    (args.output / "indices.json").write_text(json.dumps(indices, indent=2) + "\n")
    matrices = {"logE": projection(fe.amplitude_tokenizer, fe.amplitude_scale),
                "W": projection(fe.waveform_quadrature)}
    results = {"projection": {k: v[2] for k, v in matrices.items()}, "targets": {}}
    xmean = sets["train"]["tokens"].mean(0)
    xstd = sets["train"]["tokens"].std(0).clip(1e-8)
    features = {k: (v["tokens"] - xmean) / xstd for k, v in sets.items()}
    examples = {k: sets["test"][k][:192].astype(np.float32) for k in ("W", "E")}
    for target in ("logE", "W"):
        trials = []
        for alpha in (1e-8, 1e-5, 1e-2, 1.0):
            decoder = Ridge(alpha=alpha).fit(features["train"], sets["train"][target])
            error = score(sets["val"][target], decoder.predict(features["val"]))["nrmse"]
            trials.append((error, alpha, decoder))
        _, alpha, decoder = min(trials, key=lambda v: v[0])
        prediction = decoder.predict(features["test"])
        weight, bias, rank = matrices[target]
        # Interleaved coordinates are [a0,w0,a1,w1,...].
        coords = sets["test"]["tokens"][:, ::2] if target == "logE" else sets["test"]["tokens"][:, 1::2]
        inverse = (coords - bias) @ np.linalg.pinv(weight).T
        result = {"alpha": alpha, "validation_candidates": [{"alpha": a, "nrmse": e} for e, a, _ in trials],
                  "held_out_linear": score(sets["test"][target], prediction),
                  "known_weight_inverse": score(sets["test"][target], inverse),
                  "per_band": {str(b): score(sets["test"][target][sets["test"]["band"] == b],
                                prediction[sets["test"]["band"] == b]) for b in range(fe.n_bands)}}
        if target == "logE":
            physical = np.expm1(prediction)
            result["physical_envelope"] = score(sets["test"]["E"], physical)
            examples["E_decoded"] = physical[:192].astype(np.float32)
        else:
            examples["W_decoded"] = prediction[:192].astype(np.float32)
        results["targets"][target] = result
    np.savez_compressed(args.output / "examples.npz", **examples)
    results.update(wall_time_sec=time.time() - start, counts=args.counts,
                   checkpoint_updates=0, source_sha256=sha(__file__))
    (args.output / "scores.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if not args.no_plot:
        render(args.output, examples)
    print(json.dumps(results["projection"]), flush=True)
    print("FINISHED", args.output, flush=True)


if __name__ == "__main__":
    main()
