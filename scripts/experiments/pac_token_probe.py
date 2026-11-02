"""Build/cache/evaluate the PAC readout protocol. No backbone training.

Run from repository root: python -m scripts.experiments.pac_token_probe --help
"""
from __future__ import annotations

import argparse
import importlib.metadata
from itertools import combinations
from pathlib import Path
import subprocess

import numpy as np

from paclock_bench.probing.benchmark import (
    evaluate_cache, file_sha256, load_cache, load_dataset, paired_error_interval,
    save_cache, save_json, save_npz,
)


def generate(args) -> None:
    from paclock_bench.probing import synthetic
    from paclock_bench.probing.synthetic import generate_dataset
    data = generate_dataset(n_train=args.train, n_val=args.val, n_test=args.test,
                            seed=args.seed, n_bins=args.bins)
    data["generator_sha256"] = np.array(file_sha256(synthetic.__file__))
    save_npz(args.output, **data)
    load_dataset(args.output)
    print(f"Dataset: {args.output}; {len(data['signals'])} independent records")


def extract(args) -> None:
    import torch
    from paclock_bench.probing.features import (
        linear_token_budget, load_project_tokenizer, load_tfm_tokenizer,
        normalize_project_signals, project_tokens, tokenize_tfm,
    )
    from scripts.experiments.cfc_operator_capabilities import band_power

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or not np.isfinite(args.input_scale) or args.input_scale <= 0:
        raise ValueError("positive batch size and input scale required")
    data = load_dataset(args.dataset)
    fs = int(data["sample_rate"])
    if fs != 200:
        raise ValueError("first comparison protocol requires common 200 Hz inputs")
    model = None
    if args.row in ("cfft", "wavepatch"):
        if args.config is None or args.checkpoint is None or args.normalization is None:
            raise ValueError("project extraction requires --config, --checkpoint and --normalization")
        model, metadata = load_project_tokenizer(
            args.config, args.checkpoint, (args.training_channels, data["signals"].shape[-1]),
            normalization=args.normalization)
        expected = "paclock" if args.row == "cfft" else "wave_patch_twoaxis"
        if metadata["training_config"]["model"] != expected:
            raise ValueError("row name and checkpoint model do not match")
        if int(metadata["training_config"].get("sample_rate", 200)) != fs:
            raise ValueError("checkpoint sample rate differs from the comparison dataset")
    elif args.row == "tfm":
        if args.vendor_root is None or args.checkpoint is None:
            raise ValueError("TFM extraction requires --vendor-root and --checkpoint")
        model, metadata = load_tfm_tokenizer(args.vendor_root, args.checkpoint)
    else:
        metadata = {"endpoint": "fixed_band_mean_square", "bands_hz":
                    [[.5, 4.], [4., 8.], [8., 13.], [13., 30.], [30., 75.]]}
    if model is not None:
        model.to(args.device).eval()
    chunks = []
    native_shape = None
    for start in range(0, len(data["signals"]), args.batch_size):
        array = data["signals"][start:start + args.batch_size] * args.input_scale
        if model is None:
            features = np.array([[band_power(channel, low, high, fs)
                                  for channel in record for low, high in metadata["bands_hz"]]
                                 for record in array])
        else:
            x = torch.from_numpy(array).to(args.device)
            # The original frontend caches Hilbert tensors process-wide. Use
            # no_grad so a later training forward can reuse them with autograd.
            with torch.no_grad():
                if args.row == "tfm":
                    tokens = tokenize_tfm(model, x)
                else:
                    x = normalize_project_signals(x, args.normalization)
                    tokens = project_tokens(model, x, kind=args.row)
            native_shape = list(tokens.shape[1:])
            if args.time_bins > tokens.shape[1] or args.width > tokens.shape[2]:
                raise ValueError("comparison budget must not upsample time or pad feature width; "
                                 f"native shape is {native_shape}")
            features = linear_token_budget(tokens.detach().cpu().numpy(), args.time_bins,
                                           args.width, seed=args.projection_seed)
        chunks.append(features)
        print(f"{args.row}: {min(start + args.batch_size, len(data['signals']))}/{len(data['signals'])}", flush=True)
    metadata.update(row=args.row, input_scale=args.input_scale,
                    budget={"time_bins": args.time_bins, "width": args.width, "seed": args.projection_seed},
                    budget_applies=args.row != "bandpower", device=args.device,
                    native_token_shape=native_shape,
                    adapter_sha256=file_sha256(Path(__file__).resolve().parents[2] /
                                               "paclock_bench/probing/features.py"),
                    software={name: importlib.metadata.version(name)
                              for name in ("numpy", "torch", "scikit-learn")},
                    purpose="frozen_token_PAC_readout", checkpoint_selection="user_supplied_predeclared_checkpoint")
    save_cache(args.output, np.concatenate(chunks), data, args.dataset, metadata)
    print(f"Cache: {args.output}")


def evaluate(args) -> None:
    data = load_dataset(args.dataset)
    loaded, metadata, paths = {}, {}, {}
    for item in args.cache:
        name, separator, path = item.partition("=")
        if not separator or not name or name in loaded:
            raise ValueError("each --cache must have a unique NAME=PATH")
        loaded[name], metadata[name] = load_cache(path, data, file_sha256(args.dataset))
        paths[name] = str(Path(path).resolve())
    neural = [name for name in loaded if metadata[name].get("budget_applies", True)]
    if neural:
        reference = metadata[neural[0]].get("budget")
        if reference is None:
            raise ValueError("tokenizer cache lacks its linear-budget metadata")
        if any(metadata[name].get("budget") != reference for name in neural):
            raise ValueError("tokenizer caches have different linear readout budgets")
        if len({loaded[name].shape[1] for name in neural}) != 1:
            raise ValueError("tokenizer feature widths differ")
    if len({metadata[name].get("input_scale") for name in loaded}) != 1:
        raise ValueError("feature caches have different input scales")
    args.output.mkdir(parents=True, exist_ok=False)
    rows, predictions = {}, {}
    test = data["split"] == "test"
    for name, features in loaded.items():
        rows[name], predictions[name] = evaluate_cache(data, features)
        rows[name]["source"] = metadata[name]
        rows[name]["cache_path"] = paths[name]
        print(name, rows[name]["overall"], flush=True)
    pairs = {f"{a}__vs__{b}": paired_error_interval(data, predictions[a], predictions[b],
                                                   resamples=args.bootstrap, seed=args.seed)
             for a, b in combinations(loaded, 2)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=Path(__file__).resolve().parents[2]).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    save_json(args.output / "scores.json", {
        "scope": "PAC curve readability; not all CFC or downstream causal use",
        "dataset_sha256": file_sha256(args.dataset), "rows": rows, "paired_errors": pairs,
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "evaluation_source_sha256": {name: file_sha256(Path(__file__).resolve().parents[2] /
                                     "paclock_bench/probing" / name)
                                     for name in ("benchmark.py", "scoring.py")},
        "source_commit": commit,
    })
    save_npz(args.output / "predictions.npz", sample_id=data["sample_id"][test],
             target=data["curves"][test], coupled=data["coupled"][test],
             names=np.array(list(predictions)), predictions=np.stack(list(predictions.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("generate", help="generate analytic PAC targets and raw signals")
    p.add_argument("--output", type=Path, required=True)
    for name, default in (("train", 4000), ("val", 1000), ("test", 3000), ("bins", 24), ("seed", 0)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.set_defaults(function=generate)
    p = commands.add_parser("extract", help="cache final frozen tokens or fixed band power")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--row", choices=("cfft", "tfm", "wavepatch", "bandpower"), required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--vendor-root", type=Path)
    p.add_argument("--normalization", choices=("none", "div100", "q95"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--input-scale", type=float, default=20.)
    for name, default in (("batch-size", 32), ("training-channels", 16), ("time-bins", 9),
                          ("width", 64), ("projection-seed", 0)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.set_defaults(function=extract)
    p = commands.add_parser("evaluate", help="train only ridge probes, write held-out scores")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--cache", action="append", required=True, help="NAME=PATH; repeat for comparisons")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(function=evaluate)
    p = commands.add_parser("validate", help="bounded CPU correctness controls; not model ranking")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(function=lambda args: validate(args.output, args.seed))
    args = parser.parse_args()
    args.function(args)


def validate(output: Path, seed: int) -> None:
    from paclock_bench.probing.validation import validate_protocol
    report = validate_protocol(seed)
    save_json(output, report)
    print(f"Metric implementation controls: {report['passed']}; {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
