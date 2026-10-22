"""Run the pinned REVE downstream training functions on an audited shared cohort.

This is an explicitly named cohort adaptation, not a published-score reproduction.
The optimization loop is imported from official src/dt.py, not approximated by
our generic trainer. Native train_stage's test logging is deferred until the final
validation-selected checkpoint. Native LP->FT passes the last LP weights forward. Recovery snapshots preserve
weights and optimizer state but this adapter does not yet resume native
train_stage loop counters; interrupted runs must not be reported as completed.
"""

from __future__ import annotations
import argparse, contextlib, hashlib, importlib.util, json, os, random, signal, sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    from paclock_bench.training.recipe_runtime import provenance as runtime_provenance
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader, Dataset
    from omegaconf import OmegaConf
    from paclock_bench.paths import vendored, expand
    from paclock_bench.data.datasets import build_dataloaders
    from paclock_bench.training.metrics import compute_metrics
    from scripts.rebuild.reve_positions import positions_for
    from scripts.rebuild.native_adaptations import recover_uv, resample_window

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    out = Path(args.out or ROOT / "runs" / cfg["name"] / f"seed{seed}").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing REVE run {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("REVE native runtime requires allocated CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This runner is single GPU; do not silently divide global batch"
        )
    native_src = Path(vendored("reve")) / "src"
    # The upstream cluster helper only provides distributed metadata here.
    # Single rank is explicit, including local rank zero after CUDA visibility.
    sys.modules["idr_torch"] = SimpleNamespace(
        rank=0, local_rank=0, world_size=1, size=1, is_master=True
    )
    sys.path.insert(0, str(native_src))
    spec = importlib.util.spec_from_file_location(
        "cfft_reve_native_dt", native_src / "dt.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    from models.encoder import REVE
    from models.classifier import ReveClassifier
    from utils.model_utils import freeze_model, unfreeze_model

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda"
    train, val, test, info = build_dataloaders(cfg)
    channels, samples = info["input_shape"][-2:]
    if len(info["input_shape"]) != 2:
        raise ValueError("REVE sequence flattening must be an explicit data contract")
    positions = positions_for(cfg["dataset"], channels, channels_override=cfg.get("position_channels"))
    source_rate = int(cfg["sample_rate"])
    target_rate = int(cfg.get("model_sample_rate", source_rate))
    samples = samples * target_rate // source_rate
    input_bridge = cfg.get("native_input_bridge", "stored")

    class Positioned(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            x, y = self.base[index]
            if input_bridge == "uv_div10":
                x = recover_uv(x, info["manifest"]) / 10.
            elif input_bridge != "stored":
                raise ValueError(f"Unknown REVE input bridge {input_bridge}")
            x = resample_window(x, source_rate, target_rate)
            return x, y, positions.clone()

    def convert(loader, training):
        return DataLoader(
            Positioned(loader.dataset),
            batch_size=cfg["batch_size"],
            shuffle=training,
            num_workers=cfg.get("num_workers", 8),
            pin_memory=True,
            persistent_workers=cfg.get("num_workers", 8) > 0,
            drop_last=training and cfg.get("train_drop_last", False),
        )

    train, val, test = convert(train, True), convert(val, False), convert(test, False)
    weights = str(Path(vendored("reve")).parent / "reve_weights")
    query = None
    if cfg.get("pretrained", True):
        encoder, query = REVE.from_pretrained(weights, local_files_only=True)
        if query is None:
            raise RuntimeError("Released REVE query missing")
    else:
        enc_cfg = yaml.safe_load((native_src / "configs/encoder/base.yaml").read_text())
        encoder = REVE(
            SimpleNamespace(**enc_cfg["transformer"]),
            enc_cfg["freqs"],
            enc_cfg["patch_size"],
            enc_cfg["patch_overlap"],
            enc_cfg["noise_ratio"],
        )
    n_patches = 1 + (samples - encoder.patch_size) // (
        encoder.patch_size - encoder.overlap_size
    )
    model = ReveClassifier(
        encoder,
        cfg["num_classes"],
        dropout=float(cfg.get("native_classifier", {}).get("dropout", 0.0)),
        pooling=cfg.get("native_classifier", {}).get("pooling", "no"),
        out_shape=(channels * n_patches + 1) * encoder.embed_dim,
    ).to(device)
    if query is not None:
        with torch.no_grad():
            model.cls_query_token.copy_(query)
    # Upstream reads the autocast dtype from trainer.torch_dtype but the scaler
    # dtype from top-level torch_dtype. Set both supported options consistently.
    runtime = OmegaConf.create(
        {
            "torch_dtype": "fp16",
            "trainer": {"device": "cuda", "torch_dtype": "fp16", "clip_grad": 2.0},
        }
    )
    state = {"stage": None, "evaluations": 0, "optimizer": None, "scheduler": None}
    original_optimizer = native.get_optimizer

    def capture_optimizer(*a, **kw):
        state["optimizer"] = original_optimizer(*a, **kw)
        return state["optimizer"]

    native.get_optimizer = capture_optimizer
    original_scheduler = native.get_lr_scheduler

    def capture_scheduler(*a, **kw):
        state["scheduler"] = original_scheduler(*a, **kw)
        return state["scheduler"]

    native.get_lr_scheduler = capture_scheduler
    original_test = native.test
    sentinel = object()

    def deferred_test(model_, loader, *a, **kw):
        if loader is sentinel:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        metrics = original_test(model_, loader, *a, **kw)
        state["evaluations"] += 1
        with (out / "validation.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "stage": state["stage"],
                        "evaluation": state["evaluations"],
                        "accuracy": metrics[0],
                        "balanced_acc": metrics[1],
                        "cohen_kappa": metrics[2],
                        "weighted_f1": metrics[3],
                    }
                )
                + "\n"
            )
        # Atomic rolling recovery state; best stage checkpoints remain separate.
        # Keeping every optimizer snapshot would cost tens of GB per run.
        recovery = out / f'{state["stage"]}_recovery.pt'
        torch.save(
            {
                "model": model_.state_dict(),
                "optimizer": state["optimizer"].state_dict(),
                "scheduler": state["scheduler"].state_dict(),
                "state": {
                    k: v
                    for k, v in state.items()
                    if k not in ("optimizer", "scheduler")
                },
                "cfg": cfg,
                "seed": seed,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            recovery.with_suffix(".tmp"),
        )
        os.replace(recovery.with_suffix(".tmp"), recovery)
        return metrics

    native.test = deferred_test

    def stop(signum, frame):
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg,
                "seed": seed,
                "stage": state["stage"],
                "optimizer": (
                    None
                    if state["optimizer"] is None
                    else state["optimizer"].state_dict()
                ),
            },
            out / "interrupted.pt",
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    provenance = {
        "config": cfg,
        "seed": seed,
        "input": info,
        "input_bridge": input_bridge,
        "model_sample_rate": target_rate,
        "native_classifier": cfg.get("native_classifier", {"pooling":"no", "dropout":0.0}),
        "native_source_sha256": hashlib.sha256(
            (native_src / "dt.py").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "dtype": "fp16",
        "native_task_unused_fields": [
            "task.clip",
            "task.linear_probing.dropout",
            "task.fine_tuning.dropout",
        ],
        "adaptations": [
            "shared fixed patient cohort",
            "single GPU at unchanged batch",
            "seed NumPy position noise for reproducibility",
            "test metrics deferred to final validation-selected checkpoint",
            "atomic rolling recovery plus stage best/last checkpoints",
            "set both supported torch_dtype options to fp16 so official GradScaler is enabled alongside autocast",
        ],
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    if args.smoke_steps:
        import itertools

        class Limited:
            def __iter__(self):
                return itertools.islice(iter(train), args.smoke_steps)

            def __len__(self):
                return min(args.smoke_steps, len(train))

        checks = []
        for stage in cfg["native_stages"]:
            name = stage["name"]
            freeze_model(model) if name == "lp" else unfreeze_model(model)
            stage_cfg = OmegaConf.create(
                {k: v for k, v in stage.items() if k != "name"}
            )
            optimizer = native.get_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                stage_cfg.optimizer,
            )
            total_steps = len(train) * stage_cfg.warmup_epochs
            warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: (
                    (10 ** (step / total_steps) - 1) / 9 if step < total_steps else 1.0
                ),
            )
            scaler = torch.amp.GradScaler(device="cuda", enabled=True)
            before = model.cls_query_token.detach().clone()
            loss = native.train_one_epoch(
                model,
                torch.nn.CrossEntropyLoss(),
                optimizer,
                scaler,
                Limited(),
                warmup,
                runtime,
                stage_cfg,
            )
            if not np.isfinite(loss):
                raise RuntimeError("nonfinite native REVE smoke loss")
            if (
                model.cls_query_token.grad is None
                or not torch.isfinite(model.cls_query_token.grad).all()
            ):
                raise RuntimeError("missing/nonfinite native query gradient")
            if args.smoke_steps > 1 and torch.equal(before, model.cls_query_token):
                raise RuntimeError(
                    "native query did not update across warmup smoke steps"
                )
            checks.append(
                {
                    "stage": name,
                    "loss": loss,
                    "query_updated": not torch.equal(before, model.cls_query_token),
                    "optimizer": type(optimizer).__name__,
                    "scale": scaler.get_scale(),
                }
            )
        model.eval()
        x, _, pos = next(iter(val))
        x = x[:2].to(device)
        pos = pos[:2].to(device)
        with torch.no_grad():
            reference = model(x, pos.clone()).float()
        checkpoint = out / "smoke_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True
        )
        with torch.no_grad():
            restored = model(x, pos.clone()).float()
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        (out / "smoke.json").write_text(
            json.dumps(
                {
                    "checks": checks,
                    "shape": [channels, samples],
                    "batch": cfg["batch_size"],
                    "strict_reload_identical": True,
                    "config_sha256": provenance["config_sha256"],
                    "native_source_sha256": provenance["native_source_sha256"],
                    "input": info,
        "input_bridge": input_bridge,
        "model_sample_rate": target_rate,
        "native_classifier": cfg.get("native_classifier", {"pooling":"no", "dropout":0.0}),
                    "passed": True,
                    "status": "passed",
                    "test_evaluated": False,
                    "config": cfg,
                    "checkpoint_reload": "passed",
                    "provenance": runtime_provenance(args.config, cfg, info),
                },
                indent=2,
            )
            + "\n"
        )
        return
    initial_cwd = os.getcwd()
    try:
        for stage in cfg["native_stages"]:
            name = stage["name"]
            state["stage"] = name
            state["evaluations"] = 0
            stage_dir = out / name
            stage_dir.mkdir()
            os.chdir(stage_dir)
            freeze_model(model) if name == "lp" else unfreeze_model(model)
            stage_cfg = OmegaConf.create(
                {k: v for k, v in stage.items() if k != "name"}
            )
            native.train_stage(
                runtime, stage_cfg, model, train, val, sentinel, stage_name=name
            )
            torch.save(model.state_dict(), stage_dir / "model_last.pth")
        final_best = out / cfg["native_stages"][-1]["name"] / "model_best.pth"
        if not final_best.exists():
            raise RuntimeError("No validation-selected checkpoint produced")
        model.load_state_dict(
            torch.load(final_best, map_location="cpu", weights_only=False), strict=True
        )
        model.eval()
        truth = []
        scores = []
        with torch.no_grad():
            for x, y, pos in test:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x.to(device), pos.to(device))
                truth.append(y.numpy())
                scores.append(pred.float().cpu().numpy())
        y = np.concatenate(truth)
        logits = np.concatenate(scores)
        np.savez_compressed(out / "test_predictions.npz", labels=y, logits=logits)
        result = {
            "name": cfg["name"],
            "seed": seed,
            "test": compute_metrics(y, logits, cfg["num_classes"]),
            "config": cfg,
            "selected_checkpoint": str(final_best),
            "selection_metric": "balanced_acc",
            "stopped_by": "native_stage_rules",
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        os.chdir(initial_cwd)


if __name__ == "__main__":
    main()
