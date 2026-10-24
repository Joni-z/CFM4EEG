"""Run released EEGPT TUAB trainer for an explicit CHB-MIT scratch adaptation.

Launch with torchrun --nproc_per_node=2. The official task model, optimizer,
schedule, actual per-rank batch, and TUEV DeepSpeed path are retained. Dataset
bridge restores microvolts before the native engine's sole /100 normalization.
Validation/test use complete sequential cohorts (no distributed padding).
"""

from __future__ import annotations
import argparse, functools, hashlib, importlib.util, itertools, json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class SmokeFinished(Exception):
    pass


def restore_training_checkpoint(path, args, model, optimizer, scaler):
    """Restore a trusted local training checkpoint without restarting its schedule."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or not 0 <= epoch < args.epochs:
        raise ValueError("Resume needs an integer completed epoch, not a best-model checkpoint")
    if "optimizer" not in checkpoint or "scaler" not in checkpoint:
        raise ValueError("Resume requires the original optimizer and scaler states")
    tensors = list(checkpoint["model"].values())
    for values in checkpoint["optimizer"]["state"].values():
        tensors.extend(v for v in values.values() if torch.is_tensor(v))
    if any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("Cannot resume nonfinite model or optimizer state")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    args.start_epoch = epoch + 1
    return epoch


def previous_best_accuracy(path, through_epoch):
    """Use completed validation records only; held-out scores never select weights."""
    scores = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record["epoch"] > through_epoch:
            raise ValueError("Resume checkpoint predates preserved validation history")
        score = float(record["val_accuracy"])
        if not math.isfinite(score):
            raise ValueError("Nonfinite historical validation accuracy")
        scores.append(score)
    if not scores:
        raise ValueError("Resume checkpoint has no matching validation history")
    return max(scores)


def save_if_validation_improves(score, best, save):
    if not math.isfinite(score):
        raise ValueError("Nonfinite validation accuracy")
    if score > best:
        save()
        return score
    return best


def fp32_logits(module, inputs, output):
    """Keep BF16 model values/gradients while supporting the native NumPy metrics."""
    return output.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument(
        "--native-amp",
        action="store_true",
        help="Use the upstream trainer's native AMP branch instead of its TUEV DeepSpeed0.4 backend; logged compatibility deviation",
    )
    cli = parser.parse_args()
    from paclock_bench.training.recipe_runtime import provenance as runtime_provenance
    import numpy as np
    import torch, yaml
    from torch.utils.data import Dataset, DataLoader
    from paclock_bench.paths import vendored
    from paclock_bench.data.datasets import build_dataloaders
    from paclock_bench.training.metrics import compute_metrics

    amp_dtype = torch.bfloat16 if cli.precision == "bf16" else torch.float16
    amp_enabled = cli.precision != "fp32"
    if cli.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support the requested BF16 runtime")
    if cli.precision != "fp16":
        # The author engine uses this context for training and validation.
        # Keep the engine, optimizer and GradScaler; change arithmetic precision only.
        torch.cuda.amp.autocast = functools.partial(
            torch.cuda.amp.autocast, dtype=amp_dtype, enabled=amp_enabled
        )

    cfg = yaml.safe_load(Path(cli.config).read_text())
    ds = cfg["dataset"]
    if ds != "chbmit" or cfg.get("pretrained") is not False:
        raise ValueError("This release only permits CHB-MIT from scratch")
    data_root = Path(os.path.expandvars(cfg["data_root"]))
    accepted = json.loads((data_root / "ACCEPTED.json").read_text())
    ready = json.loads((data_root / "READY.json").read_text())
    manifest_hash = hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest()
    if accepted.get("status") != "accepted" or accepted.get("manifest_sha256") != manifest_hash or ready.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("CHB-MIT accepted/ready manifest hashes disagree")
    task = cfg.get("native_task", ds)
    if task not in ("tuev", "tuab"):
        raise ValueError("A native EEGPT task recipe must be declared as tuev or tuab")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError(
            "EEGPT native recipe needs 2 ranks, not gradient accumulation in place of BatchNorm microbatch"
        )
    seed = int(cfg.get("seed", 0) if cli.seed is None else cli.seed)
    out = Path(cli.out).resolve()
    if cli.resume and not cli.smoke_steps and Path(cli.resume).resolve().parent != out:
        raise ValueError("Formal continuation must preserve the existing output and selected model")
    rank = int(os.environ.get("RANK", "0"))
    if out.exists() and any(out.iterdir()) and not cli.resume and rank == 0:
        raise FileExistsError(f"Refusing to overwrite {out}; use explicit --resume")
    out.mkdir(parents=True, exist_ok=True)
    vendor = Path(vendored("eegpt")) / "downstream_tueg"
    sys.path.insert(0, str(vendor))
    file = (
        "run_class_finetuning_EEGPT_change"
        + ("_tuev" if task == "tuev" else "")
        + ".py"
    )
    spec = importlib.util.spec_from_file_location("cfft_eegpt_native", vendor / file)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    batch = 400 if task == "tuev" else 100
    epochs = 30 if task == "tuev" else 50
    argv = [
        "--model",
        "EEGPT",
        "--dataset",
        task.upper(),
        "--output_dir",
        str(out),
        "--weight_decay",
        "0.05",
        "--batch_size",
        str(batch),
        "--lr",
        "0.0005",
        "--update_freq",
        "1",
        "--warmup_epochs",
        "5",
        "--epochs",
        str(epochs),
        "--layer_decay",
        "0.65",
        "--save_ckpt_freq",
        "5",
        "--disable_rel_pos_bias",
        "--abs_pos_emb",
        "--seed",
        str(seed),
        "--num_workers",
        str(cfg.get("num_workers", 8)),
    ]
    argv += (
        ["--drop_path", "0.2", "--enable_deepspeed"]
        if task == "tuev"
        else ["--disable_qkv_bias"]
    )
    if cli.native_amp and "--enable_deepspeed" in argv:
        argv.remove("--enable_deepspeed")
    if cli.resume:
        argv += ["--resume", cli.resume]
    original_argv = sys.argv
    sys.argv = [str(vendor / file)] + argv
    try:
        opts, ds_init = native.get_args()
    finally:
        sys.argv = original_argv
    # Pretrained loading is strict in the adapter. Native utils.load_state_dict
    # accepts arbitrary missing tensors, so its permissive path is not reused.
    opts.finetune = ""
    loaders = build_dataloaders(cfg)
    train, val, test, info = loaders
    channels, samples = info["input_shape"]
    if samples != (1000 if task == "tuev" else 2000):
        raise ValueError("Native task temporal head and dataset duration disagree")

    class Microvolts(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, index):
            x, y = self.base[index]
            return x * 100.0, y

    # Every shared loader in this runner must expose uV/100 explicitly; CHB
    # reconstructs it through the q95 denominator sidecar.
    if cfg.get("loader") not in (None, "default") or (
        ds == "chbmit" and cfg.get("loader_signal_mode") != "uv_div100"
    ):
        raise ValueError("Native EEGPT bridge requires shared uV/100 loader contract")
    datasets = tuple(Microvolts(loader.dataset) for loader in (train, val, test))
    metrics = (
        ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        if cfg["num_classes"] == 2
        else ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    )

    def get_dataset(args):
        args.nb_classes = 1 if cfg["num_classes"] == 2 else cfg["num_classes"]
        return datasets[0], datasets[2], datasets[1], None, metrics

    native.get_dataset = get_dataset
    state = {"best_validation_accuracy": 0.0}
    original_load = native.utils.auto_load_model

    def auto_load_model(*args, **kwargs):
        if not cli.resume:
            return original_load(*args, **kwargs)
        if kwargs.get("model_ema") is not None:
            raise ValueError("This recovery supports the recorded non-EMA native recipe only")
        saved_epoch = restore_training_checkpoint(
            cli.resume, kwargs["args"], kwargs["model_without_ddp"],
            kwargs["optimizer"], kwargs["loss_scaler"],
        )
        history = Path(cli.resume).resolve().parent / "log.txt"
        state["best_validation_accuracy"] = previous_best_accuracy(history, saved_epoch)
        if not cli.smoke_steps and not (out / "selected_model.pt").is_file():
            raise FileNotFoundError("Historical validation-selected weights must be preserved")
        state["resume"] = {
            "checkpoint": str(Path(cli.resume).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(cli.resume).read_bytes()).hexdigest(),
            "completed_epoch": saved_epoch, "start_epoch": saved_epoch + 1,
            "previous_best_validation_accuracy": state["best_validation_accuracy"],
            "rng_state_available": False,
        }
        if rank == 0:
            (out / "resume_state.json").write_text(
                json.dumps(dict(state["resume"], precision=cli.precision), indent=2) + "\n"
            )
        print("RESUMED_NATIVE_STATE", json.dumps(state["resume"]), flush=True)

    native.utils.auto_load_model = auto_load_model

    def get_model(args):
        # Official TUAB constructor; its learned channel convolution projects
        # sixteen recorded bipolar inputs into twenty native latent positions.
        # Only in_channels changes for the newly supported CHB-MIT task.
        names = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4',
                 'P8', 'O1', 'O2']
        model = native.EEGPTClassifier(num_classes=args.nb_classes,
            in_channels=channels, img_size=[len(names), samples],
            use_channels_names=names, use_chan_conv=True,
            use_mean_pooling=args.use_mean_pooling)
        if cli.precision == "bf16":
            model.register_forward_hook(fp32_logits)
        active = [(name, value) for name, value in model.target_encoder.named_parameters()
                  if value.requires_grad]
        if not active or not all(value.requires_grad for name, value in
                                  model.target_encoder.named_parameters()
                                  if not name.endswith('freqs')):
            raise RuntimeError("Randomly initialized EEGPT encoder is frozen")
        state["encoder_probe"] = next(value for name, value in active
                                       if 'qkv.weight' in name)
        state["model"] = model
        return model

    native.get_models = get_model
    original_eval = native.evaluate

    def evaluate(loader, model, device, header="Test:", **kwargs):
        if header.startswith("Test"):
            return {k: 0.0 for k in metrics + ["loss"]}
        result = original_eval(loader, model, device, header=header, **kwargs)
        if header.startswith("Val"):
            state["validation_accuracy"] = float(result["accuracy"])
        return result

    native.evaluate = evaluate
    original_save = native.utils.save_model

    def save_model(*args, **kwargs):
        def save():
            original_save(*args, **kwargs)
            if kwargs.get("epoch") == "best" and rank == 0:
                path = out / "selected_model.pt"
                torch.save(
                    kwargs["model_without_ddp"].state_dict(), path.with_suffix(".tmp")
                )
                os.replace(path.with_suffix(".tmp"), path)

        if kwargs.get("epoch") == "best":
            state["best_validation_accuracy"] = save_if_validation_improves(
                state["validation_accuracy"], state["best_validation_accuracy"], save
            )
        else:
            save()

    native.utils.save_model = save_model
    original_train = native.train_one_epoch

    def train_one_epoch(model, criterion, loader, optimizer, *args, **kwargs):
        state["model_runtime"] = model
        if not cli.smoke_steps:
            return original_train(model, criterion, loader, optimizer, *args, **kwargs)

        class Limited:
            def __iter__(self):
                return itertools.islice(iter(loader), cli.smoke_steps)

            def __len__(self):
                return min(len(loader), cli.smoke_steps)

        before = next(state["model"].head.parameters()).detach().clone()
        encoder_before = state["encoder_probe"].detach().clone()
        stats = original_train(model, criterion, Limited(), optimizer, *args, **kwargs)
        if not np.isfinite(stats["loss"]):
            raise RuntimeError("Nonfinite native EEGPT smoke loss")
        after = next(state["model"].head.parameters())
        if not torch.isfinite(after).all():
            raise RuntimeError("Nonfinite native EEGPT parameters")
        if cli.smoke_steps > 1 and torch.equal(before, after):
            raise RuntimeError("Native EEGPT did not update")
        if not torch.isfinite(state["encoder_probe"]).all() or torch.equal(encoder_before, state["encoder_probe"]):
            raise RuntimeError("Native scratch encoder did not update in the smoke")
        state["smoke"] = stats
        raise SmokeFinished()

    native.train_one_epoch = train_one_epoch
    if rank == 0:
        receipt = {
            "config": cfg,
            "seed": seed,
            "native_argv": argv,
            "input": info,
            "global_batch": batch * 2,
            "native_amp_compatibility_override": cli.native_amp,
            "precision": cli.precision,
            "native_source_sha256": {str(path.relative_to(vendor)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (vendor / file, vendor / "engine_for_finetuning_EEGPT.py",
                              vendor / "Modules/models/EEGPT_mcae_finetune_change.py")},
            "adaptations": [
                "new task CHB-MIT, official TUAB binary recipe unchanged",
                "learned input channel convolution 23->20 changed to 16->20; no padding or electrode relabeling",
                "no pretrained weights; active target encoder verified trainable and updated",
                "validation and test sequential without sampler padding",
                "test metrics deferred to final checkpoint",
                "dataset bridge restores uV before native engine division",
            ],
        }
        (out / "provenance.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n"
        )
    try:
        native.main(opts, ds_init)
    except SmokeFinished:
        if rank == 0:
            model = state["model"]
            model.eval()
            x, _ = next(iter(val))
            x = x[:2].to("cuda").float()
            x = x.reshape(x.shape[0], channels, -1, 200)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                reference = model(x).float()
            path = out / "smoke_checkpoint.pt"
            torch.save(model.state_dict(), path)
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False), strict=True
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                restored = model(x).float()
            torch.testing.assert_close(reference, restored, rtol=0, atol=0)
            (out / "smoke.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "status": "passed",
                        "test_evaluated": False,
                        "config": cfg,
                        "checkpoint_reload": "passed",
                        "provenance": runtime_provenance(cli.config, cfg, info),
                        "stats": state["smoke"],
                        "strict_reload_identical": True,
                        "scratch_encoder_updated": True,
                        "world_size": 2,
                        "per_rank_batch": batch,
                        "precision": cli.precision,
                        "resume": state.get("resume"),
                    },
                    indent=2,
                )
                + "\n"
            )
        torch.distributed.barrier()
        return
    torch.distributed.barrier()
    if rank == 0:
        model = state["model"]
        model.load_state_dict(
            torch.load(
                out / "selected_model.pt", map_location="cpu", weights_only=False
            ),
            strict=True,
        )
        model.eval()
        truth = []
        scores = []
        with torch.no_grad():
            for x, y in test:
                x = x.to("cuda").float().reshape(x.shape[0], channels, -1, 200)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    pred = model(x)
                truth.append(y.numpy())
                scores.append(pred.float().cpu().numpy())
        y = np.concatenate(truth)
        logits = np.concatenate(scores)
        np.savez_compressed(out / "test_predictions.npz", labels=y, logits=logits)
        result = {
            "name": cfg["name"],
            "seed": seed,
            "test": compute_metrics(y, logits, cfg["num_classes"]),
            "selected_checkpoint": str(out / "selected_model.pt"),
            "selection_metric": "accuracy",
            "precision": cli.precision,
            "resume": state.get("resume"),
            "config": cfg,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
