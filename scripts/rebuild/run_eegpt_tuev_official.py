"""Run pinned EEGPT TUEV main/model/engine with explicit native23/shared16 inputs.

The upstream native AMP branch is used instead of the released shell's
DeepSpeed0.4 backend. Optimizer, schedules, batch, epochs, distributed validation,
model constructor and pickle reader remain the upstream implementations.
"""

from __future__ import annotations
import argparse, functools, hashlib, importlib.util, itertools, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_shared_tuev(root: Path, accepted: dict) -> dict:
    """Bind shared16 data to its acceptance receipt without reading test payloads."""
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (accepted.get('status') != 'accepted' or accepted.get('dataset') != 'tuev'
            or accepted.get('manifest_sha256') != digest or manifest.get('dataset') != 'tuev'):
        raise RuntimeError('Shared16 acceptance/manifest mismatch')
    protocol = manifest['protocol']
    if (protocol.get('sample_rate') != 200 or protocol.get('normalization') != 'div100'
            or len(protocol.get('channels', [])) != 16 or protocol.get('num_classes') != 6):
        raise RuntimeError('Shared16 requires declared 200Hz/16-channel/uV-div100/six-class data')
    required = {split+'_'+kind+'.npy' for split in ['train', 'val', 'test']
                for kind in ['signals', 'labels']}
    if not required.issubset(accepted.get('files', {})):
        raise RuntimeError('Shared16 acceptance does not cover all six arrays')
    for name, receipt in accepted['files'].items():
        actual = (root / name).stat()
        if actual.st_size != receipt['size'] or actual.st_mtime_ns != receipt['mtime_ns']:
            raise RuntimeError('Accepted shared16 file changed: '+name)
    return manifest


class SmokeFinished(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--smoke-proof")
    cli = parser.parse_args()
    import numpy as np
    import torch, yaml
    from torch.utils.data import Dataset, DataLoader
    from paclock_bench.paths import vendored
    from paclock_bench.training.metrics import compute_metrics

    amp_dtype = torch.bfloat16 if cli.precision == "bf16" else torch.float16
    amp_enabled = cli.precision != "fp32"
    if cli.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Allocated GPU does not support BF16")
    from scripts.rebuild.eegpt_pretrained import configure_native_autocast
    configure_native_autocast(cli.precision)

    cfg = yaml.safe_load(Path(cli.config).read_text())
    ds = cfg["dataset"]
    if ds != "tuev" or type(cfg.get("pretrained")) is not bool:
        raise ValueError("Explicit scratch/pretrained setting is required")
    from scripts.rebuild.eegpt_pretrained import load_released_pretraining, smoke_identity, require_smoke
    pretrained_identity = None
    if cfg["pretrained"] and cli.precision != cfg.get("precision", "fp16"):
        raise ValueError("Pretrained arithmetic must match the explicit reviewed config")
    data_root = Path(os.path.expandvars(cfg["data_root"]))
    accepted_file = data_root / "ACCEPTED.json"
    accepted = json.loads(accepted_file.read_text())
    shared16 = cfg.get('input_protocol', 'native23') == 'shared16'
    if cfg.get('input_protocol', 'native23') not in ['native23', 'shared16']:
        raise ValueError('Unknown TUEV input protocol')
    accepted_hash = hashlib.sha256(accepted_file.read_bytes()).hexdigest()
    if shared16:
        manifest = verify_shared_tuev(data_root, accepted)
        split_counts = {'train': 68445, 'val': 15487, 'test': 29421}
        if {k: v['n_windows'] for k, v in manifest['splits'].items()} != split_counts:
            raise RuntimeError('Shared16 manifest has the wrong cohort counts')
    else:
        transfer = json.loads((data_root / "TRANSFER_VERIFIED.json").read_text())
        if (accepted.get("status") != "accepted" or accepted.get("dataset") != "TUEV-v2.0.1-native23-LaBraM"
                or transfer.get("accepted_sha256") != accepted_hash):
            raise RuntimeError("Native23 data/transfer acceptance disagrees")
        order_file = data_root / "SPLIT_FILES.json"
        if hashlib.sha256(order_file.read_bytes()).hexdigest() != accepted["metadata"]["SPLIT_FILES.json"]:
            raise RuntimeError("Native original split/order metadata changed")
        files = json.loads(order_file.read_text())
        split_counts = {split: len(names) for split, names in files.items()}
        if split_counts != {"train": 68910, "val": 15022, "test": 28305}:
            raise RuntimeError("Wrong native23 cohort")
    if cfg['pretrained']:
        pretrained_identity = smoke_identity(cli.config, __file__, str(accepted_file), cfg)
        if not cli.smoke_steps:
            require_smoke(cli.smoke_proof, pretrained_identity)
    task = "tuev"
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Official TUEV recipe requires two ranks")
    seed = int(cfg.get("seed", 0) if cli.seed is None else cli.seed)
    if seed != 0 or cfg.get("batch_size") != 400 or cfg.get("epochs") != 30:
        raise ValueError("Only seed0 with native400/rank batch and30epochs is reviewed")
    out = Path(cli.out).resolve()
    rank = int(os.environ.get("RANK", "0"))
    if out.exists() and any(out.iterdir()) and not cli.resume and rank == 0:
        raise FileExistsError(f"Refusing to overwrite {out}; use explicit --resume")
    out.mkdir(parents=True, exist_ok=True)
    vendor = Path(vendored("eegpt")) / "downstream_tueg"
    compatibility_path = ROOT / "review/EEGPT_TUEV23_INPUT_COMPATIBILITY_20261022.json"
    compatibility = json.loads(compatibility_path.read_text())
    if compatibility.get("status") != "compatible" or (not shared16 and accepted.get("source_sha256") != compatibility["cache_preprocessing_source_sha256"]):
        raise RuntimeError("Accepted cache does not come from the verified compatible preprocessing source")
    for relative, expected in compatibility["eegpt_source_sha256"].items():
        if hashlib.sha256((vendor / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Verified native input consumer changed: {relative}")
    if not shared16 and accepted.get("units") != "uV; division by 100 remains in training engine":
        raise RuntimeError("Accepted cache physical units do not match the native EEGPT engine")
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
    # Actual upstream native AMP branch. Original shell uses DeepSpeed0.4;
    # backend difference is explicit and source training functions stay intact.
    argv += ["--drop_path", "0.2", "--dist_eval"]
    if cli.resume:
        argv += ["--resume", cli.resume]
    original_argv = sys.argv
    sys.argv = [str(vendor / file)] + argv
    try:
        opts, ds_init = native.get_args()
    finally:
        sys.argv = original_argv
    # The scratch setting intentionally does not load a pretrained checkpoint.
    opts.finetune = ""
    channels, samples = (16 if shared16 else 23), 1000
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    state = {}
    original_get_dataset = native.get_dataset
    original_prepare = native.utils.prepare_TUEV_dataset

    def prepare_dataset(_original_path):
        if shared16:
            from paclock_bench.data.contracts import verify_arrays

            class SharedTUEV(Dataset):
                def __init__(self, split):
                    self.x = np.load(data_root / (split+'_signals.npy'), mmap_mode='r')
                    self.y = np.load(data_root / (split+'_labels.npy'), mmap_mode='r')
                    verify_arrays(self.x, self.y, data_root / (split+'_signals.npy'),
                                  data_root / (split+'_labels.npy'))
                    if (self.x.shape != (split_counts[split], 16, 1000) or self.y.ndim != 1
                            or np.any(self.y < 0) or np.any(self.y >= 6)):
                        raise RuntimeError('Shared16 TUEV split shape/labels mismatch: '+split)

                def __len__(self):
                    return len(self.y)

                def __getitem__(self, index):
                    # Stored uV/100 -> physical uV; original engine performs /100.
                    x = torch.from_numpy(np.array(self.x[index], dtype=np.float32) * 100.0)
                    if not torch.isfinite(x).all():
                        raise RuntimeError('Nonfinite shared16 EEG sample')
                    return x, int(self.y[index])

            train_dataset, val_dataset = SharedTUEV('train'), SharedTUEV('val')
            test_dataset = val_dataset if cli.smoke_steps else SharedTUEV('test')
        else:
            train_dataset, test_dataset, val_dataset = original_prepare(str(data_root / "native/processed"))
            # Preserve the exact accepted original filesystem enumeration order.
            for split, dataset in (("train",train_dataset),("val",val_dataset),("test",test_dataset)):
                if set(dataset.files) != set(files[split]):
                    raise RuntimeError("Native files disagree with accepted cohort")
                dataset.files = files[split]
        state["datasets"] = (train_dataset,val_dataset,test_dataset)
        return train_dataset,test_dataset,val_dataset

    native.utils.prepare_TUEV_dataset = prepare_dataset

    def get_dataset(args):
        datasets = original_get_dataset(args)
        sample,label = datasets[0][0]
        if shared16:
            # Recorded bipolar channels are not relabeled as raw monopolar electrodes.
            datasets = (*datasets[:3], None, datasets[4])
        if sample.shape != (channels,1000) or not torch.isfinite(sample).all() or not 0<=int(label)<6:
            raise RuntimeError("Native TUEV sample bridge is invalid")
        return datasets

    native.get_dataset = get_dataset
    original_get_model = native.get_models

    def get_model(args):
        if shared16:
            # Original TUEV constructor; only the learned channel-convolution input
            # width changes. Its twenty latent positions and pretrained encoder stay intact.
            names = ['FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                     'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
            model = native.EEGPTClassifier(num_classes=args.nb_classes, in_channels=16,
                img_size=[len(names), 1000], use_channels_names=names,
                use_chan_conv=True, use_mean_pooling=args.use_mean_pooling)
        else:
            model = original_get_model(args)
        if cli.precision == "bf16":
            model.register_forward_hook(lambda module, inputs, output: output.float())
        if cfg['pretrained']:
            state['pretrained'] = load_released_pretraining(model, cfg)
            if rank == 0:
                (out / 'pretrained_audit.json').write_text(json.dumps(state['pretrained'], indent=2) + '\n')
        state["model"] = model
        state["encoder_probe"] = next(value for name,value in model.target_encoder.named_parameters()
                                       if value.requires_grad and 'qkv.weight' in name)
        return model

    native.get_models = get_model
    original_eval = native.evaluate

    def evaluate(loader, model, device, header="Test:", **kwargs):
        if header.startswith("Test"):
            return {k: 0.0 for k in metrics + ["loss"]}
        return original_eval(loader, model, device, header=header, **kwargs)

    native.evaluate = evaluate
    original_save = native.utils.save_model

    def save_model(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs.get("epoch") == "best" and rank == 0:
            path = out / "selected_model.pt"
            torch.save(
                kwargs["model_without_ddp"].state_dict(), path.with_suffix(".tmp")
            )
            os.replace(path.with_suffix(".tmp"), path)

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
            "input": {"dataset": 'tuev-shared16' if shared16 else accepted["dataset"],
                      "accepted_sha256": accepted_hash, "splits": split_counts,
                      "shape": [channels, 1000], "input_protocol": 'shared16' if shared16 else 'native23',
                      "test_payload_opened": not (shared16 and cli.smoke_steps)},
            "global_batch": batch * 2,
            "input_compatibility": ({'status': 'declared_shared16_adaptation',
                'scope': 'Shared accepted arrays; source hashes below verify original EEGPT code, not equality with native23 preprocessing.',
                'input_scale': 'stored uV/100 -> physical uV -> original engine /100',
                'learned_channel_conv': '16 recorded bipolar inputs -> 20 native latent positions'}
                if shared16 else compatibility),
            "input_compatibility_sha256": hashlib.sha256(compatibility_path.read_bytes()).hexdigest(),
            "native_amp_compatibility_override": True,
            "precision": cli.precision,
            "native_source_sha256": {str(path.relative_to(vendor)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (vendor / file, vendor / "engine_for_finetuning_EEGPT.py",
                              vendor / "Modules/models/EEGPT_mcae_finetune_change_tuev.py")},
            "adaptations": [
                "released native TUEV main, get_models, loader and engine execute directly",
                ("shared accepted16 bipolar inputs; learned input convolution16->20; no channel padding or raw-electrode relabeling" if shared16 else "original23monopolar inputs; no padding, channel relabeling or first-convolution change"),
                ("released pretrained weights strictly loaded; native full finetuning" if cfg["pretrained"] else "no pretrained weights, seed0, all active target encoder weights trainable"),
                "original DeepSpeed0.4 CLI replaced with upstream's native AMP branch; optimizer/schedule/batch retained",
                ("accepted shared array row order retained" if shared16 else "exact accepted original file-order restored after transport"),
                "test evaluation deferred to final validation-selected checkpoint",
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
            val = DataLoader(state["datasets"][1], batch_size=2, num_workers=0)
            x, _ = next(iter(val))
            x = x[:2].to("cuda").float() / 100.0
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
                        "provenance": {"runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                       "config_sha256":hashlib.sha256(Path(cli.config).read_bytes()).hexdigest(),
                                       "accepted_sha256":accepted_hash,"source":receipt},
                        "stats": state["smoke"],
                        "strict_reload_identical": True,
                        "encoder_updated": True,
                        "scratch_encoder_updated": not cfg["pretrained"],
                        "pretrained_identity": pretrained_identity,
                        "pretrained_weights": state.get("pretrained"),
                        "world_size": 2,
                        "per_rank_batch": batch,
                        "precision": cli.precision,
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
        test = DataLoader(state["datasets"][2], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=False)
        with torch.no_grad():
            for x, y in test:
                x = (x.to("cuda").float() / 100.0).reshape(x.shape[0], channels, -1, 200)
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
            "config": cfg,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
