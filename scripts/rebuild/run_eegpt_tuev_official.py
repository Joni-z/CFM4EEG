"""Run pinned EEGPT TUEV main/model/engine with the original23monopolar input.

The upstream native AMP branch is used instead of the released shell's
DeepSpeed0.4 backend. Optimizer, schedules, batch, epochs, distributed validation,
model constructor and pickle reader remain the upstream implementations.
"""

from __future__ import annotations
import argparse, hashlib, importlib.util, itertools, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class SmokeFinished(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--smoke-steps", type=int, default=0)
    cli = parser.parse_args()
    import numpy as np
    import torch, yaml
    from torch.utils.data import Dataset, DataLoader
    from paclock_bench.paths import vendored
    from paclock_bench.training.metrics import compute_metrics

    cfg = yaml.safe_load(Path(cli.config).read_text())
    ds = cfg["dataset"]
    if ds != "tuev" or cfg.get("pretrained") is not False:
        raise ValueError("Only native23 TUEV from scratch is permitted")
    data_root = Path(os.path.expandvars(cfg["data_root"]))
    accepted_file = data_root / "ACCEPTED.json"
    accepted = json.loads(accepted_file.read_text())
    transfer = json.loads((data_root / "TRANSFER_VERIFIED.json").read_text())
    accepted_hash = hashlib.sha256(accepted_file.read_bytes()).hexdigest()
    if (accepted.get("status") != "accepted" or accepted.get("dataset") != "TUEV-v2.0.1-native23-LaBraM"
            or transfer.get("accepted_sha256") != accepted_hash):
        raise RuntimeError("Native23 data/transfer acceptance disagrees")
    order_file = data_root / "SPLIT_FILES.json"
    if hashlib.sha256(order_file.read_bytes()).hexdigest() != accepted["metadata"]["SPLIT_FILES.json"]:
        raise RuntimeError("Native original split/order metadata changed")
    files = json.loads(order_file.read_text())
    if {split:len(names) for split,names in files.items()} != {"train":68910,"val":15022,"test":28305}:
        raise RuntimeError("Wrong native23 cohort")
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
    if compatibility.get("status") != "compatible" or accepted.get("source_sha256") != compatibility["cache_preprocessing_source_sha256"]:
        raise RuntimeError("Accepted cache does not come from the verified compatible preprocessing source")
    for relative, expected in compatibility["eegpt_source_sha256"].items():
        if hashlib.sha256((vendor / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Verified native input consumer changed: {relative}")
    if accepted.get("units") != "uV; division by 100 remains in training engine":
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
    channels, samples = 23, 1000
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    state = {}
    original_get_dataset = native.get_dataset
    original_prepare = native.utils.prepare_TUEV_dataset

    def prepare_dataset(_original_path):
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
        if sample.shape != (23,1000) or not torch.isfinite(sample).all() or not 0<=int(label)<6:
            raise RuntimeError("Native TUEV sample bridge is invalid")
        return datasets

    native.get_dataset = get_dataset
    original_get_model = native.get_models

    def get_model(args):
        model = original_get_model(args)
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
            "input": {"dataset":accepted["dataset"],"accepted_sha256":accepted_hash,
                      "splits":{k:v["rows"] for k,v in accepted["splits"].items()},"shape":[23,1000]},
            "global_batch": batch * 2,
            "input_compatibility": compatibility,
            "input_compatibility_sha256": hashlib.sha256(compatibility_path.read_bytes()).hexdigest(),
            "native_amp_compatibility_override": True,
            "native_source_sha256": {str(path.relative_to(vendor)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (vendor / file, vendor / "engine_for_finetuning_EEGPT.py",
                              vendor / "Modules/models/EEGPT_mcae_finetune_change_tuev.py")},
            "adaptations": [
                "released native TUEV main, get_models, loader and engine execute directly",
                "original23monopolar inputs; no padding, channel relabeling or first-convolution change",
                "no pretrained weights, seed0, all active target encoder weights trainable",
                "original DeepSpeed0.4 CLI replaced with upstream's native AMP branch; optimizer/schedule/batch retained",
                "exact accepted original file-order restored after transport",
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
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                reference = model(x).float()
            path = out / "smoke_checkpoint.pt"
            torch.save(model.state_dict(), path)
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False), strict=True
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
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
                        "scratch_encoder_updated": True,
                        "world_size": 2,
                        "per_rank_batch": batch,
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
                with torch.autocast("cuda", dtype=torch.float16):
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
            "config": cfg,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
