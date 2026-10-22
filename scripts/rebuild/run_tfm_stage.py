"""Train TFM's actual tokenizer or MTP Lightning module on TRAIN split only.

Native modules, optimizer, loss and schedules are imported unchanged. The only
scientific adaptation is the frozen shared data cohort. Full Lightning state
allows wall-clock continuation; a partial stage never exports finished weights.
"""

from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--stage", choices=["tokenizer", "mtp", "finetune"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume")
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=0)
    a = p.parse_args()
    from paclock_bench.training.recipe_runtime import provenance as runtime_provenance
    import torch, yaml, lightning as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, Dataset
    from paclock_bench.paths import vendored, expand
    from paclock_bench.data.datasets import load_manifest, WindowDataset
    from paclock_bench.data.biot_dataset import BIOTWindowDataset
    from paclock_bench.data.contracts import describe_contract
    from scripts.rebuild.native_adaptations import resample_window, recover_uv, extend_channel_embeddings, expand_encoder_state, tfm_stage_input_config

    cfg = yaml.safe_load(Path(a.config).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completed.json").exists():
        raise FileExistsError(
            "TFM stage already completed; reuse artifact, do not overwrite"
        )
    if (
        (out / "last.ckpt").exists()
        and not a.resume
        and not os.environ.get("LOCAL_RANK")
    ):
        raise FileExistsError("Use --resume for existing unfinished stage")
    vendor = Path(vendored("tfm"))
    sys.path.insert(0, str(vendor))
    source = {
        "tokenizer": "tfm_tokenizer_token_learning.py",
        "mtp": "downstream_transformer_masked_token_prediction_pretraining.py",
        "finetune": "downstream_transformer_finetuning.py",
    }[a.stage]
    spec = importlib.util.spec_from_file_location(
        f"cfft_tfm_{a.stage}", vendor / source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    raw_cfg = yaml.safe_load(
        (vendor / "configs/tfm_tokenizer_training_configs.yaml").read_text()
    )
    dataset_name = {
        "chbmit": "CHBMIT",
        "tuev": "TUEV",
        "tuab": "TUAB",
        "iiic": "IIIC",
    }.get(cfg["dataset"], cfg["dataset"].upper())
    recipe = (
        "tokenizer_training_" + dataset_name
        if a.stage == "tokenizer"
        else "finetuning" if a.stage == "finetune" else "mtp_training"
    )
    if recipe not in raw_cfg:
        if not cfg.get("allow_new_task_adaptation"):
            raise ValueError(
                f"No official TFM recipe for {dataset_name}; explicit adaptation required"
            )
        recipe = "tokenizer_training_IIIC"
    training = raw_cfg[recipe]
    data_root = expand(cfg["data_root"])
    manifest = load_manifest(data_root)
    contract = describe_contract(cfg, manifest, data_root, "biot")
    source_rate = int(cfg["sample_rate"])
    target_rate = int(cfg.get("model_sample_rate", source_rate))
    if target_rate != 200:
        raise ValueError("Native TFM STFT needs an explicit200Hz model input")

    class ResampledPhysical(Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        @property
        def shape(self):
            return (*self.base.shape[:-1], self.base.shape[-1] * target_rate // source_rate)
        def __getitem__(self, index):
            x, y = self.base[index]
            x = resample_window(recover_uv(x, manifest), source_rate, target_rate)
            # Original TFM loader: resample first, then channelwise q95.
            x = x / (torch.quantile(x.abs(), .95, dim=-1, keepdim=True) + 1e-8)
            return x, y

    def dataset_for(split):
        if source_rate != target_rate:
            return ResampledPhysical(WindowDataset(data_root, split,
                flatten_sequences=cfg.get("flatten_sequences", False)))
        return BIOTWindowDataset(data_root, split,
            flatten_sequences=cfg.get("flatten_sequences", False),
            input_multiplier=cfg.get("loader_input_multiplier", 1.0),
            already_q95=manifest.get("protocol", {}).get("normalization") == "q95")

    dataset = dataset_for("train")
    channels = dataset.shape[-2]
    if channels > 16 and cfg.get("model_kwargs", {}).get("channel_strategy") != "extend_embeddings":
        raise ValueError("Native TFM >16 channels requires explicit trainable embedding extension")
    train = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 8),
        drop_last=False,
    )
    seed = int(
        (cfg.get("seed", 0) if a.seed is None else a.seed) if a.stage == "finetune" else cfg.get("stage_seed", 5)
    )
    pl.seed_everything(seed, workers=True)
    if cfg.get("tfm_setting") == "multiple" and not cfg.get("tokenizer_path"):
        from scripts.rebuild.tfm_weights import weight_paths

        tokenizer_path, encoder_path = weight_paths(cfg["dataset"], "multiple")
    else:
        tokenizer_path, encoder_path = cfg.get("tokenizer_path", ""), cfg.get(
            "encoder_path", ""
        )
    args = SimpleNamespace(
        dataset_name=dataset_name,
        code_book_size=8192,
        emb_size=64,
        resampling_rate=200,
        random_seed=seed,
        vqvae_pretrained_path=expand(tokenizer_path),
        mem_pretrained_path=expand(encoder_path),
    )
    klass = (
        module.Pl_tfm_tokenizer_token_learning
        if a.stage == "tokenizer"
        else (
            module.Pl_downstream_transformer_finetuning
            if a.stage == "finetune"
            else module.Pl_downstream_transformer_MTP
        )
    )
    devices = int(cfg.get("finetune_devices", 1) if a.stage == "finetune" else cfg.get("stage_devices", 2))
    expected_devices = 1 if a.stage == "finetune" else 2
    if devices != expected_devices:
        raise ValueError(f"Native TFM {a.stage} uses {expected_devices} GPU(s); got {devices}")
    schedule_steps = len(train) // devices
    if schedule_steps < 1:
        raise ValueError("Dataset is too small for native global batch")
    if a.stage in ("mtp", "finetune") and cfg.get("tfm_setting", "single") == "single":
        checkpoint = Path(args.vqvae_pretrained_path)
        completion = checkpoint.parent / "completed.json"
        if not completion.exists():
            raise FileNotFoundError(
                "MTP requires a completed train-only tokenizer stage"
            )
        stage_receipt = json.loads(completion.read_text())
        if (
            stage_receipt.get("stage") != "tokenizer"
            or stage_receipt.get("epochs", 0) < 100
        ):
            raise ValueError("Tokenizer stage was partial or has wrong type")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (
            stage_receipt.get("dataset") != cfg["dataset"]
            or stage_receipt.get("manifest_sha256") != contract["manifest_sha256"]
        ):
            raise ValueError("Tokenizer was trained on a different cohort")
        if stage_receipt.get("export_sha256") != digest:
            raise ValueError("Tokenizer completion receipt does not bind these weights")
    if a.stage == "finetune" and cfg.get("tfm_setting", "single") == "single":
        checkpoint = Path(args.mem_pretrained_path)
        completion = json.loads((checkpoint.parent / "completed.json").read_text())
        if (
            completion.get("dataset") != cfg["dataset"]
            or completion.get("manifest_sha256") != contract["manifest_sha256"]
        ):
            raise ValueError("MTP was trained on a different cohort")
        if (
            completion.get("stage") != "mtp"
            or completion.get("epochs", 0) < 50
            or completion.get("export_sha256")
            != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Supervised TFM requires a completed matching MTP checkpoint"
            )
    if a.stage == "finetune" and cfg.get("tfm_setting") == "multiple":
        audit = cfg.get("pretraining_cohort_review", {})
        if audit.get("known_test_overlap") is True:
            raise ValueError("Released TFM checkpoint has known test-patient overlap")
        if audit.get("status") != "released_checkpoint_reference":
            raise ValueError("Released TFM weights need a documented checkpoint/cohort review")
    embedding_adaptation = None
    if channels > 16 and a.stage in ("mtp", "finetune"):
        original_factory = module.get_tfm_token_classifier_64x4
        def extended_factory(*factory_args, **factory_kwargs):
            return extend_channel_embeddings(original_factory(*factory_args, **factory_kwargs), channels)
        module.get_tfm_token_classifier_64x4 = extended_factory
        if a.stage == "finetune":
            source_checkpoint = Path(args.mem_pretrained_path)
            source_state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
            expanded = expand_encoder_state(source_state, channels)
            if expanded is not source_state:
                adapted = out / "encoder_input_adapted.pth"
                if not adapted.exists(): torch.save(expanded, adapted)
                args.mem_pretrained_path = str(adapted)
                embedding_adaptation = {"source_sha256":hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
                    "channels":channels, "retained_rows":source_state["channel_embed.weight"].shape[0],
                    "new_rows":"random trainable nn.Embedding initialization", "checkpoint":str(adapted)}
    dataset_params = {
        "num_classes": 1 if cfg["num_classes"] == 2 else cfg["num_classes"],
        "classification_task": "binary" if cfg["num_classes"] == 2 else "multi_class",
    }
    model = (
        klass(args, training, str(out), schedule_steps, dataset_params)
        if a.stage == "finetune"
        else klass(args, training, str(out), schedule_steps)
    )
    val = None
    if a.stage == "finetune":
        # Check every pretrained encoder tensor. The upstream constructor uses
        # strict=False, which must not hide accidental missing encoder blocks.
        state = torch.load(
            args.mem_pretrained_path, map_location="cpu", weights_only=False
        )
        missing, unexpected = model.tfm_token.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith("classification_head")},
            strict=False,
        )
        if unexpected or any(not k.startswith("classification_head") for k in missing):
            raise RuntimeError(f"TFM MTP checkpoint mismatch: {missing}, {unexpected}")
        val_dataset = dataset_for("val")
        val = DataLoader(
            val_dataset,
            batch_size=training["batch_size"],
            num_workers=cfg.get("num_workers", 8),
            shuffle=False,
            drop_last=False,
        )
    if torch.cuda.device_count() < devices:
        raise RuntimeError(f"TFM native stage needs {devices} allocated CUDA devices")
    if a.resume and not Path(a.resume).is_file():
        raise FileNotFoundError(a.resume)
    callback = ModelCheckpoint(
        dirpath=str(out),
        filename="best_model" if a.stage == "finetune" else "stage-last",
        monitor=(
            ("val_roc_auc" if cfg["num_classes"] == 2 else "val_cohen_kappa")
            if a.stage == "finetune"
            else None
        ),
        mode="max",
        save_last=True,
        save_top_k=1 if a.stage == "finetune" else 0,
        save_weights_only=False,
    )

    class AuditUpdates(Callback):
        def __init__(self):
            self.grad_steps = 0
            self.initial = None
            self.parameter = None

        def on_train_start(self, trainer, module):
            trained = module.vqvae if a.stage == "tokenizer" else module.tfm_token
            self.parameter = next(p for p in trained.parameters() if p.requires_grad)
            self.initial = self.parameter.detach().clone()

        def on_after_backward(self, trainer, module):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("TFM native stage missing or nonfinite gradients")
            self.grad_steps += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is not None and not torch.isfinite(loss).all():
                raise RuntimeError("TFM native stage nonfinite loss")

    audit = AuditUpdates()
    max_time = None if not a.max_hours else {"seconds": int(a.max_hours * 3600)}
    trainer = pl.Trainer(
        default_root_dir=str(out),
        devices=devices,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        callbacks=[callback, audit],
        max_epochs=training["num_pretrain_epochs"],
        max_steps=a.smoke_steps or -1,
        logger=CSVLogger(str(out), name="logs"),
        deterministic=True,
        precision="32-true",
        max_time=max_time,
        num_sanity_val_steps=0,
        limit_val_batches=0 if a.smoke_steps else 1.0,
    )
    receipt = {
        "config": cfg,
        "stage": a.stage,
        "source": source,
        "source_sha256": hashlib.sha256((vendor / source).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
        "native_recipe": training,
        "stage_seed": seed,
        "data_contract": contract,
        "world_size": devices,
        "model_sample_rate": target_rate,
        "embedding_adaptation": embedding_adaptation,
        "pretraining_cohort_review": cfg.get("pretraining_cohort_review"),
        "per_device_batch": training["batch_size"],
        "global_batch": training["batch_size"] * devices,
        "schedule_niter_per_ep": schedule_steps,
        "schedule_note": "Upstream uses floor(full loader length / device count); native LambdaLR clamps to its final schedule value beyond the array.",
        "split_used": "train+validation" if a.stage == "finetune" else "train",
        "tokenizer_runtime": "Preserves upstream recursive train/eval mode including EMA codebook updates in training; tokenizer has no downstream gradient optimizer.",
        "final_evaluation": "Full sequential test cohort on rank0, not a mean of rank-level metrics.",
        "resume": a.resume,
    }
    if trainer.is_global_zero:
        (out / "provenance.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n"
        )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val, ckpt_path=a.resume)
    # All ranks participate in strategy checkpoint synchronization.
    trainer.save_checkpoint(str(out / "last.ckpt"))
    if trainer.is_global_zero:
        if a.smoke_steps:
            if trainer.global_step < a.smoke_steps or audit.grad_steps < a.smoke_steps:
                raise RuntimeError(
                    "TFM smoke did not complete requested backward steps"
                )
            if a.smoke_steps > 1 and torch.equal(audit.initial, audit.parameter):
                raise RuntimeError("TFM optimizer did not update a trainable parameter")
            saved = torch.load(
                out / "last.ckpt", map_location="cpu", weights_only=False
            )["state_dict"]
            model.load_state_dict(saved, strict=True)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved[key], rtol=0, atol=0)
            (out / "smoke.json").write_text(
                json.dumps(
                    {
                        "passed": trainer.global_step >= a.smoke_steps,
                        "status": "passed",
                        "test_evaluated": False,
                        "config": cfg,
                        "checkpoint_reload": "passed",
                        "provenance": runtime_provenance(
                            a.config,
                            tfm_stage_input_config(cfg, a.stage),
                            {"manifest": manifest, "input_contract": contract},
                        ),
                        "steps": trainer.global_step,
                        "stage": a.stage,
                        "finite_grad_steps": audit.grad_steps,
                        "strict_reload_identical": True,
                        "dataset": cfg["dataset"],
                        "manifest_sha256": contract["manifest_sha256"],
                        "source_sha256": receipt["source_sha256"],
                        "config_sha256": receipt["config_sha256"],
                    }
                )
                + "\n"
            )
            return
        expected_steps = int(
            training["num_pretrain_epochs"] * trainer.num_training_batches
        )
        if (
            trainer.current_epoch < training["num_pretrain_epochs"]
            or trainer.global_step < expected_steps
        ):
            (out / "paused.json").write_text(
                json.dumps(
                    {
                        "epoch": trainer.current_epoch,
                        "required": training["num_pretrain_epochs"],
                        "expected_steps": expected_steps,
                        "actual_steps": trainer.global_step,
                        "resume": str(out / "last.ckpt"),
                    }
                )
                + "\n"
            )
            raise SystemExit(75)
        if a.stage == "finetune":
            import numpy as np
            from paclock_bench.training.metrics import compute_metrics

            best = Path(callback.best_model_path)
            model.load_state_dict(
                torch.load(best, map_location="cpu", weights_only=False)["state_dict"],
                strict=True,
            )
            model.to("cuda:0").eval()
            test_dataset = dataset_for("test")
            test = DataLoader(
                test_dataset,
                batch_size=training["batch_size"],
                num_workers=cfg.get("num_workers", 8),
                shuffle=False,
                drop_last=False,
            )
            truth = []
            scores = []
            with torch.no_grad():
                for x, y in test:
                    pred = model(x.to(model.device))
                    truth.append(y.numpy())
                    scores.append(pred.float().cpu().numpy())
            labels = np.concatenate(truth)
            logits = np.concatenate(scores)
            np.savez_compressed(
                out / "test_predictions.npz", labels=labels, logits=logits
            )
            (out / "result.json").write_text(
                json.dumps(
                    {
                        "name": cfg["name"],
                        "seed": seed,
                        "config": cfg,
                        "test": compute_metrics(labels, logits, cfg["num_classes"]),
                        "selected_checkpoint": str(best),
                        "selection_metric": callback.monitor,
                    },
                    indent=2,
                )
                + "\n"
            )
        state = (
            model.vqvae.state_dict()
            if a.stage == "tokenizer"
            else model.tfm_token.state_dict()
        )
        filename = (
            "tfm_tokenizer_last.pth"
            if a.stage == "tokenizer"
            else (
                "tfm_encoder_best.pth"
                if a.stage == "finetune"
                else "tfm_encoder_mtp.pth"
            )
        )
        torch.save(state, out / filename)
        (out / "completed.json").write_text(
            json.dumps(
                {
                    "stage": a.stage,
                    "seed": seed,
                    "epochs": trainer.current_epoch,
                    "provenance": runtime_provenance(a.config, tfm_stage_input_config(cfg,a.stage), {"manifest":manifest,"input_contract":contract}),
                    "dataset": cfg["dataset"],
                    "manifest_sha256": contract["manifest_sha256"],
                    "source_sha256": receipt["source_sha256"],
                    "config_sha256": receipt["config_sha256"],
                    "expected_steps": expected_steps,
                    "actual_steps": trainer.global_step,
                    "export": str(out / filename),
                    "export_sha256": hashlib.sha256(
                        (out / filename).read_bytes()
                    ).hexdigest(),
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
