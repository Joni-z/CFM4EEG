"""Invoke the pinned LaBraM main/engine with explicit initialization and data bridges.

TUEV explicitly selects shared16 bipolar arrays or the original23 pickle
cohort. Bipolar inputs use fresh task-specific spatial slots, with no
monopolar electrode identity claim. Stored normalization is inverted before
the unchanged original engine divides physical microvolts by100.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

COMMIT = "c431221e6cfd23dbfa9950e0180682fb322b0548"
SOURCE = {
    "run_class_finetuning.py": "5aeeb1290ed5a1e2be0ba7244181864d86ebd8d3f67489c46f27979b50fd57dc",
    "engine_for_finetuning.py": "7e0e8dad3f208645ac3c966ef356433b84def3f371852daf2858778562726f92",
    "utils.py": "c5ad729bd57ab31e04572f426d94f7fb164ef0fbfd1f7076da6bf8dfbd833e32",
    "modeling_finetune.py": "7b3514c9a661ebd417a2dd9834a9f2fab0a2e518c8212c73ebebaaa165c29d88",
    "optim_factory.py": "7f82eeef697f6abc5a1c76e5e6be0087a42c95656d89a6b7d46b8216d246e1ab",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--vendor", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-proof", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if cfg["seed"] != 0 or not isinstance(cfg.get("pretrained"), bool):
        raise ValueError("An explicit scratch/pretrained seed0 setting is required")
    pretrained = cfg["pretrained"]
    weights = Path(os.path.expandvars(os.path.expanduser(cfg["pretrained_path"]))) if pretrained else None
    if weights is not None and (not weights.is_file() or sha(weights) != cfg["pretrained_sha256"]):
        raise ValueError("Released pretrained checkpoint missing or hash mismatch")
    root = Path(os.path.expandvars(os.path.expanduser(cfg["data_root"])))
    accepted = json.loads((root / "ACCEPTED.json").read_text())
    if accepted.get("status") != "accepted":
        raise ValueError("Unaccepted dataset")
    for name, expected in SOURCE.items():
        if sha(a.vendor / name) != expected:
            raise ValueError("Pinned source drift: " + name)
    identity = dict(config_sha256=sha(a.config), runner_sha256=sha(__file__),
                    accepted_sha256=sha(root / "ACCEPTED.json"), source_sha256=SOURCE,
                    vendor_commit=COMMIT)
    if pretrained:
        identity["pretrained_sha256"] = cfg["pretrained_sha256"]
    if not a.smoke:
        if a.smoke_proof is None:
            raise ValueError("Formal run requires successful smoke proof")
        proof = json.loads(a.smoke_proof.read_text())
        if proof.get("status") != "passed" or proof["identity"] != identity:
            raise ValueError("Smoke proof does not match this exact run")
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError("Preserving existing output " + str(a.out))
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.vendor))
    import numpy as np
    import torch
    from torch.utils.data import Dataset, Subset, DataLoader
    import run_class_finetuning as native
    import utils
    if pretrained:
        # Torch>=2.6 defaults to tensor-only loading. The SHA-verified author
        # file also contains legacy NumPy/Namespace metadata, irrelevant to
        # initialization; expose its unchanged model tensors to native loading.
        released = torch.load(weights, map_location="cpu", weights_only=False)
        if not isinstance(released.get("model"), dict) or not all(
                isinstance(v, torch.Tensor) for v in released["model"].values()):
            raise ValueError("Expected the released tensor model dictionary")
        bridge = a.out / "pretrained_tensor_bridge.pth"
        torch.save({"model": released["model"]}, bridge)
        del released
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one physical GPU; official update_freq preserves global batch")
    final_data = {}
    if cfg["dataset"] == "TUEV":
        transfer = json.loads((root / "TRANSFER_VERIFIED.json").read_text())
        if transfer["accepted_sha256"] != identity["accepted_sha256"]:
            raise ValueError("Native transfer acceptance mismatch")
        original = utils.prepare_TUEV_dataset
        utils.prepare_TUEV_dataset = lambda ignored: original(str(root / "native/processed"))
        original_get = native.get_dataset
        def get_dataset(args):
            tr, te, va, names, metrics = original_get(args)
            if a.smoke:
                tr, va = Subset(tr, range(256)), Subset(va, range(96))
                te = va  # no held-out test payload is read by the smoke
            final_data.update(test=te, names=names, metrics=metrics)
            return tr, te, va, names, metrics
    elif cfg["dataset"] in ("CHBMIT", "ADFD", "TUEV_SHARED16"):
        manifest = json.loads((root / "manifest.json").read_text())
        expected_manifest = accepted["manifest_sha256"]
        is_chb = cfg["dataset"] == "CHBMIT"
        shared_tuev = cfg["dataset"] == "TUEV_SHARED16"
        ready = json.loads((root / "READY.json").read_text()) if is_chb else {"manifest_sha256": expected_manifest}
        if sha(root / "manifest.json") != expected_manifest or ready["manifest_sha256"] != expected_manifest:
            raise ValueError("Manifest/build completion mismatch")
        norm = "q95" if is_chb else "div100"
        n_channels, n_classes = (16, 6) if shared_tuev else ((16, 2) if is_chb else (19, 3))
        n_samples = 1000 if shared_tuev else 2000
        if manifest["protocol"]["sample_rate"] != 200 or manifest["protocol"]["normalization"] != norm:
            raise ValueError("Unexpected stored units or sample rate")
        if shared_tuev:
            if accepted.get("dataset") != "tuev" or manifest.get("dataset") != "tuev":
                raise ValueError("Shared16 TUEV acceptance required")
            expected_files = {f"{s}_{k}.npy" for s in ("train", "val", "test") for k in ("signals", "labels")}
            if not expected_files.issubset(accepted.get("files", {})):
                raise ValueError("Acceptance must cover all six shared16 arrays")
            for name in expected_files:
                actual, recorded = (root/name).stat(), accepted["files"][name]
                if (actual.st_size != recorded["size"] or actual.st_mtime_ns != recorded["mtime_ns"]
                        or len(recorded.get("sha256", "")) != 64):
                    raise ValueError("Accepted shared16 file changed: " + name)
        class SharedCHB(Dataset):
            def __init__(self, split):
                self.x = np.load(root / (split + "_signals.npy"), mmap_mode="r")
                self.y = np.load(root / (split + "_labels.npy"), mmap_mode="r")
                self.scale = np.load(root / (split + "_q95_denominators.npy"), mmap_mode="r") if is_chb else None
                if self.x.shape[1:] != (n_channels, n_samples) or len(self.y) != len(self.x):
                    raise ValueError("Array shape mismatch")
                if shared_tuev and (self.x.dtype != np.float32 or self.y.dtype.kind not in "iu"
                        or np.any(self.y < 0) or np.any(self.y >= n_classes)
                        or len(self.x) != {"train":68445,"val":15487,"test":29421}[split]):
                    raise ValueError("Shared16 TUEV dtype/labels/cohort mismatch")
                if is_chb and self.scale.shape != (len(self.x), n_channels, 1):
                    raise ValueError("CHB scale mismatch")
            def __len__(self):
                return len(self.y)
            def __getitem__(self, index):
                uv = (self.x[index].astype(np.float64) * (self.scale[index] if is_chb else 100.)).astype(np.float32)
                if not np.isfinite(uv).all():
                    raise ValueError("Nonfinite recovered microvolts")
                return torch.from_numpy(uv), int(self.y[index])
        original_chans = utils.get_input_chans
        names = ["SCRATCH_CHANNEL_SLOT_" + str(i) for i in range(n_channels)]
        if pretrained:
            if not (is_chb or shared_tuev) or cfg.get("channel_initialization") != "fresh_bipolar_slots":
                raise ValueError("Pretrained bipolar data needs fresh task-specific spatial embeddings")
            names = list(manifest["protocol"]["channels"])
            if len(names) != 16 or not all("-" in n for n in names):
                raise ValueError("Expected sixteen named bipolar derivations")
        def slots(ch_names):
            return list(range(n_channels + 1)) if ch_names == names else original_chans(ch_names)
        utils.get_input_chans = slots
        def get_dataset(args):
            args.nb_classes = 1 if is_chb else n_classes
            tr, va = SharedCHB("train"), SharedCHB("val")
            if a.smoke:
                # Include both classes in validation to exercise native binary metrics.
                ids = [int(np.flatnonzero(va.y == label)[0]) for label in range(n_classes)]
                va = Subset(va, ids * (96 // n_classes))
                tr = Subset(tr, range(256))
                te = va
            else:
                te = SharedCHB("test")
            metrics = (["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"] if is_chb else ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"])
            final_data.update(test=te, names=names, metrics=metrics)
            return tr, te, va, names, metrics
    else:
        raise ValueError("Unsupported native bridge")
    native.get_dataset = get_dataset
    argv = ["--dataset", "TUEV" if cfg["dataset"] in ("TUEV", "TUEV_SHARED16") else "TUAB",
            "--output_dir", str(a.out), "--model", "labram_base_patch200_200",
            "--seed", "0", "--batch_size", str(cfg["batch_size"]),
            "--update_freq", "1" if a.smoke else str(cfg["update_freq"]),
            "--epochs", "1" if a.smoke else str(cfg["epochs"]),
            "--warmup_epochs", "0" if a.smoke else str(cfg["warmup_epochs"]),
            "--lr", str(cfg["lr"]), "--min_lr", "1e-6",
            "--layer_decay", str(cfg["layer_decay"]), "--weight_decay", "0.05",
            "--num_workers", "0" if a.smoke else str(cfg["num_workers"]),
            "--save_ckpt_freq", "5", "--drop_path", "0.1", "--smoothing", "0.1",
            "--disable_rel_pos_bias", "--abs_pos_emb", "--disable_qkv_bias", "--no_auto_resume"]
    if pretrained:
        argv += ["--finetune", str(bridge)]
    sys.argv = ["run_class_finetuning.py"] + argv
    args, ds_init = native.get_args()
    if args.resume or ds_init or bool(args.finetune) != pretrained:
        raise ValueError("Unexpected resume, initialization or DeepSpeed setting")
    captured = {}
    original_models = native.get_models
    def capture(args):
        m = original_models(args)
        captured["model"] = m
        captured["head_before"] = m.head.weight.detach().cpu().clone()
        if pretrained and cfg.get("channel_initialization") == "fresh_bipolar_slots":
            captured["fresh_spatial"] = m.pos_embed[:, 1:17].detach().clone()
        return m
    native.get_models = capture
    if pretrained:
        captured["optimizer_updates"] = 0
        original_optimizer = native.create_optimizer
        def counted_optimizer(*args, **kwargs):
            optimizer = original_optimizer(*args, **kwargs)
            def updated(*_args, **_kwargs):
                captured["optimizer_updates"] += 1
            optimizer.register_step_post_hook(updated)
            return optimizer
        native.create_optimizer = counted_optimizer
        original_load = utils.load_state_dict
        def checked_load(model, state_dict, prefix="", **kwargs):
            """Validate the native student extraction before its permissive loader."""
            target = model.state_dict()
            missing = set(target) - set(state_dict)
            unexpected = set(state_dict) - set(target)
            allowed_missing = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
            allowed_extra = {"mask_token", "lm_head.weight", "lm_head.bias", "norm.weight", "norm.bias"}
            if prefix or missing != allowed_missing or not unexpected.issubset(allowed_extra):
                raise ValueError(f"Unexpected pretrained coverage: missing={missing}, extra={unexpected}")
            loaded = sorted(set(target) & set(state_dict))
            if any(target[k].shape != state_dict[k].shape for k in loaded):
                raise ValueError("Pretrained backbone shape mismatch")
            original_load(model, state_dict, prefix=prefix, **kwargs)
            if not all(torch.equal(model.state_dict()[k].cpu(), state_dict[k].cpu()) for k in loaded):
                raise RuntimeError("Native loader did not restore every expected backbone tensor")
            if cfg.get("channel_initialization") == "fresh_bipolar_slots":
                with torch.no_grad():
                    model.pos_embed[:, 1:17].copy_(captured["fresh_spatial"])
            captured["backbone_before"] = model.blocks[-1].attn.qkv.weight.detach().cpu().clone()
            audit = dict(checkpoint_sha256=cfg["pretrained_sha256"], loaded_tensors=loaded,
                         loaded_numel=sum(target[k].numel() for k in loaded),
                         native_fresh_tensors=sorted(missing), discarded_pretraining_tensors=sorted(unexpected),
                         channel_policy=cfg.get("channel_initialization", "native_monopolar_positions"),
                         fresh_spatial_rows=list(range(1,17)) if cfg.get("channel_initialization") == "fresh_bipolar_slots" else [],
                         exact_loaded_tensor_equality=True)
            (a.out / "pretrained_load.json").write_text(json.dumps(audit, indent=2))
        utils.load_state_dict = checked_load
        # Native selection is validation accuracy; held-out testing is deferred.
        original_evaluate = native.evaluate
        def deferred_test(loader, model, device, header="Test:", **kwargs):
            if header.startswith("Test"):
                return {key: 0. for key in final_data["metrics"] + ["loss"]}
            return original_evaluate(loader, model, device, header=header, **kwargs)
        native.evaluate = deferred_test
    (a.out / "provenance.json").write_text(json.dumps(dict(identity=identity, config=cfg,
        args=vars(args), recipe_scope=cfg["recipe_scope"], input_scope=cfg["input_scope"],
        smoke=a.smoke, test_payload_read=not a.smoke), indent=2))
    native.main(args, ds_init)
    model = captured["model"]
    checkpoint = a.out / ("checkpoint-best.pth" if pretrained and not a.smoke else "checkpoint.pth")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if not all(torch.isfinite(t).all().item() for t in model.state_dict().values()):
        raise RuntimeError("Nonfinite trained state")
    if torch.equal(captured["head_before"], model.head.weight.detach().cpu()):
        raise RuntimeError("Native optimizer made no classifier update")
    proof = dict(status="passed", identity=identity, checkpoint_strict_reload=True,
                 classifier_updated=True, finite_model=True, smoke=a.smoke)
    if pretrained:
        if captured["optimizer_updates"] < 1:
            raise RuntimeError("Native scaler skipped every optimizer update")
        if torch.equal(captured["backbone_before"], model.blocks[-1].attn.qkv.weight.detach().cpu()):
            raise RuntimeError("Pretrained backbone did not update")
        proof["pretrained_backbone_loaded_and_updated"] = True
        proof["optimizer_updates"] = captured["optimizer_updates"]
        outputs, labels, native_predictions = [], [], []
        loader = DataLoader(final_data["test"], batch_size=int(1.5 * cfg["batch_size"]),
                            shuffle=False, num_workers=0 if a.smoke else cfg["num_workers"], pin_memory=True)
        class CaptureLabels:
            def __len__(self):
                return len(loader)
            def __iter__(self):
                for x, y in loader:
                    labels.append(y.cpu())
                    yield x, y
        def capture_predictions(module, inp, output):
            outputs.append(output.detach().cpu())
            # Match the native GPU sigmoid exactly before moving to CPU.
            native_predictions.append((torch.sigmoid(output) if args.nb_classes == 1 else output).detach().cpu())
        handle = model.register_forward_hook(capture_predictions)
        scores = original_evaluate(CaptureLabels(), model, torch.device("cuda"), header="Export Validation:" if a.smoke else "Final Test:",
                                   ch_names=final_data["names"], metrics=final_data["metrics"],
                                   is_binary=args.nb_classes == 1)
        handle.remove()
        raw_logits = torch.cat(outputs)
        native_values = torch.cat(native_predictions).numpy()
        y_true = torch.cat(labels).numpy()
        probabilities = native_values.reshape(-1) if args.nb_classes == 1 else torch.softmax(raw_logits.float(), dim=-1).numpy()
        recomputed = utils.get_metrics(native_values, y_true.reshape(-1,1) if args.nb_classes == 1 else y_true,
                                       final_data["metrics"], args.nb_classes == 1, 0.5)
        if any(not np.isclose(recomputed[k],scores[k],rtol=0.,atol=1e-12) for k in final_data["metrics"]):
            raise RuntimeError("Saved native predictions do not reproduce native evaluation metrics")
        proof["prediction_export_recomputed"] = True
        np.savez_compressed(a.out / ("smoke_validation_scores.npz" if a.smoke else "test_scores.npz"),
                            y_true=y_true, logits=raw_logits.float().numpy(), y_prob=probabilities,
                            native_predictions=native_values)
        (a.out / ("smoke_export.json" if a.smoke else "result.json")).write_text(json.dumps(dict(config=cfg, seed=0, metrics=scores,
            pretrained_load=json.loads((a.out/"pretrained_load.json").read_text()),
            evaluation_split="validation" if a.smoke else "test",
            checkpoint_selection="smoke_terminal" if a.smoke else "native_validation_accuracy",
            selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=sha(checkpoint)), indent=2))
    (a.out / ("smoke_pass.json" if a.smoke else "native_completed.json")).write_text(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
