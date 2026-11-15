"""Frozen feature hooks on unchanged project or pinned author implementations."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenReadout:
    def __init__(self, spec: dict, input_shape: tuple[int, int], device: str):
        self.kind, self.device = spec["kind"], device
        scope = spec.get("readout_scope", "all")
        if scope not in ("all", "tokenizer"):
            raise ValueError("readout_scope must be all or tokenizer")
        self.tokenizer_only = scope == "tokenizer"
        if self.tokenizer_only and self.kind != "cfft":
            raise ValueError("Tokenizer-only execution currently supports CfFT")
        checkpoint = Path(spec["checkpoint"])
        if sha256(checkpoint) != spec["checkpoint_sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.training_config = None
        self.source_hashes = {}
        if self.kind == "cfft":
            from paclock_bench.models.build import build_model
            cfg = saved["config"]
            if cfg["seed"] != spec.get("seed", 0) or cfg["model"] != "paclock":
                raise ValueError("CfFT checkpoint seed differs from the declared job seed")
            if cfg["model_kwargs"].get("tokenizer_mode") != "pac_interaction":
                raise ValueError("This launch evaluates original measured-rotation CfFT")
            if cfg["dataset"] != spec["dataset"]:
                raise ValueError("Checkpoint dataset mismatch")
            self.training_config = cfg
            self.model = build_model({k: v for k, v in cfg.items() if k != "checkpoint"}, input_shape)
            self.model.load_state_dict(saved["model"], strict=True)
            stem = self.model.frontend
            layers = () if self.tokenizer_only else self.model.encoder.blocks
            root = Path(__file__).resolve().parents[1] / "models"
            self.source_hashes = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*.py")}
        else:
            vendor = Path(spec["vendor"])
            for relative, expected in spec["source_sha256"].items():
                actual = sha256(vendor / relative)
                if actual != expected:
                    raise ValueError("Author source drift: " + relative)
                self.source_hashes[relative] = actual
            sys.path.insert(0, str(vendor))
            if self.kind == "cbramod":
                module = importlib.import_module("models.model_for_" + ("chb" if spec["dataset"] == "chbmit" else "tuev"))
                params = SimpleNamespace(use_pretrained_weights=False, classifier="all_patch_reps",
                                         dropout=.1, num_of_classes=6, cuda=0)
                self.model = module.Model(params)
                state = saved.get("model", saved)
                self.model.load_state_dict(state, strict=True)
                stem, layers = self.model.backbone.patch_embedding, self.model.backbone.encoder.layers
            elif self.kind == "labram":
                native = importlib.import_module("modeling_finetune")
                self.model = native.labram_base_patch200_200(
                    num_classes=spec["num_classes"], drop_rate=0., drop_path_rate=.1,
                    attn_drop_rate=0., drop_block_rate=None, use_mean_pooling=True,
                    init_scale=.001, use_rel_pos_bias=False, use_abs_pos_emb=True,
                    init_values=.1, qkv_bias=False)
                self.model.load_state_dict(saved["model"], strict=True)
                stem, layers = self.model.patch_embed, self.model.blocks
            elif self.kind == "tfm":
                from paclock_bench.probing.features import load_tfm_tokenizer
                tokenizer_path = Path(spec["tokenizer_checkpoint"])
                if sha256(tokenizer_path) != spec["tokenizer_checkpoint_sha256"]:
                    raise ValueError("TFM tokenizer checkpoint SHA256 mismatch")
                self.tokenizer, _ = load_tfm_tokenizer(vendor, tokenizer_path, inference_only=True)
                native = importlib.import_module("models.tfm_token")
                self.model = native.get_tfm_token_classifier_64x4(
                    n_classes=spec["num_classes"], code_book_size=8192, emb_size=64)
                self.model.load_state_dict(saved, strict=True)
                if input_shape[0] != 16:
                    raise ValueError("Released TFM classifier expects its native 16-channel montage")
                self.tfm_dataset = spec["dataset"]
            else:
                raise ValueError("Unknown model: " + self.kind)
        del saved
        self.model.eval().requires_grad_(False).to(device)
        self.features = {}
        if self.kind == "tfm":
            self.tokenizer.eval().requires_grad_(False).to(device)
            executor = self.model.LAT.layers
            if type(executor).__name__ != "SequentialSequence" or len(executor.layers) != 4:
                raise ValueError("Expected author's four-layer non-reversible TFM encoder")
            self.endpoints = ["tokenizer"] + [f"layer_{i+1:02d}" for i in range(4)]
            # The package returns attention/FF deltas inside each residual block.
            # The next block's input is the complete preceding block's output.
            self.handles = [executor.layers[i+1][0].register_forward_pre_hook(
                            self._input_hook(self.endpoints[i+1])) for i in range(3)]
            self.handles.append(self.model.LAT.register_forward_hook(self._hook("layer_04")))
        else:
            self.endpoints = ["tokenizer"] + [f"layer_{i+1:02d}" for i in range(len(layers))]
            self.handles = [stem.register_forward_hook(self._hook("tokenizer"))]
            self.handles += [layer.register_forward_hook(self._hook(name))
                             for layer, name in zip(layers, self.endpoints[1:])]

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim < 3 or not torch.isfinite(output).all():
                raise ValueError("Invalid representation at " + name)
            self.features[name] = output.mean(tuple(range(1, output.ndim - 1))).double().cpu().numpy()
        return capture

    def _input_hook(self, name: str):
        capture = self._hook(name)
        def before(module, inputs):
            capture(module, None, inputs[0])
        return before

    @torch.inference_mode()
    def __call__(self, stored: torch.Tensor, denominators=None):
        self.features.clear()
        if self.kind == "cfft":
            if self.tokenizer_only:
                self.model.frontend(stored.to(self.device))
            else:
                self.model(stored.to(self.device))
        elif self.kind == "tfm":
            from paclock_bench.probing.features import get_stft_torch
            physical = stored.double().numpy()
            if self.tfm_dataset == "chbmit":
                if denominators is None:
                    raise ValueError("CHB-MIT TFM input requires its physical-unit sidecar")
                physical = physical * denominators
            elif self.tfm_dataset == "tuev":
                # Match BIOTWindowDataset: float32 multiply before NumPy q95.
                physical = stored.numpy() * 100.0
            else:
                raise ValueError("This TFM launch supports only declared TUEV/CHB-MIT units")
            # Author datasets/data_loaders.py: per-channel linear q95 + 1e-8.
            normalized = physical / (np.quantile(np.abs(physical), .95, axis=-1,
                                                 method="linear", keepdims=True) + 1e-8)
            x = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            n, channels, length = x.shape
            spectrum = get_stft_torch(x).reshape(n * channels, 100, -1)
            quantized, codes, _ = self.tokenizer.tokenize(spectrum, x.reshape(n * channels, length))
            if quantized.ndim != 3 or quantized.shape[0] != n * channels:
                raise ValueError("Unexpected native TFM quantized-embedding shape")
            tokens = quantized.reshape(n, channels, -1, quantized.shape[-1])
            self.features["tokenizer"] = tokens.mean((1, 2)).double().cpu().numpy()
            self.model(codes.reshape(n, channels, -1), num_ch=channels)
        else:
            x = stored
            if denominators is not None:
                # Same physical-microvolt bridge as the original clinical runners.
                physical = stored.double().numpy() * denominators
                if self.kind == "cbramod":
                    from scipy.signal import resample
                    physical = resample(physical, 2000, axis=-1) / 100
                else:
                    # LaBraM native engine receives float32 physical uV, then /100.
                    physical = physical.astype("float32") / 100
                x = torch.as_tensor(physical, dtype=torch.float32)
            elif self.kind == "labram":
                x = (stored.double() * 100).float() / 100
            x = x.to(self.device).reshape(len(x), x.shape[1], -1, 200)
            if self.kind == "cbramod":
                self.model.backbone(x)
            else:
                self.model.forward_features(x, input_chans=list(range(x.shape[1] + 1)))
        if list(self.features) != self.endpoints:
            raise ValueError("Some frozen representation hooks did not run")
        return self.features.copy()
