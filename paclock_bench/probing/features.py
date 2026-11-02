"""Frozen final-token adapters; original model implementations remain unchanged."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import torch
import yaml

from paclock_bench.models.build import build_model

ROOT = Path(__file__).resolve().parents[2]
NORMALIZATIONS = ("none", "q95", "div100")
CONFIG_FIELDS = ("model", "dataset", "num_classes", "sample_rate", "model_kwargs")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def load_project_tokenizer(config_path, checkpoint_path, input_shape: tuple[int, int],
                           *, normalization: str | None = None):
    """Strictly restore the recorded model; caller applies the declared scaling."""
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    cfg = yaml.safe_load(config_path.read_text())
    normalization = normalization or cfg.get("probe_input_normalization")
    if normalization not in NORMALIZATIONS:
        raise ValueError("Declare project normalization explicitly: none, q95, or div100")
    if len(input_shape) != 2 or min(input_shape) <= 0:
        raise ValueError("input_shape must be positive (channels, samples)")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise ValueError("Project checkpoint must contain its original training config")
    training = saved["config"]
    differing = [k for k in CONFIG_FIELDS if cfg.get(k) != training.get(k)]
    if differing:
        raise ValueError(f"Checkpoint training config differs: {differing}")
    if cfg.get("model") not in ("paclock", "wave_patch_twoaxis"):
        raise ValueError("Only original CfFT and waveform-patch models are supported")
    state_key = "model" if "model" in saved else "state_dict"
    if not isinstance(saved.get(state_key), dict):
        raise ValueError("Project checkpoint has no model/state_dict weights")
    constructor_cfg = {k: v for k, v in cfg.items() if k != "checkpoint"}
    model = build_model(constructor_cfg, input_shape)
    model.load_state_dict(saved[state_key], strict=True)
    metadata = dict(config_path=str(config_path.resolve()), config_sha256=_sha(config_path),
                    checkpoint_path=str(checkpoint_path.resolve()),
                    checkpoint_sha256=_sha(checkpoint_path), state_key=state_key,
                    input_shape=list(input_shape), input_normalization=normalization,
                    normalization_applied_by="caller", training_config_match=True,
                    checked_config_fields=list(CONFIG_FIELDS), training_config=training,
                    model_source_sha256={str(path.relative_to(ROOT)): _sha(path)
                                         for path in sorted((ROOT / "paclock_bench/models").rglob("*.py"))},
                    checkpoint_loading="strict", endpoint="frontend_before_positional_encoding")
    return _freeze(model), metadata


def normalize_project_signals(signals: torch.Tensor, normalization: str) -> torch.Tensor:
    """Explicit input-only normalization; project_tokens never normalizes secretly."""
    if normalization == "none":
        return signals
    if normalization == "div100":
        return signals / 100
    if normalization == "q95":
        return signals / (torch.quantile(signals.abs(), .95, dim=-1, keepdim=True) + 1e-8)
    raise ValueError(f"Unknown project normalization: {normalization}")


def _input(signals: torch.Tensor) -> None:
    if signals.ndim != 3 or min(signals.shape) < 1 or not signals.is_floating_point():
        raise ValueError("signals must be floating point [samples, channels, time]")
    if not bool(torch.isfinite(signals).all()):
        raise ValueError("signals must be finite")


@torch.no_grad()
def project_tokens(model, signals: torch.Tensor, kind: str = "cfft") -> torch.Tensor:
    """Preserve every channel/band coordinate; return [samples, patches, features]."""
    _input(signals)
    if model.training:
        raise ValueError("Frozen token extraction requires model.eval()")
    if kind == "cfft":
        tokens = model.frontend(signals)[0]
        if tokens.ndim != 5:
            raise ValueError("CfFT frontend must return [N,C,bands,P,D]")
        return tokens.permute(0, 3, 1, 2, 4).flatten(2)
    if kind != "wavepatch":
        raise ValueError(f"Unknown tokenizer kind: {kind}")
    n, channels, length = signals.shape
    if length % model.patch_len:
        raise ValueError("Waveform input length must be divisible by patch_len")
    tokens = model.frontend(signals.reshape(n, channels, -1, model.patch_len))
    return tokens.permute(0, 2, 1, 3).flatten(2)


def load_tfm_tokenizer(vendor_root, checkpoint_path):
    """Import the pinned author's factory and strictly load tokenizer-only weights."""
    vendor_root, checkpoint_path = Path(vendor_root).resolve(), Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    pin = json.loads((ROOT / "upstream/sources.json").read_text())["tfm"]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(vendor_root), *args])

    commit = git("rev-parse", "HEAD").decode().strip()
    if commit != pin["commit"]:
        raise ValueError(f"TFM source revision differs: {commit} != {pin['commit']}")
    sources = {}
    for relative in ("models/tfm_token.py", "utils/utils.py"):
        path = vendor_root / relative
        if path.read_bytes() != git("show", f"HEAD:{relative}"):
            raise ValueError(f"TFM author source has local modifications: {relative}")
        sources[relative] = _sha(path)
    license_text = git("show", "HEAD:LICENSE").decode()
    source = vendor_root / "models/tfm_token.py"
    spec = importlib.util.spec_from_file_location(f"pac_probe_tfm_{sources['models/tfm_token.py']}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    metadata = dict(vendor_root=str(vendor_root), source_url=pin["url"], source_commit=commit,
                    source_sha256=sources, license=license_text,
                    license_sha256=hashlib.sha256(license_text.encode()).hexdigest(),
                    checkpoint_path=str(checkpoint_path.resolve()),
                    checkpoint_sha256=_sha(checkpoint_path), checkpoint_loading="strict",
                    endpoint="tokenize.quant_out", input_normalization="per_channel_q95",
                    sample_rate=200, stft_window=200, stft_hop=100, stft_center=False)
    return _freeze(model), metadata


# get_stft_torch below is the author's 200 Hz operation with reshape replacing
# einops layout syntax. Source: TFM-Tokenizer/utils/utils.py:165-176, pinned by
# upstream/sources.json. MIT License, Copyright (c) 2025 Jathurshan Pradeepkumar.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def get_stft_torch(signals: torch.Tensor) -> torch.Tensor:
    n, channels, length = signals.shape
    flat = signals.reshape(n * channels, length)
    window = torch.hann_window(200).to(flat.device)
    spectrum = torch.stft(flat, n_fft=200, hop_length=100, onesided=True,
                          return_complex=True, center=False, window=window).abs()[:, :100]
    return spectrum.reshape(n, channels, 100, -1)


@torch.no_grad()
def tokenize_tfm(model, signals: torch.Tensor) -> torch.Tensor:
    """200 Hz input -> author's frozen quantized embeddings, never integer IDs."""
    _input(signals)
    if model.training:
        raise ValueError("TFM must be in eval mode to prevent EMA codebook updates")
    n, channels, length = signals.shape
    if length < 200:
        raise ValueError("TFM requires at least 200 samples at 200 Hz")
    x = normalize_project_signals(signals, "q95")
    spectrogram = get_stft_torch(x).reshape(n * channels, 100, -1)
    quant_out, _, _ = model.tokenize(spectrogram, x.reshape(n * channels, length))
    return quant_out.reshape(n, channels, -1, quant_out.shape[-1]).permute(0, 2, 1, 3).flatten(2)


def linear_token_budget(tokens: np.ndarray, time_bins: int, width: int,
                        seed: int = 0) -> np.ndarray:
    """Fixed linear time averaging and feature projection; no fitted data statistics."""
    if np.iscomplexobj(tokens):
        raise ValueError("Token features must be real-valued embeddings")
    x = np.asarray(tokens, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("tokens must be finite [samples, patches, features]")
    if isinstance(time_bins, bool) or isinstance(width, bool) or int(time_bins) != time_bins or int(width) != width:
        raise ValueError("time_bins and width must be positive integers")
    time_bins, width = int(time_bins), int(width)
    if min(time_bins, width) < 1:
        raise ValueError("time_bins and width must be positive integers")
    patches, features = x.shape[1:]
    edges = np.linspace(0, patches, time_bins + 1)
    overlap = np.maximum(0, np.minimum(edges[1:, None], np.arange(patches) + 1)
                         - np.maximum(edges[:-1, None], np.arange(patches)))
    pooled = np.einsum("bp,npd->nbd", overlap / np.diff(edges)[:, None], x)
    if width < features:
        projection = GaussianRandomProjection(n_components=width, random_state=seed)
        projection.fit(np.zeros((1, features)))
        pooled = projection.transform(pooled.reshape(-1, features)).reshape(len(x), time_bins, width)
    elif width > features:
        pooled = np.pad(pooled, ((0, 0), (0, 0), (0, width - features)))
    return pooled.reshape(len(x), time_bins * width)
