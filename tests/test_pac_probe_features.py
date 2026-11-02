"""Small random fixtures test adapters only; none represent trained benchmarks."""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import stft
import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.probing.features import (
    get_stft_torch, linear_token_budget, load_project_tokenizer, load_tfm_tokenizer,
    normalize_project_signals, project_tokens, tokenize_tfm,
)


@pytest.fixture
def project_fixture(tmp_path):
    """Generated random weights exercise restoration, not scientific accuracy."""
    cfg = dict(model="wave_patch_twoaxis", dataset="synthetic", sample_rate=200,
               num_classes=2, model_kwargs=dict(d_model=8, depth=1, n_heads=2,
               dropout=.1, patch_len=50, spatial_pe="index", augmentations=[]))
    model = build_model(cfg, (2, 400))
    config = tmp_path / "fixture.yaml"
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / "random-fixture.pt"
    torch.save(dict(model=model.state_dict(), config=cfg, purpose="test fixture only"), checkpoint)
    return config, checkpoint, cfg


def test_strict_project_restore_and_explicit_normalization(project_fixture):
    config, checkpoint, cfg = project_fixture
    with pytest.raises(ValueError, match="normalization explicitly"):
        load_project_tokenizer(config, checkpoint, (2, 400))
    model, meta = load_project_tokenizer(config, checkpoint, (2, 400), normalization="div100")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert meta["training_config_match"] and meta["input_normalization"] == "div100"
    x = torch.randn(3, 2, 400)
    torch.testing.assert_close(normalize_project_signals(x, "div100"), x / 100)
    # A new training recipe cannot silently reuse old weights, even when shapes fit.
    changed = copy.deepcopy(cfg)
    changed["model_kwargs"]["dropout"] = .2
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="training config differs"):
        load_project_tokenizer(config, checkpoint, (2, 400), normalization="none")


def test_missing_or_incomplete_checkpoint_never_falls_back(project_fixture):
    config, checkpoint, _ = project_fixture
    with pytest.raises(FileNotFoundError):
        load_project_tokenizer(config, checkpoint.with_name("absent.pt"), (2, 400), normalization="none")
    saved = torch.load(checkpoint, weights_only=False)
    saved["model"].pop("frontend.weight")
    torch.save(saved, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_project_tokenizer(config, checkpoint, (2, 400), normalization="none")


def test_wave_tokens_preserve_channels_without_running_encoder(project_fixture):
    config, checkpoint, _ = project_fixture
    model, _ = load_project_tokenizer(config, checkpoint, (2, 400), normalization="none")
    def forbidden(*args):
        raise AssertionError("Downstream encoder must not execute")
    hook = model.encoder.register_forward_pre_hook(forbidden)
    x = torch.randn(3, 2, 400)
    tokens = project_tokens(model, x, "wavepatch")
    swapped = project_tokens(model, x.flip(1), "wavepatch")
    hook.remove()
    assert tokens.shape == (3, 8, 16) and not tokens.requires_grad
    torch.testing.assert_close(swapped[..., :8], tokens[..., 8:])
    torch.testing.assert_close(swapped[..., 8:], tokens[..., :8])
    model.train()
    with pytest.raises(ValueError, match="eval"):
        project_tokens(model, x, "wavepatch")


def test_cfft_endpoint_keeps_band_coordinates(project_fixture):
    _, _, cfg = project_fixture
    cfg = copy.deepcopy(cfg)
    cfg["model"] = "paclock"
    cfg["model_kwargs"].update(arch="triaxial", n_bands=2, kernel_size=31,
                               tokenizer_mode="raw", freq_mixer="none")
    model = build_model(cfg, (2, 400)).eval()
    x = torch.randn(2, 2, 400)
    tokens = project_tokens(model, x)
    swapped = project_tokens(model, x.flip(1))
    assert tokens.shape == (2, 8, 32)
    torch.testing.assert_close(swapped[..., :16], tokens[..., 16:])


def test_author_stft_matches_independent_scipy_operation():
    x = torch.randn(2, 3, 1000)
    _, _, reference = stft(x.numpy(), fs=200, nperseg=200, noverlap=100,
                           boundary=None, padded=False, scaling="spectrum", axis=-1)
    np.testing.assert_allclose(get_stft_torch(x).numpy(), abs(reference[..., :100, :]) * 100,
                               rtol=2e-5, atol=1e-5)


def test_budget_is_linear_and_independent_of_sample_order():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(4, 7, 13)), rng.normal(size=(4, 7, 13))
    reduce = lambda x: linear_token_budget(x, time_bins=3, width=5, seed=11)
    np.testing.assert_allclose(reduce(a + b), reduce(a) + reduce(b), atol=1e-12)
    np.testing.assert_allclose(reduce(a[::-1]), reduce(a)[::-1])
    np.testing.assert_allclose(reduce(a[:1]), reduce(a)[:1])
    assert reduce(a).shape == (4, 15)
    padded = linear_token_budget(np.ones((2, 7, 2)), 9, 4).reshape(2, 9, 4)
    np.testing.assert_allclose(padded[..., :2], 1)
    np.testing.assert_array_equal(padded[..., 2:], 0)
    with pytest.raises(ValueError, match="finite"):
        linear_token_budget(a * np.nan, 3, 5)


@pytest.mark.skipif(not os.environ.get("TFM_VENDOR_ROOT"), reason="optional pinned author checkout")
def test_actual_tfm_random_fixture_strict_restore_and_quantized_endpoint(tmp_path):
    """Optional integration test; this temporary random checkpoint is fixture-only."""
    root = Path(os.environ["TFM_VENDOR_ROOT"])
    spec = importlib.util.spec_from_file_location("tfm_fixture_author", root / "models/tfm_token.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_tfm_tokenizer_2x2x8().eval()
    checkpoint = tmp_path / "author-random-fixture.pth"
    torch.save(model.state_dict(), checkpoint)
    restored, metadata = load_tfm_tokenizer(root, checkpoint)
    assert metadata["endpoint"] == "tokenize.quant_out"
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    x = torch.randn(2, 2, 1000)
    tokens = tokenize_tfm(restored, x)
    assert tokens.shape == (2, 9, 128)
    torch.testing.assert_close(tokens, tokenize_tfm(restored, x * 7))
    # Output vectors belong to the frozen codebook, never numeric token IDs.
    vectors = tokens.reshape(2, 9, 2, 64).reshape(-1, 64)
    distances = torch.cdist(vectors, restored.quantizer.embedding.weight)
    assert bool((distances.min(1).values < 1e-6).all())
