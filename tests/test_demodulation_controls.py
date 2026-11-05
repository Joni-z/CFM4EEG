"""Content controls preserve parameters, initialization, and available coordinates."""
from copy import deepcopy

import pytest
import torch

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.analytic import hilbert
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend


def frontend(**extra):
    torch.manual_seed(812)
    return TriAxialFrontend(n_bands=4, hidden_dim=32, sample_rate=200,
                           patch_len=8, kernel_size=31, filter_range=(.5, 75.),
                           tokenizer_mode="pac_interaction", interaction_mode="rotation",
                           **{"pac_token_mode": "identity", "waveform_modulation": "quadrature", **extra})


def test_defaults_keep_checkpoint_keys_and_initial_parameters():
    reference = frontend()
    for changed in (frontend(content_view="envelope"),
                    frontend(content_view="waveform"), frontend(band_mix_seed=2)):
        assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in changed.parameters())
        for (name, expected), (other, actual) in zip(reference.named_parameters(), changed.named_parameters()):
            assert name == other
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert "band_mix" not in reference.state_dict()
    frontend(content_view="both").load_state_dict(reference.state_dict(), strict=True)


@pytest.mark.parametrize("mode", ("both", "waveform", "envelope"))
def test_actual_coordinate_content_and_gradients(mode):
    fe = frontend(content_view=mode)
    # The second projection starts at zero in the reference. Make it nonzero
    # here so an incorrect or inactive second path cannot pass the test.
    with torch.no_grad():
        fe.waveform_quadrature.weight.copy_(fe.amplitude_tokenizer.weight)
    x = torch.randn(2, 3, 64)
    tokens = fe(x)[0]
    filtered = fe.sinc(x.reshape(6, 1, 64)).reshape(2, 3, 4, 64)
    envelope = torch.log1p(hilbert(filtered).abs())
    first = filtered if mode == "waveform" else envelope
    second = envelope if mode == "envelope" else filtered
    a = fe.amplitude_tokenizer(first.reshape(-1, 1, 64)).transpose(1, 2) * fe.amplitude_scale
    w = fe.waveform_quadrature(second.reshape(-1, 1, 64)).transpose(1, 2)
    torch.testing.assert_close(tokens[..., ::2], a.reshape(2, 3, 4, 8, 16))
    torch.testing.assert_close(tokens[..., 1::2], w.reshape(2, 3, 4, 8, 16))
    tokens.square().mean().backward()
    for conv in (fe.amplitude_tokenizer, fe.waveform_quadrature):
        assert conv.weight.grad is not None
        assert torch.isfinite(conv.weight.grad).all() and conv.weight.grad.abs().sum() > 0


def test_band_mix_is_fixed_invertible_and_preserves_full_grid():
    fe, mixed = frontend(), frontend(band_mix_seed=2)
    x = torch.randn(2, 3, 64)
    tokens, coupling, frequencies = fe(x)
    actual, other_coupling, other_frequencies = mixed(x)
    q = mixed.band_mix
    torch.testing.assert_close(q.T @ q, torch.eye(4), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.einsum("ji,bcjpd->bcipd", q, actual), tokens, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.square().sum(), tokens.square().sum())
    torch.testing.assert_close(coupling, other_coupling, rtol=0, atol=0)
    torch.testing.assert_close(frequencies, other_frequencies, rtol=0, atol=0)
    assert not q.requires_grad and actual.shape == tokens.shape
    restored = frontend(band_mix_seed=2)
    restored.load_state_dict(mixed.state_dict(), strict=True)
    torch.testing.assert_close(restored(x)[0], actual, rtol=0, atol=0)


def test_public_builder_wires_controls_and_rejects_other_mixers():
    cfg = {"model": "paclock", "dataset": "tuev", "sample_rate": 200, "num_classes": 6,
           "model_kwargs": {"arch": "triaxial", "d_model": 32, "depth": 1,
                            "n_bands": 4, "n_heads": 4, "patch_len": 8,
                            "filter_range": [.5, 75.], "freq_mixer": "attention",
                            "tokenizer_mode": "pac_interaction", "interaction_mode": "rotation",
                            "pac_token_mode": "identity", "waveform_modulation": "quadrature",
                            "content_view": "envelope", "band_mix_seed": 2}}
    model = build_model(cfg, (16, 64))
    assert model.frontend.content_view == "envelope"
    assert model.frontend.band_mix is not None
    bad = deepcopy(cfg)
    bad["model_kwargs"]["freq_mixer"] = "coupling"
    with pytest.raises(ValueError, match="content/grid controls"):
        build_model(bad, (16, 64))


@pytest.mark.parametrize("extra", ({"pac_token_mode": "measured"}, {"waveform_content": False}))
def test_content_controls_cannot_silently_change_rotated_experiments(extra):
    with pytest.raises(ValueError):
        frontend(content_view="envelope", **extra)
