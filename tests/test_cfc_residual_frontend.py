"""Contracts for explicit PAC conditioning without sparse physical row routing."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.cfc import CFCFrontend
from paclock_bench.training.train import cpu_state_dict


def frontend():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    return CFCFrontend(
        n_bands=16, hidden_dim=32, sample_rate=200, patch_len=50,
        filter_range=(.5, 75.), cfc_fusion="residual", cfc_stabilized=False,
    )


def signal(samples=2000):
    t = torch.arange(samples) / 200
    phase = 2 * torch.pi * 4 * t
    envelope = 1 + .2 * torch.cos(phase + .3) + .35 * torch.cos(2 * phase - .6)
    return (torch.cos(phase) + envelope * torch.cos(2 * torch.pi * 32 * t))[None, None]


@pytest.fixture(scope="module")
def measured():
    return frontend().measurement(signal())


def test_relation_reaches_every_content_band_and_retains_strength(measured):
    f = frontend()
    response = f.relation_residual(measured)
    assert response.shape == (1, 1, 40, 16, 32)
    # No low-frequency row silently loses all explicit relation input.
    assert bool((response.abs().amax(dim=(0, 1, 2, 4)) > 1e-7).all())
    tiny = f.relation_residual(replace(measured, raw_coef=measured.raw_coef * 1e-4))
    # Tiny PAC must remain tiny: no unit-length/LayerNorm promotion of weak estimates.
    torch.testing.assert_close(tiny, response * 1e-4, atol=1e-9, rtol=3e-5)


def test_harmonics_are_used_separately_and_reference_invariant(measured):
    f = frontend()
    reference = f.relation_residual(measured)
    for harmonic in range(2):
        assert bool((measured.raw_coef[..., harmonic].abs() > .05).any())
        removed = measured.raw_coef.clone()
        removed[..., harmonic] = 0
        difference = reference - f.relation_residual(replace(measured, raw_coef=removed))
        assert float(difference.detach().abs().max()) > 1e-6
    # Independent source-coordinate shifts require k*delta for harmonic k.
    shift = torch.linspace(-1.1, 1.3, measured.source_phase.shape[2])
    source_rotation = torch.exp(1j * shift)[None, None, :, None]
    coefficient_rotation = torch.exp(1j * shift[:, None] * torch.arange(1, 3))
    coefficient_rotation = coefficient_rotation[None, None, None, :, None, :]
    rephased = replace(
        measured, source_phase=measured.source_phase * source_rotation,
        raw_coef=measured.raw_coef * coefficient_rotation,
    )
    torch.testing.assert_close(f.relation_residual(rephased), reference, atol=2e-6, rtol=3e-5)


def test_zero_or_invalid_measurement_preserves_content_exactly(measured):
    f = frontend().eval()
    zero = replace(measured, raw_coef=torch.zeros_like(measured.raw_coef))
    invalid = replace(measured, valid=torch.zeros_like(measured.valid))
    for packet in (zero, invalid):
        response = f.relation_residual(packet)
        assert torch.count_nonzero(response) == 0
        with patch.object(f.measurement, "forward", return_value=packet):
            f.cfc_guidance = "measured"
            guided = f(signal())[0]
            f.cfc_guidance = "identity"
            content = f(signal())[0]
        torch.testing.assert_close(guided, content, atol=0, rtol=0)


def test_full_model_gradients_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = dict(
        model="paclock", dataset="tuev", num_classes=6, sample_rate=200,
        model_kwargs=dict(
            arch="triaxial", tokenizer_mode="cfc", cfc_fusion="residual",
            cfc_stabilized=False, d_model=32, depth=1, n_bands=16, n_heads=4,
            dropout=0., patch_len=50, kernel_size=201, filter_range=[.5, 75.],
            freq_mixer="none", space_over_bands=True, spatial_pe="xyz",
            band_pe="index", augmentations=[],
        ),
    )
    model = build_model(cfg, (16, 1000))
    assert model.frontend.cfc_fusion == "residual"
    x = signal(1000).expand(1, 16, -1).clone()
    F.cross_entropy(model(x), torch.tensor([2])).backward()
    params = dict(model.named_parameters())
    required = [
        "frontend.phase_tokenizer.weight",
        "frontend.waveform_quadrature.weight",
        "frontend.amplitude_tokenizer.weight",
        "frontend.relation_mix_logits",
        "frontend.residual_gain",
    ]
    for name in required:
        assert name in params, name
        grad = params[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert bool((grad != 0).any()), name
    relation_projections = [p for name, p in params.items() if "relation" in name and name.endswith("weight")]
    assert relation_projections
    assert all(p.grad is not None and bool((p.grad != 0).any()) for p in relation_projections)
    encoder_params = [p for name, p in params.items() if not name.startswith("frontend.")]
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in encoder_params)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert not list(model.frontend.measurement.parameters())

    model.eval()
    with torch.no_grad():
        prediction = model(x)
    state = cpu_state_dict(model)
    assert isinstance(state["frontend.measurement._extra_state"], dict)
    checkpoint = tmp_path / "residual.pt"
    torch.save(state, checkpoint)
    restored = build_model(cfg, (16, 1000)).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), prediction, rtol=1e-6, atol=1e-6)
