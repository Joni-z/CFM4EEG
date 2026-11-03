from unittest.mock import patch

import pytest
import numpy as np
import torch

from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from scripts.experiments.pac_singlepath import SinglePathPAC
from scripts.experiments.pac_candidate_search import probe
from scripts.experiments.pac_singlepath_search import evaluate
from paclock_bench.probing.synthetic import generate_dataset


@pytest.fixture
def example():
    torch.set_num_threads(2)
    torch.manual_seed(7301)
    f = TriAxialFrontend(n_bands=16, hidden_dim=128, sample_rate=200, patch_len=50,
        filter_range=(.5, 75.), tokenizer_mode='pac_interaction', interaction_mode='rotation',
        waveform_modulation='quadrature')
    t = torch.arange(1000) / 200
    phase = 2 * torch.pi * 4 * t
    x = (torch.cos(phase) + (1 + .4 * torch.cos(phase - .7) + .2 * torch.cos(2 * phase + .3))
         * torch.cos(2 * torch.pi * 32 * t))[None, None] * .2
    return f, SinglePathPAC(f), x


def complex_tokens(tokens):
    return torch.view_as_complex(tokens.reshape(*tokens.shape[:-1], -1, 2).contiguous())


def test_single_measurement_and_no_old_pac(example):
    f, model, x = example
    with patch('paclock_bench.models.paclock.frontend.triaxial.patch_pac_vector', side_effect=AssertionError('old PAC')):
        with patch.object(model.operator, 'forward', wraps=model.operator.forward) as call:
            out = model(x, mode='aligned')
            assert call.call_count == 1
    assert out.shape == (1, 1, 16, 20, 128)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in f.parameters())


def test_content_dependence_identity_and_bounds(example):
    _, model, x = example
    prepared = model.prepare(x)
    content = prepared['content']
    unmapped = prepared['mapping'].sum(-1) == 0
    assert unmapped.any()
    zero = torch.zeros_like(prepared['coefficient'])
    for mode in model.MODES:
        out = complex_tokens(model.transform(prepared, mode))
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[:, :, unmapped], content[:, :, unmapped], rtol=0, atol=0)
        torch.testing.assert_close(complex_tokens(model.transform(prepared, mode, zero)), content, rtol=0, atol=0)
        empty = dict(prepared, content=torch.zeros_like(content))
        assert not model.transform(empty, mode).count_nonzero()
        gain = out.abs() / content.abs().clamp_min(1e-12)
        if mode == 'aligned':
            torch.testing.assert_close(out.abs(), content.abs(), rtol=1e-5, atol=1e-6)
        elif mode == 'strength':
            assert gain.min() >= .99999 and gain.max() <= 1.30001
        else:
            assert gain.min() >= .49999 and gain.max() <= 1.50001
            relative_angle = torch.angle(out * content.conj())
            assert relative_angle.abs().max() <= torch.pi / 6 + 1e-5


def test_reference_invariance_applies_to_aligned_modes(example):
    _, model, x = example
    p = model.prepare(x)
    delta = torch.linspace(-1, 1, model.operator.n_source)
    harmonic = torch.arange(1, 3)
    beta = p['coefficient'] * torch.exp(1j * delta[:, None] * harmonic)[None, None, None, :, None, :]
    changed = dict(p, source_features=[v * torch.exp(1j * delta * (h + 1))[None, None, None, :, None]
                                      for h, v in enumerate(p['source_features'])])
    for mode in ('aligned', 'strength'):
        torch.testing.assert_close(model.transform(p, mode), model.transform(changed, mode, beta), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('mode', SinglePathPAC.MODES)
def test_finite_gradients_and_serialization(example, mode, tmp_path):
    f, model, x = example
    out = model(x, mode=mode)
    out.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.original.amplitude_tokenizer.weight.grad.abs().sum() > 0
    torch.save({'mode': mode, 'state_dict': model.state_dict()}, tmp_path / 'prototype.pt')
    payload = torch.load(tmp_path / 'prototype.pt', weights_only=True)
    restored = SinglePathPAC(f)
    restored.load_state_dict(payload['state_dict'], strict=True)
    torch.testing.assert_close(restored(x, mode=payload['mode']), out, rtol=0, atol=0)


def test_multichannel_and_no_signal(example):
    _, model, x = example
    p = model.prepare(x.expand(2, 3, -1).clone())
    for mode in model.MODES:
        assert model.transform(p, mode).shape == (2, 3, 16, 20, 128)
    zero = model.prepare(torch.zeros_like(x))
    for mode in model.MODES:
        assert torch.isfinite(model.transform(zero, mode)).all()


def test_cached_readouts_equal_existing_protocol():
    data = generate_dataset(40, 16, 16, seed=117)
    features = np.random.default_rng(71).normal(size=(72, 19))
    scores, predictions = evaluate(features, data)
    for degree in (1, 2):
        expected, prediction = probe(features, data, degree)
        actual = scores['pac']['readouts'][degree - 1]
        assert actual['alpha'] == expected['alpha']
        assert actual['validation_mse'] == expected['validation_mse']
        np.testing.assert_array_equal(predictions[f'pac_degree{degree}'], prediction)
        assert actual['test'] == expected['overall']
    changed = dict(data, curves=data['curves'].copy())
    changed['curves'][data['split'] == 'test'] += 10
    other, _ = evaluate(features, changed)
    assert other['pac']['selected']['degree'] == scores['pac']['selected']['degree']
    assert other['pac']['selected']['alpha'] == scores['pac']['selected']['alpha']
