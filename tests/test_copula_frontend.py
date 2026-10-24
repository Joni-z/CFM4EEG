"""Numerical and isolation checks for the copula-guided tokenizer pilot."""
import numpy as np
import pytest
import torch
from scipy.special import ndtri
from scipy.stats import rankdata

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.copula import (
    CopulaFrontend, copula_relations, gaussian_copula_scores,
)


def test_midrank_scores_match_scipy_and_handle_constant_channels():
    x = torch.tensor([[0., 0., 1., 2., 2., 5.], [7., 7., 7., 7., 7., 7.]])
    expected = []
    for row in x.numpy():
        values = ndtri((rankdata(row, method='average') - .5) / len(row))
        values -= values.mean()
        expected.append(values / max(np.sqrt(np.mean(values ** 2)), 1e-4))
    actual = gaussian_copula_scores(x)
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7)
    assert torch.equal(actual[1], torch.zeros(6))
    assert not actual.requires_grad


def test_relation_is_invariant_to_monotone_envelope_transforms_and_batch_companions():
    torch.manual_seed(3)
    envelope = torch.rand(2, 1, 2, 3, 200)
    phase = torch.polar(torch.ones_like(envelope), torch.randn_like(envelope))
    centers = torch.tensor([2., 8., 25.])
    relations = copula_relations(phase, envelope, centers)
    transformed = copula_relations(phase, envelope.exp(), centers)
    alone = copula_relations(phase[:1], envelope[:1], centers)
    for original, changed, single in zip(relations, transformed, alone):
        torch.testing.assert_close(original, changed, atol=2e-6, rtol=0)
        torch.testing.assert_close(original[:1], single)


def test_known_aac_and_phase_amplitude_relations_have_correct_orientation():
    torch.manual_seed(7)
    length = 4000
    angle = torch.rand(length) * (2 * torch.pi)
    common = torch.randn(length)
    phase = torch.polar(torch.ones(3, length), torch.stack((angle, angle * 3, angle * 7)))
    envelope = torch.stack((common.exp(), (2 * common).exp(), 2 + angle.cos()))
    aac, pac = copula_relations(phase, envelope, torch.tensor([3., 9., 30.]))
    assert aac[0, 1] > .94
    assert abs(float(aac[0, 2])) < .05
    assert pac[2, 0] > .85  # phase source band 0 -> amplitude target band 2
    assert pac[0, 2] == 0  # reverse direction excluded by actual band centers
    assert torch.isfinite(aac).all() and torch.isfinite(pac).all()


def test_zero_signal_is_finite_and_content_and_guide_parameters_receive_gradients():
    torch.manual_seed(4)
    frontend = CopulaFrontend(4, 16, 200, kernel_size=31, patch_len=25,
                              relation_window_samples=100, filter_range=(2., 70.))
    zeros, relation, _ = frontend(torch.zeros(2, 2, 200))
    assert zeros.shape == (2, 2, 4, 8, 16)
    assert torch.isfinite(zeros).all()
    assert torch.count_nonzero(relation) == 0
    output = frontend(torch.randn(2, 2, 200))[0]
    output.square().mean().backward()
    for name in ('waveform_projection.weight', 'envelope_projection.weight',
                 'qkv.weight', 'relation_weight', 'sinc.low_hz_'):
        gradient = dict(frontend.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_frontend_eval_does_not_depend_on_other_batch_examples():
    torch.manual_seed(8)
    frontend = CopulaFrontend(4, 16, 200, kernel_size=31, patch_len=25,
                              relation_window_samples=100, filter_range=(2., 70.)).eval()
    x = torch.randn(2, 2, 200)
    torch.testing.assert_close(frontend(x)[0][:1], frontend(x[:1])[0], atol=2e-6, rtol=2e-5)


def test_model_build_with_original_twoaxis_encoder_and_bad_window_rejection():
    cfg = dict(model='paclock', dataset='tuev', num_classes=6, sample_rate=200,
               model_kwargs=dict(arch='triaxial', tokenizer_mode='copula', d_model=16,
                                 n_bands=4, depth=1, n_heads=4, freq_mixer='none',
                                 space_over_bands=True, patch_len=25, kernel_size=31,
                                 relation_window_samples=100, filter_range=(2., 70.)))
    model = build_model(cfg, (2, 200))
    logits = model(torch.randn(2, 2, 200))
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    with pytest.raises(ValueError, match='divisible'):
        model.frontend(torch.randn(1, 2, 175))
