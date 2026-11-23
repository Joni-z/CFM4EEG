"""The AAC control changes only the coupling statistic of the two-axis CfFT."""
import pytest
import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.triaxial import (
    NORM_CONST, TriAxialFrontend, patch_aac_vector)


def configs():
    pac = yaml.safe_load(open('configs/model/tuev-scratch.yaml'))
    aac = yaml.safe_load(open('configs/model/tuev-aac-control.yaml'))
    return pac, aac


def test_config_differs_only_in_coupling_measure():
    pac, aac = configs()
    skip = {'name', 'group', 'reproduction', 'model_kwargs'}
    assert {k: v for k, v in pac.items() if k not in skip} == {k: v for k, v in aac.items() if k not in skip}
    kw = dict(aac['model_kwargs'])
    assert kw.pop('coupling_measure') == 'aac' and kw == pac['model_kwargs']


def test_same_parameters_and_initialization_as_pac():
    torch.set_num_threads(2)
    pac, aac = configs()
    torch.manual_seed(2); a = build_model(pac, (16, 1000)).eval()
    torch.manual_seed(2); b = build_model(aac, (16, 1000)).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert sum(p.numel() for p in b.parameters()) == 1_224_486
    x = torch.randn(2, 16, 1000, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ya, yb = a(x), b(x)
    assert torch.isfinite(yb).all() and float((ya - yb).abs().max()) > 1e-6


def test_aac_is_envelope_correlation_times_target_scale():
    A = torch.rand(1, 2, 3, 200, dtype=torch.float64) + 0.5
    R = patch_aac_vector(A, 4)
    assert torch.equal(R.imag, torch.zeros_like(R.imag))
    w = A.reshape(1, 2, 3, 4, 50)
    c = w - w.mean(-1, keepdim=True)
    sd = c.square().mean(-1).sqrt()                       # (B, C, band, P)
    r = torch.einsum('bcipl,bcjpl->bcpij', c, c) / 50
    r = r / (sd.permute(0, 1, 3, 2)[..., :, None] * sd.permute(0, 1, 3, 2)[..., None, :])
    expected = r * sd.permute(0, 1, 3, 2)[..., None, :] / NORM_CONST
    torch.testing.assert_close(R.real, expected)


def test_underflowing_envelope_has_finite_gradient():
    A = (torch.rand(2, 3, 5, 200) * 1e-24).requires_grad_(True)
    R = patch_aac_vector(A, 4)
    R.real.sum().backward()
    assert torch.isfinite(R.real).all() and torch.isfinite(A.grad).all()


@pytest.mark.parametrize('bad', [dict(pac_token_mode='own'), dict(coupling_angle_pool='electrode'),
                                 dict(coupling_measure='plv')])
def test_rejects_settings_without_a_matched_pac_counterpart(bad):
    kw = dict(n_bands=16, hidden_dim=32, sample_rate=200, patch_len=50, filter_range=(.5, 75.),
              tokenizer_mode='pac_interaction', interaction_mode='rotation',
              waveform_modulation='quadrature', pac_token_mode='measured', coupling_measure='aac')
    with pytest.raises(ValueError, match='AAC control requires|coupling_measure must be'):
        TriAxialFrontend(**{**kw, **bad})
