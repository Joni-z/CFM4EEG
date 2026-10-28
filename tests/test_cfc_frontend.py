from dataclasses import replace
import copy

import numpy as np
import torch

from paclock_bench.models.paclock.frontend.cfc import CFCFrontend
from paclock_bench.models.build import build_model
from paclock_bench.training.train import cpu_state_dict


def frontend():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    return CFCFrontend(n_bands=16, hidden_dim=32, sample_rate=200,
                       patch_len=50, filter_range=(.5,75.))


def signal():
    t = torch.arange(2000)/200
    phase = 2*torch.pi*4*t
    return (torch.cos(phase)+(1+.35*torch.cos(2*phase-.6))*
            torch.cos(2*torch.pi*32*t))[None, None]


def test_physical_mapping_matches_direct_filter_response():
    f = frontend()
    weights = f.target_mapping()
    assert weights.shape == (16,5)
    assert bool((weights >= 0).all()) and bool((weights.sum(-1) <= 1.000001).all())
    kernel = f.sinc.filters().detach().numpy()[:,0]
    frequency = f.measurement.physical_grid()['target_centers'].numpy()
    basis = np.exp(-2j*np.pi*np.arange(kernel.shape[-1])[:,None]*frequency[None,:]/200)
    power = abs(kernel@basis)**2
    peak = abs(np.fft.rfft(kernel,n=2048))**2
    expected = np.clip(power/peak.max(-1,keepdims=True),0,1)
    expected /= np.maximum(expected.sum(-1,keepdims=True),1)
    np.testing.assert_allclose(weights.detach(),expected,atol=2e-6,rtol=2e-4)
    assert float(weights[0].detach().max()) < .01  # No normalized long-range assignment.


def test_both_harmonics_and_reference_equivariance_reach_tokens():
    f=frontend(); measurement=f.measurement(signal())
    context,bound=f.relation_context(measurement)
    assert bool((measurement.raw_coef[...,1].abs() > .1).any())
    no_second=measurement.raw_coef.clone();no_second[...,1]=0
    changed,_=f.relation_context(replace(measurement,raw_coef=no_second))
    assert not torch.allclose(context,changed,atol=1e-6)
    shift=.7
    factors=torch.exp(1j*torch.tensor([shift,2*shift]))
    rephased=replace(measurement, source_phase=measurement.source_phase*complex(np.exp(1j*shift)),
                     raw_coef=measurement.raw_coef*factors)
    rotated,rotated_bound=f.relation_context(rephased)
    torch.testing.assert_close(context,rotated,rtol=3e-5,atol=2e-6)
    torch.testing.assert_close(bound,rotated_bound,rtol=3e-5,atol=2e-6)
    f.eval();guided=f(signal())[0];f.cfc_guidance='identity';plain=f(signal())[0]
    assert not torch.allclose(guided,plain,atol=1e-6)


def test_invalid_relations_use_identity_and_gradients_remain_finite():
    f=frontend();x=torch.zeros(1,1,1000)
    guided=f(x)[0];f.cfc_guidance='identity';identity=f(x)[0]
    torch.testing.assert_close(guided,identity,atol=0,rtol=0)
    f.cfc_guidance='measured'
    tokens=f(signal())[0];assert tokens.shape==(1,1,16,40,32)
    tokens.square().mean().backward()
    for p in f.parameters():
        if p.grad is not None: assert bool(torch.isfinite(p.grad).all())
    assert bool((f.phase_tokenizer.weight.grad!=0).any())
    assert bool((f.waveform_quadrature.weight.grad!=0).any())
    assert not list(f.measurement.parameters())


def test_full_model_checkpoint_preserves_operator_extra_state(tmp_path):
    cfg=dict(model='paclock',dataset='tuev',num_classes=6,sample_rate=200,
        model_kwargs=dict(arch='triaxial',tokenizer_mode='cfc',d_model=32,depth=1,
            n_bands=16,n_heads=4,dropout=0.,patch_len=50,kernel_size=201,
            filter_range=[.5,75.],freq_mixer='none',space_over_bands=True,
            spatial_pe='xyz',band_pe='index',augmentations=[]))
    torch.set_num_threads(2);torch.manual_seed(0)
    m=build_model(cfg,(16,1000)).eval()
    state=cpu_state_dict(m)
    assert isinstance(state['frontend.measurement._extra_state'],dict)
    original=copy.deepcopy(state['frontend.measurement._extra_state'])
    m.frontend.measurement.configuration['test_only']=True
    assert state['frontend.measurement._extra_state']==original
    del m.frontend.measurement.configuration['test_only']
    x=signal()[...,:1000].expand(1,16,-1).clone()
    reference=m(x);torch.save(state,tmp_path/'state.pt')
    restored=build_model(cfg,(16,1000)).eval()
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True),strict=True)
    torch.testing.assert_close(restored(x),reference,rtol=1e-6,atol=1e-6)
