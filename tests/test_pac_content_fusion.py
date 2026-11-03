import pytest
import torch
from dataclasses import replace
from scripts.experiments.pac_content_fusion import PACContentFusion
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend


def fixture():
    torch.set_num_threads(2);torch.manual_seed(7301)
    f=TriAxialFrontend(n_bands=16,hidden_dim=128,sample_rate=200,patch_len=50,
        filter_range=(.5,75.),tokenizer_mode='pac_interaction',interaction_mode='rotation',waveform_modulation='quadrature')
    t=torch.arange(1000)/200;ph=2*torch.pi*4*t
    x=(torch.cos(ph)+(1+.4*torch.cos(ph-.7)+.2*torch.cos(2*ph+.3))*torch.cos(2*torch.pi*32*t))[None,None]*.2
    return f,PACContentFusion(f),x


def test_exact_old_grid_and_zero_gain():
    f,m,x=fixture();a=m.prepare(x)
    torch.testing.assert_close(a['baseline'],f(x)[0],rtol=0,atol=0)
    for mode in ('trace_gain','carrier_binding','aligned_binding','radial_binding'):
        torch.testing.assert_close(m.fuse(a,mode=mode,gain=0.),a['baseline'],rtol=0,atol=0)
        out=m.fuse(a,mode=mode,gain=.1)
        assert out.shape==(1,1,16,20,128)and torch.isfinite(out).all()


def test_trace_is_phase_coordinate_invariant():
    f,m,x=fixture();a=m.prepare(x);beta=a['coefficient'];old=m.modulation_trace(a,beta)
    delta=torch.linspace(-1,1,m.operator.n_source);harmonic=torch.arange(1,3)
    shifted=replace(a['measured'],source_phase=a['measured'].source_phase*torch.exp(1j*delta)[None,None,:,None])
    coefficient=beta*torch.exp(1j*delta[:,None]*harmonic)[None,None,None,:,None,:]
    changed=dict(a,measured=shifted)
    torch.testing.assert_close(m.modulation_trace(changed,coefficient),old,rtol=3e-5,atol=3e-6)


def test_no_content_independent_correction_and_band_bound():
    f,m,x=fixture();a=m.prepare(x)
    empty=dict(a,amplitude=torch.zeros_like(a['amplitude']),filtered=torch.zeros_like(a['filtered']))
    torch.testing.assert_close(m.carrier(empty),torch.zeros_like(m.carrier(empty)),rtol=0,atol=0)
    for mode in ('carrier_binding','radial_binding'):
        torch.testing.assert_close(m.fuse(empty,mode=mode,gain=.1),a['baseline'],rtol=0,atol=0)
    for mode in ('carrier_binding','aligned_binding','radial_binding'):
        out=m.fuse(a,mode=mode,gain=.1)
        norm=a['baseline'].square().mean((-1,-2)).sqrt().clamp_min(1e-12)
        assert float(((out-a['baseline']).square().mean((-1,-2)).sqrt()/norm).max().detach())<=.100001


def test_content_gradients_remain_trainable():
    f,m,x=fixture();out=m(x,mode='trace_gain',gain=.1);out.square().mean().backward()
    g=m.original.amplitude_tokenizer.weight.grad
    assert g is not None and torch.isfinite(g).all()and g.abs().sum()>0


def test_calibrated_radial_fusion_is_bounded_and_serializable(tmp_path):
    f,m,x=fixture();x=x.expand(2,3,-1).clone();a=m.prepare(x)
    m.residual_scale.fill_(20.)
    out=m.fuse(a,mode='radial_binding',gain=.3)
    assert out.shape==(2,3,16,20,128)
    size=(out-a['baseline']).square().mean((-1,-2)).sqrt()
    norm=a['baseline'].square().mean((-1,-2)).sqrt()
    assert bool((size<=.300001*norm).all())
    torch.save(m.state_dict(),tmp_path/'prototype.pt')
    restored=PACContentFusion(f)
    restored.load_state_dict(torch.load(tmp_path/'prototype.pt',weights_only=True),strict=True)
    torch.testing.assert_close(restored(x,mode='radial_binding',gain=.3),out,rtol=0,atol=0)
    m.original.zero_grad();out.square().mean().backward()
    assert m.original.amplitude_tokenizer.weight.grad.abs().sum()>0
    with pytest.raises(ValueError):m.fuse(a,mode='radial_binding',gain=float('nan'))
    with pytest.raises(ValueError):m.fuse(a,mode='radial_binding',gain=-.1)
