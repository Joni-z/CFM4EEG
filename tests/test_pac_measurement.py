"""Physical recovery and invariants, independently constructed from coefficients."""
import math

import torch

from paclock_bench.models.paclock.frontend.measurement import harmonic_regression, PhaseAmplitudeMeasurement


def test_harmonic_recovers_depth_phase_and_linear_nuisance():
    t=torch.linspace(-1,1,1001,dtype=torch.float64)
    phi=2*math.pi*3.7*t+.4
    p=torch.exp(1j*phi)
    y=10+1.5*t+3*torch.cos(phi-.8)
    fit=harmonic_regression(p,y,ridge=0)
    assert fit['valid']
    torch.testing.assert_close(fit['baseline'],torch.tensor(10.,dtype=t.dtype),atol=1e-7,rtol=0)
    torch.testing.assert_close(fit['depth'],torch.tensor(.3,dtype=t.dtype),atol=1e-7,rtol=0)
    torch.testing.assert_close(fit['coef'].angle(),torch.tensor(.8,dtype=t.dtype),atol=1e-7,rtol=0)
    for null in (torch.ones_like(t)*10,10+1.5*t):
        assert harmonic_regression(p,null)['depth'] < 1e-10


def test_phase_origin_and_amplitude_scale_and_backward():
    t=torch.linspace(0,5,1000,dtype=torch.float64)
    p=torch.exp(1j*(2*math.pi*4.3*t+.2))
    y=(2+.8*torch.cos(p.angle()-.6)).requires_grad_()
    base=harmonic_regression(p,y)
    shifted=harmonic_regression(p*torch.exp(torch.tensor(.7j)),y*12)
    torch.testing.assert_close(shifted['depth'],base['depth'],atol=1e-7,rtol=0)
    torch.testing.assert_close(shifted['coef']/12,base['coef']*torch.exp(torch.tensor(.7j)),atol=1e-6,rtol=0)
    base['depth'].backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum()>0


def test_flat_phase_is_not_a_valid_measurement():
    fit=harmonic_regression(torch.ones(400,dtype=torch.complex64),torch.ones(400))
    assert not fit['valid'] and torch.isfinite(fit['coef'])


def test_raw_pipeline_and_batch_independence():
    torch.set_num_threads(2)
    m=PhaseAmplitudeMeasurement([6,40])
    t=torch.arange(1000)/200
    x=(.8*torch.cos(2*math.pi*6*t)+(1+.5*torch.cos(2*math.pi*6*t-.8))*torch.cos(2*math.pi*40*t))[None,None]
    one=m(x)
    assert one.valid.any()
    torch.testing.assert_close(one.depth[one.valid].median(),torch.tensor(.5),atol=.03,rtol=0)
    both=m(torch.cat([x,torch.randn_like(x)],0))
    torch.testing.assert_close(one.depth,both.depth[:1],atol=1e-6,rtol=0)
    assert not m(torch.zeros_like(x)).valid.any()
