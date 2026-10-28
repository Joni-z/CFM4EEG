import math

import numpy as np
import torch

from paclock_bench.models.paclock.frontend.measurement import harmonic_regression
from paclock_bench.models.paclock.frontend.measurement_curve import curve_regression, PACCurveOperator


def test_first_harmonic_matches_existing_estimator():
    torch.manual_seed(7)
    angle=torch.linspace(0,14*math.pi,1001)[None]+torch.rand(3,1)
    phase=torch.exp(1j*angle)
    env=2+.3*phase.real+.4*phase.imag+.1*torch.randn(3,1001)
    mask=(torch.rand_like(env)>.1).float()
    old=harmonic_regression(phase,env,mask)
    new=curve_regression(phase,env,mask,harmonics=1)
    torch.testing.assert_close(new['coef'][...,0],old['coef'],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(new['baseline'],old['baseline'],atol=2e-6,rtol=2e-6)


def test_joint_harmonics_and_amplitude_nuisance_against_numpy():
    t=np.linspace(-1,1,1201); phi=12*np.pi*t+.2
    amp=1+.3*np.sin(2*np.pi*.73*t)
    a=np.log(amp);a=(a-a.mean())/a.std();a2=a*a-(a*a).mean()
    x=np.stack([np.cos(phi),np.sin(phi),np.cos(2*phi),np.sin(2*phi)],-1)
    truth=np.array([.3,-.2,.4,.1]);y=2+.1*t+.2*a+.1*a2+x@truth
    fit=curve_regression(torch.tensor(np.exp(1j*phi)),torch.tensor(y),torch.ones(1201),
                         torch.tensor(amp),ridge=0.)
    np.testing.assert_allclose(torch.view_as_real(fit['coef']).flatten().numpy(),truth,atol=2e-6)
    assert abs(float(fit['baseline'])-2)<1e-6
    assert bool(fit['valid'])


def test_phase_reference_equivariance_and_scaling():
    angle=torch.linspace(0,18*math.pi,1001);p=torch.exp(1j*angle)
    y=2+.4*torch.cos(angle-.7)+.3*torch.cos(2*angle+.2)
    original=curve_regression(p,y,torch.ones_like(y))
    shift=.63
    changed=curve_regression(p*complex(math.cos(shift),math.sin(shift)),y*7,torch.ones_like(y))
    expected=original['coef']*torch.exp(1j*torch.tensor([shift,2*shift]))*7
    torch.testing.assert_close(changed['coef'],expected,atol=3e-6,rtol=3e-6)
    torch.testing.assert_close(changed['baseline'],original['baseline']*7,atol=3e-6,rtol=3e-6)


def test_raw_operator_batch_independence_finite_and_gradients():
    torch.set_num_threads(4)
    t=torch.arange(1000)/200
    x=(.8*torch.cos(2*math.pi*4*t)+(1+.5*torch.cos(2*math.pi*4*t-.4))*torch.cos(2*math.pi*32*t))[None,None]
    op=PACCurveOperator(local=False)
    result=op(x)
    batch=op(torch.cat([x,x*5]))
    torch.testing.assert_close(result.coef,batch.coef[:1])
    assert result.coef.shape==(1,1,20,6,5,2)
    assert bool(result.global_valid.any())
    assert not bool(op(torch.zeros_like(x)).valid.any())
    assert not bool(op(x*float('nan')).valid.any())
    x=x.clone().requires_grad_();out=op(x)
    loss=out.coef.abs().sum()+out.baseline.sum();loss.backward()
    assert bool(torch.isfinite(x.grad).all())


def test_second_harmonic_support_is_not_assumed_from_first():
    op=PACCurveOperator(local=False)
    assert bool((op.harmonic_support[...,0]&~op.harmonic_support[...,1]).any())
    assert not bool((~op.harmonic_support[...,0]&op.harmonic_support[...,1]).any())


def test_local_mixed_validity_has_finite_gradient():
    torch.manual_seed(8);torch.set_num_threads(4)
    t=torch.arange(1000)/200
    x=(torch.cos(2*math.pi*4*t)+(1+.5*torch.cos(2*math.pi*4*t))*torch.cos(2*math.pi*32*t))
    x=(x[None,None]+.02*torch.randn(2,1,1000)).requires_grad_()
    result=PACCurveOperator(local=True)(x)
    (result.coef.abs().sum()+result.baseline.sum()).backward()
    assert bool(torch.isfinite(x.grad).all())


def test_frequency_refinement_is_signal_derived_and_batched():
    torch.set_num_threads(4);t=torch.arange(1000)/200
    phi=2*math.pi*3.27*t
    x=(.8*torch.cos(phi)+(1+.5*torch.cos(2*phi-.4))*torch.cos(2*math.pi*32*t))[None,None]
    op=PACCurveOperator(refine_phase=True)
    a=op(x);b=op(torch.cat([x,x*3]))
    assert abs(float(a.source_frequency[0,0,1])-3.27)<.03
    torch.testing.assert_close(a.coef,b.coef[:1],rtol=2e-5,atol=2e-5)
    assert bool(a.valid[0,0,10,1,2])
