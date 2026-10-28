import math
import numpy as np
import pytest
import torch
from scipy.stats import binom

from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator,curve_regression
from paclock_bench.models.paclock.frontend.measurement_uncertain import UncertainPACOperator,coefficient_uncertainty
from paclock_bench.models.paclock.frontend.pac_calibration import PACCalibration,calibration_rank


def test_cluster_covariance_against_independent_numpy_sandwich():
    rng=np.random.default_rng(3);a=np.arange(2000)*2*np.pi/100
    phase=torch.tensor(np.exp(1j*a),dtype=torch.complex64)
    y=torch.tensor(2+.4*np.cos(a)+.2*np.sin(2*a)+rng.normal(0,.1,len(a)),dtype=torch.float32)
    mask=torch.ones_like(y)
    fit=curve_regression(phase,y,mask,ridge=0)
    actual,good=coefficient_uncertainty(phase,y,mask,fit,2,ridge=0)
    assert bool(good)
    # Independent full-design OLS includes intercept/time explicitly.
    t=np.linspace(-1,1,len(a),dtype=np.float32)
    d=np.column_stack([np.ones(len(a)),t,phase.real.numpy(),phase.imag.numpy(),
                      (phase**2).real.numpy(),(phase**2).imag.numpy()])
    beta=np.linalg.lstsq(d,y.numpy(),rcond=None)[0];residual=y.numpy()-d@beta
    path=torch.nn.functional.pad(torch.angle(phase[1:]*phase[:-1].conj()).cumsum(-1),(1,0)).numpy()
    groups=np.floor(path/(4*np.pi)).astype(int)
    scores=[]
    for group in np.unique(groups):
        inside=groups==group
        cycles=np.sum(np.diff(path)*inside[1:])/(2*np.pi)
        if cycles>=.98:scores.append(np.sum(d[inside]*residual[inside,None],axis=0))
    score=np.array(scores);inverse=np.linalg.inv(d.T@d)
    cov=inverse@(score.T@score)@inverse*len(scores)/(len(scores)-1)
    expected=np.diag(cov)[2:].reshape(2,2).sum(-1)
    np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-3,atol=1e-8)


def test_uncertainty_scale_batch_gradient_and_checkpoint_contract():
    torch.set_num_threads(2);t=torch.arange(2000)/200;ph=2*torch.pi*4*t
    x=(.8*torch.cos(ph)+(1+.4*torch.cos(ph-.3))*torch.cos(2*torch.pi*32*t))[None,None]
    x=x+.05*torch.randn(x.shape,generator=torch.Generator().manual_seed(12))
    op=UncertainPACOperator();a=op(x);b=op(torch.cat([x,x*3]))
    torch.testing.assert_close(a.coef,b.coef[:1],atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(a.reliability,b.reliability[1:],atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(a.coefficient_variance*9,b.coefficient_variance[1:],atol=2e-5,rtol=2e-4)
    assert bool(a.uncertainty_valid.any())
    assert bool(((a.reliability>=0)&(a.reliability<=1)).all())
    torch.testing.assert_close(a.coef,a.raw_coef)
    f=a.relation_features()
    assert f['features'].shape==(1,1,40,6,5,2,2)
    assert not bool(a.phase_valid[...,0].any())
    assert not bool(f['features'][~f['harmonic_valid']].any())
    xx=x.requires_grad_();z=op(xx)
    (z.relation_features()['features'].square().sum()+z.coefficient_variance.sum()).backward()
    assert bool(torch.isfinite(xx.grad).all())
    UncertainPACOperator().load_state_dict(op.state_dict())
    with pytest.raises(ValueError,match='configuration'):
        PACCurveOperator().load_state_dict(op.state_dict())


@pytest.mark.parametrize('n',[1,20,512,2000])
def test_tolerance_rank_matches_independent_binomial_quantile(n):
    assert calibration_rank(n,.05,.95)==int(binom.ppf(.95,n,.95))+1
    assert calibration_rank(n,.05)==min(n+1,math.ceil((n+1)*.95))


def test_calibration_ties_abstention_serialization_and_input_binding(tmp_path):
    torch.set_num_threads(2);op=PACCurveOperator()
    cal=PACCalibration.fit(np.zeros(512),op,1,1000,confidence=.95)
    assert cal.rank==495
    x=torch.zeros(2,1,1000);out=op(x)
    assert not bool(cal.detect(out,op,1,1000).any())
    p=tmp_path/'cal.json';cal.save(p);assert PACCalibration.load(p)==cal
    with pytest.raises(ValueError,match='match'):
        cal.detect(out,op,1,2000)
    assert PACCalibration.fit([0.],op,1,1000).threshold is None
    with pytest.raises(ValueError,match='finite'):
        PACCalibration.fit([float('nan')],op,1,1000)
