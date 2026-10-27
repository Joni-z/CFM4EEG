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


def test_actual_frontend_gradients_identity_fallback_and_roundtrip():
    import io
    from paclock_bench.models.paclock.frontend.measured import MeasuredPACFrontend
    torch.manual_seed(0)
    frontend=MeasuredPACFrontend(n_bands=4,hidden_dim=32,sample_rate=200,
                                patch_len=50,filter_range=(.5,75))
    t=torch.arange(1000)/200
    x=(torch.cos(2*math.pi*4*t)+(1+.5*torch.cos(2*math.pi*4*t-.8))*torch.cos(2*math.pi*40*t))[None,None]
    tokens,_,_=frontend(x)
    assert tokens.shape==(1,1,4,20,32) and torch.isfinite(tokens).all()
    tokens.square().mean().backward()
    assert frontend.amplitude_tokenizer.weight.grad.abs().sum()>0
    # Energy is preserved by a rotation, so use a direction-sensitive loss to
    # test the phase-projection gradient rather than squared token norm.
    frontend.zero_grad()
    tokens=frontend(x)[0]
    (tokens[...,::2].mean()).backward()
    assert frontend.phase_tokenizer.weight.grad is not None
    assert torch.isfinite(frontend.phase_tokenizer.weight.grad).all()
    assert frontend.phase_tokenizer.weight.grad.abs().sum()>0
    frontend.measurement_mode='measured'
    modulated=frontend(x)[0]
    frontend.measurement_mode='identity'
    unmodulated=frontend(x)[0]
    torch.testing.assert_close(modulated.reshape(*modulated.shape[:-1],-1,2).square().sum(-1),
                               unmodulated.reshape(*unmodulated.shape[:-1],-1,2).square().sum(-1),atol=1e-5,rtol=1e-5)
    assert (modulated-unmodulated).abs().max()>1e-7
    # Invalid/no-coupling data retain the exact unrotated carrier.
    frontend.measurement_mode='measured'
    measured=frontend(torch.zeros_like(x))[0]
    frontend.measurement_mode='identity'
    identity=frontend(torch.zeros_like(x))[0]
    torch.testing.assert_close(measured,identity)
    state=io.BytesIO();torch.save(frontend.state_dict(),state);state.seek(0)
    other=MeasuredPACFrontend(n_bands=4,hidden_dim=32,sample_rate=200,
                             patch_len=50,filter_range=(.5,75),measurement_mode='identity')
    other.load_state_dict(torch.load(state,weights_only=True))
    torch.testing.assert_close(frontend(x)[0],other(x)[0])


def test_out_of_band_leakage_has_no_valid_low_frequency_phase():
    t=torch.arange(1660)/200
    x=torch.cos(2*math.pi*32*t)[None,None]
    result=PhaseAmplitudeMeasurement([2.1,32.])(x)
    assert not result.valid.any()


def test_reason_codes_and_nonfinite_safety():
    m=PhaseAmplitudeMeasurement([6.,40.])
    t=torch.arange(1000)/200
    x=(torch.cos(2*math.pi*6*t)+(1+.5*torch.cos(2*math.pi*6*t))*torch.cos(2*math.pi*40*t))[None,None]
    result=m(x)
    assert torch.equal(result.reason_code==0,result.valid)
    assert (result.reason_code[...,1,0]==1).all()
    bad=x.clone();bad[...,300]=float('nan')
    assert (m(bad).reason_code==6).all()


def test_notch_excludes_source_and_modulation_support():
    clear=PhaseAmplitudeMeasurement([6.,40.,60.])
    notched=PhaseAmplitudeMeasurement([6.,40.,60.],excluded_intervals=((56.,64.),))
    assert clear.eligible[0,2] and not notched.eligible[0,2]
    assert clear.eligible[0,1] and notched.eligible[0,1]
    assert not notched.eligible[:,2].any()
    assert not PhaseAmplitudeMeasurement([6.,40.],excluded_intervals=((5.,7.),)).eligible.any()


def test_checkpoint_restores_physical_eligibility_and_context_settings():
    # Both banks allocate the same kernel shape, but have different eligibility
    # and context settings; merely copying the buffers would leave stale caches.
    original=PhaseAmplitudeMeasurement([4.,20.,40.,60.],min_cycles=4,excluded_intervals=((56.,64.),))
    restored=PhaseAmplitudeMeasurement([4.,20.,40.,60.],excluded_intervals=((90.,91.),))
    restored.load_state_dict(original.state_dict())
    t=torch.arange(1000)/200
    x=(torch.cos(2*math.pi*4*t)+(1+.5*torch.cos(2*math.pi*4*t))*torch.cos(2*math.pi*40*t))[None,None]
    a,b=original(x),restored(x)
    assert restored.min_cycles==4 and torch.equal(a.valid,b.valid)
    torch.testing.assert_close(a.depth,b.depth)
    # Frozen first-release weights remain loadable with matching config.
    legacy={k:v for k,v in original.state_dict().items() if k!='_settings'}
    same=PhaseAmplitudeMeasurement([4.,20.,40.,60.],min_cycles=4,excluded_intervals=((56.,64.),))
    same.load_state_dict(legacy)
    torch.testing.assert_close(a.depth,same(x).depth)
