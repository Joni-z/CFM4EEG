"""Tests of residual bounds, baseline preservation and readout semantics."""
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.probing.synthetic import generate_dataset
from scripts.experiments.pac_candidate_search import probe
from scripts.experiments.pac_metric_calibration import signal_reference
from scripts.experiments.pac_token_candidates import PACResidualCandidates


def test_raw_reference_recovers_both_harmonics_without_latent_inputs():
    t=np.arange(1000)/200;phi=2*np.pi*4*t+.2
    coeff=np.array([.25,.12,-.13,.18])
    a=1+coeff[0]*np.cos(phi)+coeff[1]*np.sin(phi)+coeff[2]*np.cos(2*phi)+coeff[3]*np.sin(2*phi)
    x=(np.cos(phi)+a*np.cos(2*np.pi*32*t-.4))[None,None]
    np.testing.assert_allclose(signal_reference(x)[0],coeff,atol=.025)


def test_residual_preserves_baseline_zero_and_bounds_each_record():
    torch.set_num_threads(2);torch.manual_seed(3)
    f=TriAxialFrontend(n_bands=16,hidden_dim=128,sample_rate=200,patch_len=50,
        filter_range=(.5,75.),tokenizer_mode='pac_interaction',interaction_mode='rotation',waveform_modulation='quadrature').eval()
    original={k:v.clone() for k,v in f.state_dict().items() if torch.is_tensor(v)}
    candidates=PACResidualCandidates(f)
    t=torch.arange(1000)/200;ph=2*torch.pi*4*t
    x=(torch.cos(ph)+(1+.4*torch.cos(ph))*torch.cos(2*torch.pi*32*t))[None,None]*.2
    b,*parts=candidates.components(x)
    torch.testing.assert_close(b,f(x)[0],rtol=0,atol=0)
    torch.testing.assert_close(candidates(x,gain=0.),b,rtol=0,atol=0)
    for delta in parts[:3]:
        assert torch.equal(candidates.fuse(b,delta,0.,.1),b)
        out=candidates.fuse(b,delta,.1,.1)
        assert torch.isfinite(out).all()
        assert float((out-b).norm()/b.norm())<=.100001
    for key,value in original.items():
        torch.testing.assert_close(f.state_dict()[key],value,rtol=0,atol=0)


def test_linear_kernel_matches_affine_ridge():
    data=generate_dataset(40,16,16,seed=58)
    rng=np.random.default_rng(17);features=rng.normal(size=(72,18))
    data['curves']=rng.normal(size=(72,24))
    result,pred=probe(features,data,degree=1)
    scaler=StandardScaler().fit(features[:40])
    direct=Ridge(alpha=result['alpha']*18,solver='svd').fit(scaler.transform(features[:40]),data['curves'][:40])
    np.testing.assert_allclose(pred,direct.predict(scaler.transform(features[56:])),atol=1e-8)


def test_readout_choice_does_not_use_test_targets():
    data=generate_dataset(40,16,16,seed=59)
    x=np.random.default_rng(1).normal(size=(72,12))
    first,_=probe(x,data,degree=2)
    changed=dict(data);changed['curves']=data['curves'].copy();changed['curves'][56:]+=100
    second,_=probe(x,changed,degree=2)
    assert first['validation_mse']==second['validation_mse']
    assert first['alpha']==second['alpha']
