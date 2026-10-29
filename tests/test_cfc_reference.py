"""New measurements enter the original content/rotation model without extra weights."""
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.frontend.cfc import CFCFrontend
from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.training.train import cpu_state_dict


def signal():
    t = torch.arange(1000)/200
    ph = 2*torch.pi*4*t
    return (torch.cos(ph)+(1+.4*torch.cos(ph+.7)+.2*torch.cos(2*ph-.3))*
            torch.cos(2*torch.pi*32*t))[None, None]


def pair():
    torch.set_num_threads(2)
    kw = dict(n_bands=16, hidden_dim=32, sample_rate=200, patch_len=50,
              filter_range=(.5,75.), tokenizer_mode='pac_interaction',
              interaction_mode='rotation', waveform_modulation='quadrature')
    torch.manual_seed(0);old = TriAxialFrontend(**kw)
    torch.manual_seed(0);new = CFCFrontend(**kw)
    return old,new


def test_unsupported_measurements_reproduce_original_exactly():
    old,new = pair();x=signal();m=new.measurement(x)
    for unavailable in [replace(m,valid=torch.zeros_like(m.valid)),
                        replace(m,raw_coef=torch.zeros_like(m.raw_coef))]:
        with patch.object(new.measurement,'forward',return_value=unavailable):
            torch.testing.assert_close(new(x)[0],old(x)[0],atol=0,rtol=0)
    mapping=new.target_mapping()
    assert bool(((mapping==0)|(mapping>=.5)).all())
    # Correspondence is physical, not a silently normalized remote target.
    assert torch.count_nonzero(mapping[0])==0


def test_supported_measurement_changes_only_eligible_reference_and_preserves_content():
    old,new=pair();x=signal();m=new.measurement(x)
    reference,available=new.measured_reference(m)
    assert available.any() and (~available).any()
    legacy=old(x)[0];output=new(x)[0];mask=available.permute(0,1,3,2)
    torch.testing.assert_close(output[~mask],legacy[~mask],atol=0,rtol=0)
    assert float((output[mask]-legacy[mask]).detach().abs().max())>1e-5
    old_norm=legacy.reshape(*legacy.shape[:-1],-1,2).square().sum(-1)
    new_norm=output.reshape(*output.shape[:-1],-1,2).square().sum(-1)
    torch.testing.assert_close(new_norm,old_norm,rtol=2e-5,atol=2e-6)
    assert bool((reference.real[available]<0).any())  # Full circle, no identity anchor.
    h2_changed=m.raw_coef.clone();h2_changed[...,1]*=100j
    same,_=new.measured_reference(replace(m,raw_coef=h2_changed))
    torch.testing.assert_close(same,reference,rtol=0,atol=0)
    # Independent source reference changes cancel, without treating H2 as H1.
    shift=torch.linspace(-1,1,m.source_phase.shape[2]);k=torch.arange(1,3)
    changed=replace(m,source_phase=m.source_phase*torch.exp(1j*shift)[None,None,:,None],
        raw_coef=m.raw_coef*torch.exp(1j*shift[:,None]*k)[None,None,None,:,None,:])
    shifted,valid=new.measured_reference(changed)
    torch.testing.assert_close(shifted,reference,rtol=3e-5,atol=3e-6)
    assert torch.equal(valid,available)


def test_recipe_and_initialization_equal_original():
    for ds in ['tuev','chbmit']:
        original=yaml.safe_load(open(f'configs/model/{ds}-scratch.yaml'))
        cfg=yaml.safe_load(open(f'configs/model/{ds}-cfc-reference.yaml'))
        excluded={'name','group','reproduction','preload','model_kwargs'}
        assert {k:v for k,v in original.items() if k not in excluded}=={
            k:v for k,v in cfg.items() if k not in excluded}
        assert {k:v for k,v in original['model_kwargs'].items() if k!='tokenizer_mode'}=={
            k:v for k,v in cfg['model_kwargs'].items() if k!='tokenizer_mode'}
        torch.manual_seed(0);old=build_model(original,(16,1000 if ds=='tuev' else 2000))
        torch.manual_seed(0);new=build_model(cfg,(16,1000 if ds=='tuev' else 2000))
        assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)


def test_gradients_and_checkpoint(tmp_path):
    cfg=yaml.safe_load(open('configs/model/tuev-cfc-reference.yaml'))
    cfg['model_kwargs'].update(d_model=32,depth=1,dropout=0.,augmentations=[])
    torch.set_num_threads(2);torch.manual_seed(0);model=build_model(cfg,(16,1000))
    x=signal().expand(1,16,-1).clone();F.cross_entropy(model(x),torch.tensor([2])).backward()
    for name in ['phase_tokenizer.weight','amplitude_tokenizer.weight','waveform_quadrature.weight']:
        grad=dict(model.frontend.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and (grad!=0).any(),name
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():expected=model(x)
    torch.save(cpu_state_dict(model),tmp_path/'state.pt')
    restored=build_model(cfg,(16,1000)).eval()
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True),strict=True)
    with torch.no_grad():torch.testing.assert_close(restored(x),expected,rtol=1e-6,atol=1e-6)
