import torch
import yaml
from paclock_bench.models.build import build_model
from scripts.experiments.pac_singlepath import SinglePathPAC


def test_exact_candidate_and_encoder_initialization():
 torch.set_num_threads(2)
 cfg=yaml.safe_load(open('configs/model/tuev-scratch.yaml'))
 cfg['model_kwargs'].update(depth=1,d_model=32,dropout=0.,augmentations=[])
 torch.manual_seed(0);old=build_model(cfg,(16,1000))
 cfg['model_kwargs']['tokenizer_mode']='cfc_direct'
 torch.manual_seed(0);new=build_model(cfg,(16,1000))
 for key,value in old.encoder.state_dict().items():torch.testing.assert_close(value,new.encoder.state_dict()[key],rtol=0,atol=0)
 assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
 x=torch.randn(2,16,1000)*.2
 torch.testing.assert_close(new.frontend(x)[0],SinglePathPAC(old.frontend)(x,mode='direct'),rtol=0,atol=0)
 loss=torch.nn.functional.cross_entropy(new(x),torch.tensor([1,2]));loss.backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
 assert new.frontend.original.waveform_quadrature.weight.grad.abs().sum()>0
 restored=build_model(cfg,(16,1000));restored.load_state_dict(new.state_dict(),strict=True)
 torch.testing.assert_close(restored.eval()(x),new.eval()(x),rtol=0,atol=0)


def test_f_keeps_waveform_component_unmodulated():
 from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
 from scripts.experiments.pac_singlepath import SinglePathPAC
 f=TriAxialFrontend(n_bands=16,hidden_dim=32,sample_rate=200,patch_len=50,filter_range=(.5,75.),tokenizer_mode='pac_interaction',interaction_mode='rotation',waveform_modulation='quadrature')
 m=SinglePathPAC(f);p=m.prepare(torch.randn(1,1,1000)*.2)
 p['content']=1j*p['content'].imag
 out=m.transform(p,'envelope_only')
 torch.testing.assert_close(out,torch.view_as_real(p['content']).flatten(-2),rtol=0,atol=0)
