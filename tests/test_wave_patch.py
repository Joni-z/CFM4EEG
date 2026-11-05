"""The waveform control preserves electrodes and reuses the actual encoder."""
from pathlib import Path

import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.triaxial import FreqNone, FreqAttention


ROOT = Path(__file__).resolve().parents[1]


def test_wave_control_uses_raw_per_electrode_patches_and_same_encoder():
    full_cfg = yaml.safe_load((ROOT / 'configs/model/tuev-scratch.yaml').read_text())
    wave_cfg = yaml.safe_load((ROOT / 'configs/model/tuev-wave-patch.yaml').read_text())
    permitted = {'name', 'model', 'model_kwargs', 'reproduction'}
    assert {k: v for k, v in full_cfg.items() if k not in permitted} == {
        k: v for k, v in wave_cfg.items() if k not in permitted}
    for cfg in (full_cfg, wave_cfg):
        cfg['model_kwargs'] = {**cfg['model_kwargs'], 'depth': 1}
    full, wave = [build_model(cfg, (16, 1000)) for cfg in (full_cfg, wave_cfg)]
    assert {k: v.shape for k, v in full.encoder.state_dict().items()} == {
        k: v.shape for k, v in wave.encoder.state_dict().items()}
    assert all(isinstance(block.freq, FreqNone) for block in wave.encoder.blocks)
    wave.eval()
    x = torch.randn(2, 16, 1000)
    shapes = []
    hook = wave.encoder.register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape))
    logits = wave(x)
    hook.remove()
    assert shapes == [torch.Size([2, 16, 1, 20, 128])]
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    logits.square().sum().backward()
    assert wave.frontend.weight.grad is not None
    assert torch.isfinite(wave.frontend.weight.grad).all()
    # Linear projection shared across electrodes; neither mixes nor filters them.
    patches = x.unfold(-1, 50, 50)
    torch.testing.assert_close(wave.frontend(patches)[:, 4, 3],
                               wave.frontend(x[:, 4, 150:200]))


def test_three_axis_raw_control_trains_without_measured_coupling(tmp_path):
    cfg = dict(model="wave_patch_threeaxis", dataset="tuev", num_classes=6,
               model_kwargs=dict(d_model=16, depth=1, n_heads=4, dropout=0.,
                                 patch_len=50, n_rows=4, freq_mixer="attention"))
    model = build_model(cfg, (3, 100))
    x = torch.randn(2, 3, 100)
    shapes = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape))
    logits = model(x)
    hook.remove()
    assert shapes == [torch.Size([2, 3, 4, 2, 16])]
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    assert all(isinstance(block.freq, FreqAttention) for block in model.encoder.blocks)
    assert isinstance(model.frontend, torch.nn.Linear)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.frontend.weight.detach().clone()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (model.frontend.weight.grad.reshape(4, -1).abs().sum(1) > 0).all()
    optimizer.step()
    assert not torch.equal(before, model.frontend.weight)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(cfg, (3, 100))
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    model.eval(); restored.eval()
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
