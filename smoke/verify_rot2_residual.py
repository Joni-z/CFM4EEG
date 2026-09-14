"""Allocated-GPU contract check, followed separately by real-batch timing."""
import copy
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
import torch
import yaml
from paclock_bench.models.build import build_model

assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
records = []
for dataset, shape in [('tuev', (16, 1000)), ('sleepedf', (2, 3000))]:
    cfg = yaml.safe_load(Path(f'configs/rot2_residual/{dataset}_control_s0.yaml').read_text())
    altered = copy.deepcopy(cfg)
    altered['model_kwargs']['local_residual'] = True
    torch.manual_seed(0)
    baseline = build_model(cfg, shape).cuda()
    torch.manual_seed(0)
    residual = build_model(altered, shape).cuda()
    shared, extra = baseline.state_dict(), residual.state_dict()
    assert set(extra) - set(shared) == {'frontend.local_projection.weight'}
    assert all(torch.equal(v, extra[k]) for k, v in shared.items())
    delta = sum(p.numel() for p in residual.parameters()) - sum(p.numel() for p in baseline.parameters())
    assert delta == 6400
    x = torch.randn(2, *shape, device='cuda')
    baseline.eval(); residual.eval()
    with torch.no_grad():
        assert torch.equal(baseline(x), residual(x))
        assert torch.isfinite(residual(torch.zeros_like(x))).all()
    baseline.train(); residual.train()
    torch.manual_seed(7); a = baseline(x)
    torch.manual_seed(7); b = residual(x)
    assert torch.equal(a, b)
    loss = b.square().mean()
    loss.backward()
    projection = residual.frontend.local_projection.weight
    assert torch.isfinite(projection.grad).all() and projection.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in residual.parameters())
    torch.optim.AdamW(residual.parameters(), lr=1e-4).step()
    assert projection.abs().sum() > 0
    residual.eval()
    clone = build_model(altered, shape).cuda().eval()
    clone.load_state_dict(residual.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(clone(x), residual(x))
    records.append(dict(dataset=dataset, extra_parameters=delta, initial_logits_exact=True,
                        shared_initial_state_exact=True, train_dropout_exact=True,
                        residual_gradient_nonzero=True, checkpoint_roundtrip_exact=True))
    del baseline, residual, clone, x, a, b, loss, shared, extra, projection
    torch.cuda.empty_cache()
path = Path(f'results/rot2-residual-contract-{os.environ["SLURM_JOB_ID"]}.json')
path.write_text(json.dumps(records, indent=2) + '\n')
print(path.read_text(), flush=True)
