"""Short capped-memory CUDA functional check, no tokenizer/model training."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator

root=Path(__file__).parent;data=np.load(root/'engineering-inputs.npz')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.06,0)
report=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            memory_fraction_cap=.06,input_sha256=hashlib.sha256((root/'engineering-inputs.npz').read_bytes()).hexdigest(),records={})
op=CFCOperator(block_cycles=1,shrink=False)
original={k:v.clone() for k,v in op.named_buffers()}
torch.nn.Sequential(op).to(dtype=torch.bfloat16)
for k,v in op.named_buffers():
    torch.testing.assert_close(v,original[k],atol=0,rtol=0)
op=op.cuda()
report['parent_bfloat16_preserves_physical_state']=True
for name in data.files:
    x=torch.tensor(data[name][:1],device='cuda',requires_grad=True)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=op(x);packet=out.relation_features();features=packet['features']
        features.square().mean().backward()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(x.grad).all())
    gpu_value=features.detach().cpu()
    cpu=CFCOperator(block_cycles=1,shrink=False)
    with torch.no_grad():expected=cpu(torch.tensor(data[name][:1])).relation_features()['features']
    error=float((gpu_value-expected).abs().max())
    torch.testing.assert_close(gpu_value,expected,atol=3e-4,rtol=3e-3)
    report['records'][name]=dict(shape=list(x.shape),finite_features=True,finite_gradient=True,
        cpu_cuda_max_abs_error=error,wall_seconds_including_cpu_reference=time.monotonic()-start,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
        physical_valid_fraction=float(out.valid.float().mean()))
    del x,out,features,packet;torch.cuda.empty_cache()
(root/'gpu-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
