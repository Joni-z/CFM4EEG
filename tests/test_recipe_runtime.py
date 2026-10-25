import math
import numpy as np
import torch
from paclock_bench.training.recipe_runtime import make_schedule, normalize_gradients, update_count
from paclock_bench.training.param_groups import layer_decay_param_groups


def test_accumulated_short_tail_matches_full_batch_gradient():
    torch.manual_seed(7)
    a=torch.nn.Linear(3,2).double(); b=torch.nn.Linear(3,2).double(); b.load_state_dict(a.state_dict())
    x=torch.randn(11,3,dtype=torch.double); y=torch.randn(11,2,dtype=torch.double)
    oa=torch.optim.SGD(a.parameters(),lr=.07); ob=torch.optim.SGD(b.parameters(),lr=.07)
    # Two update groups: 8 examples and the remaining 3, the latter microbatch
    # has 3 examples. A nominal group-size division would give the wrong tail.
    for start,end in [(0,8),(8,11)]:
        oa.zero_grad();ob.zero_grad()
        torch.nn.functional.mse_loss(a(x[start:end]),y[start:end]).backward()
        oa.step()
        for j in range(start,end,4):
            n=min(4,end-j)
            (torch.nn.functional.mse_loss(b(x[j:j+n]),y[j:j+n])*n).backward()
        normalize_gradients(b.parameters(),end-start);ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=1e-13,atol=1e-13)
    assert update_count(3,2)==2


def test_cbramod_schedule_matches_upstream_each_update():
    a=torch.nn.Parameter(torch.ones(1));b=torch.nn.Parameter(torch.ones(1))
    oa=torch.optim.AdamW([{'params':[a],'lr':1e-4},{'params':[b],'lr':5e-4}])
    ob=torch.optim.AdamW([{'params':[torch.nn.Parameter(torch.ones(1))],'lr':1e-4},
                         {'params':[torch.nn.Parameter(torch.ones(1))],'lr':5e-4}])
    ours=make_schedule(oa,dict(scheduler='cosine_updates',epochs=3,lr=1e-4,scheduler_min_lr=1e-6),7)
    ref=torch.optim.lr_scheduler.CosineAnnealingLR(ob,T_max=21,eta_min=1e-6)
    for _ in range(21):
        assert [g['lr'] for g in oa.param_groups]==[g['lr'] for g in ob.param_groups]
        oa.step();ob.step();ours.step();ref.step()
    assert all(math.isclose(g['lr'],1e-6) for g in oa.param_groups)


def test_labram_upstream_array_and_time_embed_exclusion():
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__();self.time_embed=torch.nn.Parameter(torch.ones(1,16,4));self.head=torch.nn.Linear(4,2)
        def no_weight_decay(self):return {'time_embed','pos_embed','cls_token'}
        def get_num_layers(self):return 1
    model=Inner(); groups=layer_decay_param_groups(model,5e-4,.05,.65)
    assert next(g['weight_decay'] for g in groups if any(p is model.time_embed for p in g['params']))==0
    opt=torch.optim.AdamW(groups)
    cfg=dict(epochs=3,lr=5e-4,warmup_epochs=1,scheduler='labram_cosine',scheduler_min_lr=1e-6)
    sch=make_schedule(opt,cfg,4)
    expected=np.concatenate((np.linspace(0,5e-4,4),1e-6+.5*(5e-4-1e-6)*(1+np.cos(np.pi*np.arange(8)/8))))
    for i in range(12):
        np.testing.assert_allclose([g['lr'] for g in opt.param_groups],expected[i]*np.array(sch.scales))
        opt.step();sch.step()


def test_native_labram_drops_incomplete_effective_batch():
    # Upstream schedules floor(N/global_batch), even when more microbatches fit.
    from paclock_bench.training.recipe_runtime import update_count
    assert update_count(1069, 8, drop_incomplete=True) == 133
    assert update_count(1069, 8) == 134
    import pytest
    with pytest.raises(ValueError, match='no complete'):
        update_count(7, 8, drop_incomplete=True)


def test_pretrained_receipt_expands_cbramod_weight_environment(tmp_path, monkeypatch):
    import hashlib
    from paclock_bench.training.recipe_runtime import pretrained_payloads
    path = tmp_path / 'released.pth'
    path.write_bytes(b'checkpoint receipt fixture')
    monkeypatch.setenv('CFM_TEST_WEIGHTS', str(path))
    receipt = pretrained_payloads(dict(model='cbramod', pretrained=True,
                                     pretrained_path='$CFM_TEST_WEIGHTS'))
    assert receipt[str(path)]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt[str(path)]['size'] == path.stat().st_size


def test_pretrained_receipt_accepts_explicit_biot_weight_path(tmp_path, monkeypatch):
    import hashlib
    from paclock_bench.training.recipe_runtime import pretrained_payloads
    path = tmp_path / 'biot.ckpt'
    path.write_bytes(b'BIOT checkpoint receipt fixture')
    monkeypatch.setenv('CFM_TEST_BIOT_WEIGHTS', str(path))
    receipt = pretrained_payloads(dict(model='biot', checkpoint='$CFM_TEST_BIOT_WEIGHTS'))
    assert receipt[str(path)]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
