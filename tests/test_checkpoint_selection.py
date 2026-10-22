"""End-to-end guard against confusing native terminal and validation-best weights."""
import json
import sys
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from paclock_bench.training import train

@pytest.mark.parametrize('policy,scores,selected_epoch', [
    ('terminal', [.8, .5, .4], 2),
    ('validation_best', [.8, .5, .4], 0),
    ('validation_best', [.8, .8, .4], 0),
])
def test_reported_weights_follow_recipe_not_validation_peak(tmp_path, monkeypatch, policy, scores, selected_epoch):
    cfg = dict(name='selection-check', dataset='tuev', model='test', num_classes=6,
               device='cpu', epochs=3, batch_size=2, lr=.01, weight_decay=0,
               optimizer='adam', loss='cross_entropy', patience=0,
               checkpoint_selection=policy, data_root=str(tmp_path), run_monitor=True)
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    data=TensorDataset(torch.tensor([[1.,0.],[0.,1.],[1.,1.],[.5,1.]]),
                       torch.tensor([0,1,2,1]))
    loaders=[DataLoader(data,batch_size=2) for _ in range(3)]
    info=dict(input_shape=(2,),n_samples=dict(train=4,val=4,test=4),
              class_counts=dict(train=[1,2,1,0,0,0],val=[1,2,1,0,0,0]),manifest={})
    monkeypatch.setattr(train,'build_dataloaders',lambda _: (*loaders,info))
    monkeypatch.setattr(train,'build_model',lambda *_:torch.nn.Linear(2,6))
    snapshots=[]; tested=[]
    def evaluate(model, loader, *args, **kwargs):
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if loader is loaders[1]:
            snapshots.append(state)
            score=scores[len(snapshots)-1]
        else:
            tested.append(state);score=.3
        return .5, {'cohen_kappa':score}, np.zeros((4,6)),np.array([0,1,2,1])
    monkeypatch.setattr(train,'evaluate',evaluate)
    monkeypatch.setattr(sys,'argv',['train','--config',str(path),'--out',str(tmp_path/'runs')])
    train.main()
    assert len(snapshots)==3 and len(tested)==1
    for k in tested[0]:
        torch.testing.assert_close(tested[0][k],snapshots[selected_epoch][k],rtol=0,atol=0)
    assert not torch.equal(snapshots[0]['weight'],snapshots[-1]['weight'])
    out=tmp_path/'runs/selection-check/seed0'
    result=json.loads((out/'result.json').read_text())
    assert result['selected_validation']['evaluation_index']==selected_epoch
    checkpoint=torch.load(out/result['selected_checkpoint'],weights_only=False)
    for k in tested[0]:
        torch.testing.assert_close(checkpoint['model'][k],tested[0][k],rtol=0,atol=0)
    assert (out/'best.pt').exists()
