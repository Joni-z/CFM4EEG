from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.rebuild.native_cbramod_entry import adapt_iiic_classifier, make_array_loader


def test_iiic_head_accepts_full_ten_seconds_without_hidden_width_change():
    model=SimpleNamespace(classifier=torch.nn.Sequential(
        torch.nn.Flatten(1), torch.nn.Linear(16000,1000,device='meta'),
        torch.nn.ELU(), torch.nn.Linear(1000,200,device='meta'),
        torch.nn.ELU(), torch.nn.Linear(200,6,device='meta')))
    tail=list(model.classifier.children())[2:]
    adapt_iiic_classifier(model)
    assert model.classifier(torch.empty(2,16,10,200,device='meta')).shape==(2,6)
    assert list(model.classifier.children())[2:]==tail


def test_unknown_native_head_cannot_be_silently_adapted():
    model=SimpleNamespace(classifier=torch.nn.Sequential(
        torch.nn.Identity(),torch.nn.Linear(10,5,device='meta')))
    with pytest.raises(ValueError,match='layout changed'):
        adapt_iiic_classifier(model)


def test_smoke_keeps_all_samples_units_and_labels_without_test_files(tmp_path):
    x=np.arange(8*16*2000,dtype=np.float32).reshape(8,16,2000)/1000
    y=np.arange(8,dtype=np.int64)%6
    for split in ['train','val']:
        np.save(tmp_path/f'{split}_signals.npy',x)
        np.save(tmp_path/f'{split}_labels.npy',y)
    class NativeDataset:
        def collate(self,batch):
            return torch.as_tensor(np.stack([r[0] for r in batch])),torch.tensor([r[1] for r in batch])
    loader=make_array_loader(SimpleNamespace(CustomDataset=NativeDataset),tmp_path,
                             smoke_steps=1,dataset='iiic')(SimpleNamespace(batch_size=2)).get_data_loader()
    sample,label=loader['val'].dataset[0]
    assert sample.shape==(16,10,200) and label==0
    np.testing.assert_array_equal(sample.reshape(16,2000),x[0])
    with pytest.raises(AssertionError,match='test data'):
        next(iter(loader['test']))
