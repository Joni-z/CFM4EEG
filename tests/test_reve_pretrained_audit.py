"""HF-generated model state cannot hide missing or corrupted released tensors."""
import hashlib

import pytest
import torch
from safetensors.torch import save_file

from scripts.rebuild.reve_pretrained_audit import verify_initialization


def payload(tmp_path, *, omitted_query=False, extra=False):
    encoder=torch.nn.Linear(2,2)
    query=torch.randn(1,1,2)
    state={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    if not omitted_query:state['cls_query_token']=query.clone()
    if extra:state['unrecognized_tensor']=torch.zeros(1)
    path=tmp_path/'model.safetensors';save_file(state,str(path))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return encoder,query,path,digest


def test_exact_raw_encoder_and_query_are_required(tmp_path):
    encoder,query,path,digest=payload(tmp_path)
    result=verify_initialization(encoder,query,path,digest)
    assert result['strict_encoder_key_coverage'] and result['all_encoder_tensors_equal'] and result['query_exact']
    with torch.no_grad():encoder.bias.add_(1)
    with pytest.raises(ValueError,match='encoder differs'):
        verify_initialization(encoder,query,path,digest)


def test_hash_query_and_unexpected_keys_fail(tmp_path):
    encoder,query,path,digest=payload(tmp_path)
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_initialization(encoder,query,path,'0'*64)
    with pytest.raises(ValueError,match='query'):
        verify_initialization(encoder,query+1,path,digest)
    encoder,query,path,digest=payload(tmp_path,omitted_query=True)
    with pytest.raises(ValueError,match='query'):
        verify_initialization(encoder,query,path,digest)
    encoder,query,path,digest=payload(tmp_path,extra=True)
    with pytest.raises(ValueError,match='coverage'):
        verify_initialization(encoder,query,path,digest)


def test_nonfinite_released_values_fail_even_when_equal(tmp_path):
    encoder,query,path,digest=payload(tmp_path)
    query.fill_(float('inf'))
    state={k:v.detach().clone() for k,v in encoder.state_dict().items()};state['cls_query_token']=query
    save_file(state,str(path));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Nonfinite'):
        verify_initialization(encoder,query,path,digest)
