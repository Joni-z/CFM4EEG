"""Small, explicit input/montage adaptations outside official training loops."""
from __future__ import annotations
import numpy as np
import torch


def resample_window(x: torch.Tensor, source_rate: int, target_rate: int):
    """Native TFM uses scipy Fourier resampling; preserve duration exactly."""
    if source_rate == target_rate:
        return x
    from scipy.signal import resample
    length = x.shape[-1] * target_rate / source_rate
    if int(length) != length:
        raise ValueError('Resampling would change window duration')
    value = resample(x.numpy(), int(length), axis=-1).astype(np.float32)
    return torch.from_numpy(value)


def recover_uv(x: torch.Tensor, manifest: dict):
    norm = manifest.get('protocol', {}).get('normalization')
    if norm == 'div100':
        return x * 100.
    if norm == 'train_stats':
        qc = manifest.get('qc', {})
        mean = torch.as_tensor(qc['norm_mean_per_channel'],dtype=x.dtype).reshape(-1,1)
        std = torch.as_tensor(qc['norm_std_per_channel'],dtype=x.dtype).reshape(-1,1)
        if len(mean)!=x.shape[-2] or not torch.isfinite(std).all() or (std<=0).any():
            raise ValueError('Invalid per-channel training normalization metadata')
        return x * std + mean
    raise ValueError(f'Physical units cannot be recovered from {norm}')


def extend_channel_embeddings(encoder, channels: int):
    """Extend native TFM trainable montage embeddings, preserving existing rows."""
    old = encoder.channel_embed
    if channels <= old.num_embeddings:
        return encoder
    extended = torch.nn.Embedding(channels,old.embedding_dim,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad(): extended.weight[:old.num_embeddings].copy_(old.weight)
    encoder.channel_embed = extended
    encoder.index = torch.nn.Parameter(torch.arange(channels,device=old.weight.device),requires_grad=False)
    return encoder


def expand_encoder_state(state: dict, channels: int):
    """Adapt released16-channel state only; new rows remain explicitly random."""
    key='channel_embed.weight'
    old=state[key]
    if old.shape[0]==channels:
        return state
    if old.shape[0]>channels:
        raise ValueError('Do not silently discard pretrained channel embeddings')
    new=torch.nn.Embedding(channels,old.shape[1],dtype=old.dtype).weight.detach()
    new[:old.shape[0]].copy_(old)
    result=dict(state);result[key]=new;result['index']=torch.arange(channels,dtype=state['index'].dtype)
    return result


def tfm_stage_input_config(cfg: dict, stage: str):
    """Hash only weights consumed by a stage, never that stage's outputs."""
    import copy
    result=copy.deepcopy(cfg)
    if stage=='tokenizer':
        result['pretrained']=False
        result.pop('tokenizer_path',None);result.pop('encoder_path',None)
    elif stage=='mtp':
        result['pretrained']=True
        result.pop('encoder_path',None)
    elif stage=='finetune':
        result['pretrained']=True
        if cfg.get('tfm_setting')=='multiple' and not cfg.get('tokenizer_path'):
            from scripts.rebuild.tfm_weights import weight_paths
            result['tokenizer_path'],result['encoder_path']=weight_paths(cfg['dataset'],'multiple')
    else:raise ValueError(stage)
    return result
