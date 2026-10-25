"""Verify released REVE initialization against the actual safetensors payload."""
from __future__ import annotations

import hashlib
from pathlib import Path


def verify_initialization(encoder, query, weights: Path, expected_sha256: str) -> dict:
    import torch
    from safetensors.torch import load_file

    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Released REVE safetensors hash mismatch')
    released = load_file(str(weights), device='cpu')
    reference_query = released.pop('cls_query_token', None)
    current = encoder.state_dict()
    if set(current) != set(released):
        raise ValueError('Released REVE encoder key coverage mismatch')
    if any(not torch.equal(current[k].detach().cpu(), released[k]) for k in current):
        raise ValueError('REVE encoder differs from released safetensors')
    if (reference_query is None or query is None or query.shape != reference_query.shape
            or not torch.equal(query.detach().cpu(), reference_query)):
        raise ValueError('Released REVE query missing or not exactly loaded')
    if not all(torch.isfinite(v).all() for v in released.values()) or not torch.isfinite(query).all():
        raise ValueError('Nonfinite released REVE parameters')
    return dict(status='passed', weights=str(weights), weights_sha256=digest,
                encoder_tensors=len(current), encoder_values=sum(v.numel() for v in current.values()),
                strict_encoder_key_coverage=True, all_encoder_tensors_equal=True,
                query_shape=list(query.shape), query_exact=True)
