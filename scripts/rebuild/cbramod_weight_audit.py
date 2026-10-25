"""Observe the author's initialized backbone without changing its loading path."""
from __future__ import annotations

import hashlib
from pathlib import Path


def verify_pretrained_backbone(model, checkpoint: Path, expected_sha256: str) -> dict:
    import torch

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Released CBraMod checkpoint hash differs from the reviewed config")
    released = torch.load(checkpoint, map_location="cpu", weights_only=True)
    active = model.backbone.state_dict()
    # The unchanged author task model replaces only the pretraining output
    # projection with Identity after strictly loading the complete checkpoint.
    excluded = set(released) - set(active)
    if set(active) - set(released) or not all(k.startswith("proj_out.") for k in excluded):
        raise RuntimeError("Unexpected pretrained backbone key coverage")
    for key, value in active.items():
        torch.testing.assert_close(value.detach().cpu(), released[key], rtol=0, atol=0)
    return {
        "sha256": digest,
        "active_tensors_verified": len(active),
        "active_numel_verified": sum(v.numel() for v in active.values()),
        "removed_pretraining_projection": sorted(excluded),
        "all_active_tensors_equal_released_weights": True,
        "trainable_backbone_parameters": sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
    }
