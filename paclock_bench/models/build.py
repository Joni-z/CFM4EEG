"""Construct CfFT. Baselines execute their own original upstream trainers."""
from __future__ import annotations

import torch.nn as nn
from ..paths import expand


def build_model(cfg: dict, input_shape: tuple[int, ...]) -> nn.Module:
    if cfg["model"] != "paclock":
        raise ValueError(
            f"Baseline {cfg['model']!r} must use its original entrypoint under "
            "scripts/rebuild/. Generic baseline construction has been retired."
        )
    from .paclock.build import build_model as build_cfft, load_pretrained_backbone

    channels, length = input_shape[-2:]
    model_config = {
        **cfg.get("model_kwargs", {}),
        "n_channels": channels,
        "seq_len": length,
        "num_classes": cfg["num_classes"],
        "sample_rate": cfg.get("sample_rate", 200),
        "sampling_rate": cfg.get("sample_rate", 200),
        "dataset": cfg["dataset"],
    }
    model = build_cfft(model_config)
    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        report = load_pretrained_backbone(
            model, expand(checkpoint),
            exclude=tuple(cfg.get("checkpoint_exclude", ())),
        )
        print(f"[CfFT] loaded {len(report['loaded'])} tensors from {checkpoint}; "
              f"skipped shapes: {report['skipped_shape']}; "
              f"excluded: {report['skipped_excluded']}")
    return model


def count_params(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6
