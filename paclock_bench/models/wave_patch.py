"""Waveform-patch control using CfFT's existing two-axis encoder.

One shared linear map embeds each electrode's non-overlapping raw patch.
No filterbank, spectral features, measured relations or synthetic band copies.
"""
from __future__ import annotations

import torch
from torch import nn

from .paclock.augment import RandomAugment
from .paclock.head import ClassificationHead
from .paclock.montage import coords_for
from .paclock.triaxial import SpatialPE, TriAxialEncoder


class WavePatchTwoAxis(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        if cfg.get("freq_mixer", "none") != "none":
            raise ValueError("WavePatchTwoAxis has no frequency attention axis")
        self.patch_len = int(cfg.get("patch_len", 50))
        if self.patch_len <= 0 or cfg["seq_len"] % self.patch_len:
            raise ValueError("Waveform patches must divide the input length exactly")
        width = cfg["d_model"]
        self.augment = RandomAugment(cfg.get("augmentations", []))
        self.frontend = nn.Linear(self.patch_len, width)
        coords = coords_for(cfg["dataset"]) if cfg.get("spatial_pe") == "xyz" else None
        if coords is not None and len(coords) != cfg["n_channels"]:
            raise ValueError("Spatial coordinates and input channels differ")
        self.spatial_pe = SpatialPE(cfg["n_channels"], width, coords=coords)
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=width, freq_mixer="none",
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", .1),
            space_over_bands=True, block_variant=cfg.get("block_variant", "legacy"),
        )
        self.head = ClassificationHead(width, cfg["num_classes"], mode="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        batch, channels, length = x.shape
        if length % self.patch_len:
            raise ValueError("Input length is not divisible by patch_len")
        # Singleton grid row only adapts the existing encoder interface.
        tokens = self.frontend(x.reshape(batch, channels, length // self.patch_len,
                                         self.patch_len)).unsqueeze(2)
        tokens = tokens + self.spatial_pe(channels, x.device).view(1, channels, 1, 1, -1)
        encoded = self.encoder(tokens, coupling=None)
        return self.head(encoded.flatten(1, 3))
