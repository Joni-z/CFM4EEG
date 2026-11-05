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


class WavePatchThreeAxis(nn.Module):
    """Raw linear patch projections on a channel x content-row x time grid.

    Rows are learned linear views of the same raw patch, not frequency bands.
    This preserves the original three-axis encoder and grid size without sinc,
    Hilbert, envelope features, PAC, or a rotation. Frontend counts can differ.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        if cfg.get("freq_mixer") != "attention" or cfg.get("space_over_bands", False):
            raise ValueError("Three-axis waveform control requires separate row attention")
        if cfg.get("head", "mean") != "mean" or cfg.get("aux_recon_weight", 0):
            raise ValueError("Three-axis waveform control requires mean readout, no auxiliary loss")
        self.patch_len = int(cfg.get("patch_len", 50))
        self.n_rows = int(cfg.get("n_rows", 16))
        self.width = int(cfg["d_model"])
        if self.patch_len <= 0 or cfg["seq_len"] % self.patch_len or self.n_rows < 2:
            raise ValueError("Require divisible raw patches and at least two content rows")
        self.augment = RandomAugment(cfg.get("augmentations", []))
        self.frontend = nn.Linear(self.patch_len, self.n_rows * self.width)
        self.row_pe = nn.Embedding(self.n_rows, self.width)
        coords = coords_for(cfg["dataset"]) if cfg.get("spatial_pe") == "xyz" else None
        if coords is not None and len(coords) != cfg["n_channels"]:
            raise ValueError("Spatial coordinates and input channels differ")
        self.spatial_pe = SpatialPE(cfg["n_channels"], self.width, coords=coords)
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=self.width, freq_mixer="attention",
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", .1),
            space_over_bands=False, block_variant=cfg.get("block_variant", "legacy"),
        )
        self.head = ClassificationHead(self.width, cfg["num_classes"], mode="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        batch, channels, length = x.shape
        if length % self.patch_len:
            raise ValueError("Input length is not divisible by patch_len")
        patches = x.reshape(batch, channels, length // self.patch_len, self.patch_len)
        tokens = self.frontend(patches).reshape(
            batch, channels, length // self.patch_len, self.n_rows, self.width,
        ).permute(0, 1, 3, 2, 4).contiguous()
        tokens = tokens + self.row_pe.weight.view(1, 1, self.n_rows, 1, self.width)
        tokens = tokens + self.spatial_pe(channels, x.device).view(
            1, channels, 1, 1, self.width,
        )
        # The existing encoder reshapes this argument before calling attention;
        # FreqAttention ignores its values. No relation is measured or supplied.
        unused_coupling = x.new_zeros(batch, channels, length // self.patch_len, 1, 1)
        return self.head(self.encoder(tokens, coupling=unused_coupling).flatten(1, 3))
