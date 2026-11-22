"""Residual transport of band content under measured phase relations.

The phase reference acts on projected features, not on physical EEG time.
The source content remains fixed in the analytic-reference invariance test.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class GeometricContentFusion(nn.Module):
    """Transport gated complex content, then add a small real token residual.

    All three modes allocate the same parameters. ``no_alignment`` removes
    only the relation angle in this branch; ``ordinary`` aggregates the same
    gated content without a phase carrier. The original token path is intact.
    """

    def __init__(self, hidden_dim: int, n_bands: int, mode: str = "aligned",
                 init_scale: float = 0.01, max_scale: float = 0.1):
        super().__init__()
        if mode not in ("aligned", "no_alignment", "ordinary"):
            raise ValueError(f"Unknown geometric fusion mode: {mode}")
        if hidden_dim % 2 or not 0 < init_scale < max_scale:
            raise ValueError("Even hidden_dim and 0 < init_scale < max_scale required")
        self.mode = mode
        self.max_scale = float(max_scale)
        self.content_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        # No bias: absent sources must produce an exactly zero residual.
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.output_projection.weight)
        self.scale_logit = nn.Parameter(torch.full(
            (n_bands, 1), math.atanh(init_scale / max_scale)))

    def aggregate(self, phase: torch.Tensor, amplitude: torch.Tensor,
                  waveform: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Inputs (B,C,P,band,K), Z (B,C,P,source,target); output complex."""
        content = torch.complex(amplitude, waveform)
        gate = self.content_gate(torch.cat((amplitude, waveform), dim=-1))
        value = gate * content
        if self.mode != "ordinary":
            value = value * (phase / phase.abs().clamp_min(1e-6))

        edge = relations.transpose(-2, -1)  # target, source
        bands = edge.shape[-1]
        valid = torch.ones(bands, bands, device=edge.device,
                           dtype=torch.bool).tril(-1)
        strength = edge.abs() * valid
        total = strength.sum(dim=-1, keepdim=True)
        alpha = strength / total.clamp_min(1e-8)
        # Unmeasured relations transport nothing. The original path retains
        # its historical fallback; this branch makes no unaligned fallback.
        alpha = torch.where(total > 1e-8, alpha, torch.zeros_like(alpha))
        coefficient = alpha.to(edge.dtype)
        if self.mode == "aligned":
            coefficient = coefficient * (edge / edge.abs().clamp_min(1e-8)).conj()
        return torch.einsum("bcpji,bcpik->bcpjk", coefficient, value)

    def forward(self, phase: torch.Tensor, amplitude: torch.Tensor,
                waveform: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        message = self.aggregate(phase, amplitude, waveform, relations)
        # Same real/imag interleaving as the existing token interface.
        residual = self.output_projection(torch.view_as_real(message).flatten(-2))
        scale = self.max_scale * torch.tanh(self.scale_logit)
        residual = residual * scale
        return torch.view_as_complex(residual.reshape(*message.shape, 2).contiguous())
