"""Copula-guided band-content fusion, independent of the old PAC rotation.

Local empirical marginal CDFs are Gaussianized before estimating dependence.
AAC uses signed Gaussian-copula correlation; phase/amplitude dependence uses
the Gaussian multiple-correlation statistic for (cos phase, sin phase) and an
envelope. These are restricted dependence summaries, not the full copula or
an estimator of every possible nonlinear coupling. See Ince et al. (2017),
doi:10.1002/hbm.23471.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .analytic import hilbert, phase_amplitude
from .sinc import SincBandpass


@torch.no_grad()
def gaussian_copula_scores(x: torch.Tensor) -> torch.Tensor:
    """Empirical CDF midranks -> normal scores, along the final time axis.

    Average ranks handle ties without inventing a temporal ordering. A constant
    channel maps to zero. The measured guide is deliberately stop-gradient;
    filter/content parameters still learn through the content branch.
    """
    length = x.shape[-1]
    if length < 4:
        raise ValueError("Copula estimation needs at least four time samples")
    values, order = x.float().sort(dim=-1)
    left = torch.searchsorted(values.contiguous(), values.contiguous(), right=False)
    right = torch.searchsorted(values.contiguous(), values.contiguous(), right=True)
    cdf = (left.float() + right.float()) / (2 * length)
    scores = math.sqrt(2) * torch.erfinv(2 * cdf - 1)
    scores = torch.zeros_like(scores).scatter_(-1, order, scores)
    centered = scores - scores.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return centered / variance.clamp_min(1e-8).sqrt()


@torch.no_grad()
def copula_relations(phase: torch.Tensor, envelope: torch.Tensor,
                     band_centers: torch.Tensor, shrinkage: float = 0.05
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs (..., band, time); outputs (..., target band, source band).

    Shrink the full three-variable correlation matrix toward identity before
    computing phase/amplitude multiple correlation. This stabilizes short
    windows and nearly collinear phase coordinates. PAC is directional by
    frequency convention, not evidence of a causal influence.
    """
    if not 0 < shrinkage < 1:
        raise ValueError("shrinkage must be strictly between zero and one")
    amp = gaussian_copula_scores(envelope)
    real = gaussian_copula_scores(phase.real)
    imag = gaussian_copula_scores(phase.imag)
    length = envelope.shape[-1]
    attenuation = 1 - shrinkage
    aac = attenuation * (amp @ amp.transpose(-1, -2)) / length
    # Rows below are phase sources, columns are amplitude targets.
    ra = attenuation * (real @ amp.transpose(-1, -2)) / length
    ia = attenuation * (imag @ amp.transpose(-1, -2)) / length
    ri = attenuation * (real * imag).mean(dim=-1, keepdim=True)
    r_squared = (ra.square() + ia.square() - 2 * ri * ra * ia) / (1 - ri.square())
    phase_amp = r_squared.clamp(0, 1 - 1e-6)
    eligible = band_centers[:, None] < band_centers[None, :]
    phase_amp = phase_amp.masked_fill(~eligible, 0).transpose(-1, -2)
    diagonal = torch.eye(envelope.shape[-2], device=envelope.device, dtype=torch.bool)
    return aac.masked_fill(diagonal, 0), phase_amp


class CopulaFrontend(nn.Module):
    """Per-band waveform/envelope content plus copula-guided residual fusion."""

    def __init__(self, n_bands: int, hidden_dim: int, sample_rate: int,
                 kernel_size: int = 201, patch_len: int = 50,
                 filter_range: tuple[float, float] | None = None,
                 relation_window_samples: int = 200, fusion_heads: int = 4,
                 copula_shrinkage: float = 0.05, **_):
        super().__init__()
        if n_bands < 2 or hidden_dim % 2 or hidden_dim % fusion_heads:
            raise ValueError("Need >=2 bands, an even width divisible by fusion_heads")
        if relation_window_samples < patch_len or relation_window_samples % patch_len:
            raise ValueError("Relation window must be an integer multiple of token length")
        if not 0 < copula_shrinkage < 1:
            raise ValueError("copula_shrinkage must be in (0, 1)")
        self.n_bands = n_bands
        self.patch_len = patch_len
        self.relation_window_samples = relation_window_samples
        self.fusion_heads = fusion_heads
        self.copula_shrinkage = copula_shrinkage
        self.tokenizer_mode = "copula"
        self.sinc = SincBandpass(n_bands, sample_rate, kernel_size=kernel_size)
        if filter_range is not None:
            low, high = map(float, filter_range)
            if not 0.1 <= low < high < sample_rate / 2:
                raise ValueError("filter_range must lie strictly below Nyquist")
            edges = torch.logspace(math.log10(low), math.log10(high), n_bands + 1)
            if (edges.diff() <= 0.05).any():
                raise ValueError("filter_range produces bands narrower than .05 Hz")
            self.sinc.min_low_hz = 0.1
            self.sinc.min_band_hz = 0.05
            with torch.no_grad():
                self.sinc.low_hz_.copy_((edges[:-1] - 0.1).view(-1, 1))
                self.sinc.band_hz_.copy_((edges.diff() - 0.05).view(-1, 1))
        self.waveform_projection = nn.Linear(patch_len, hidden_dim // 2)
        self.envelope_projection = nn.Linear(patch_len, hidden_dim // 2)
        self.content_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        # Two interpretable relation coefficients per head; neither sign is fixed.
        self.relation_weight = nn.Parameter(torch.full((fusion_heads, 2), 0.1))
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))

    @property
    def n_token_bands(self) -> int:
        return self.n_bands

    def band_hz(self) -> torch.Tensor:
        low = self.sinc.min_low_hz + self.sinc.low_hz_.abs()
        high = (low + self.sinc.min_band_hz + self.sinc.band_hz_.abs()).clamp_max(
            self.sinc.sample_rate / 2)
        return torch.cat(((low + high) / 2, high - low), dim=1)

    def token_band_hz(self) -> torch.Tensor:
        return self.band_hz()

    def forward(self, x: torch.Tensor, return_amp_target: bool = False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x.float(), return_amp_target)

    def _forward_fp32(self, x: torch.Tensor, return_amp_target: bool = False):
        batch, channels, length = x.shape
        window, patch = self.relation_window_samples, self.patch_len
        if length % window or length % patch:
            raise ValueError("Input length must be divisible by relation and token windows")
        bands, patches, windows = self.n_bands, length // patch, length // window
        filtered = self.sinc(x.reshape(batch * channels, 1, length)).reshape(
            batch, channels, bands, length)
        phase, envelope = phase_amplitude(hilbert(filtered))
        waveform = self.waveform_projection(filtered.reshape(batch, channels, bands, patches, patch))
        amplitude = self.envelope_projection(torch.log1p(envelope).reshape(
            batch, channels, bands, patches, patch))
        content = torch.cat((amplitude, waveform), dim=-1).permute(0, 1, 3, 2, 4)
        shape = (batch, channels, bands, windows, window)
        phase_windows = phase.reshape(shape).transpose(-3, -2)
        amp_windows = envelope.reshape(shape).transpose(-3, -2)
        aac, phase_amp = copula_relations(
            phase_windows, amp_windows, self.band_hz()[:, 0], self.copula_shrinkage)
        relations = torch.stack((aac, phase_amp), dim=-1).repeat_interleave(window // patch, dim=2)
        width, heads = content.shape[-1], self.fusion_heads
        flat = content.reshape(batch * channels * patches, bands, width)
        qkv = self.qkv(self.content_norm(flat)).reshape(-1, bands, 3, heads, width // heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        relation_bias = torch.einsum('mijr,hr->mhij', relations.reshape(-1, bands, bands, 2),
                                     self.relation_weight)
        diagonal = torch.eye(bands, device=x.device, dtype=torch.bool)
        relation_bias = relation_bias.masked_fill(diagonal, float('-inf'))
        fused = F.scaled_dot_product_attention(q, k, v, attn_mask=relation_bias)
        fused = fused.transpose(1, 2).reshape(-1, bands, width)
        tokens = flat + self.fusion_scale * self.output_projection(fused)
        tokens = tokens.reshape(batch, channels, patches, bands, width).transpose(2, 3).contiguous()
        result = (tokens, phase_amp.repeat_interleave(window // patch, dim=2), self.band_hz())
        if return_amp_target:
            target = envelope.reshape(batch, channels, bands, patches, patch).mean(dim=-1)
            result += (torch.log(target + 1e-6),)
        return result
