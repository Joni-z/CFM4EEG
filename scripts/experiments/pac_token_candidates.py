"""Small, frozen-checkpoint PAC residual candidates for synthetic evaluation.

No downstream training or claimed clinical equivalence. Every candidate uses
the complete original CfFT output and adds a bounded residual of the measured
coefficients. Fixed seeded projections are not fitted to synthetic PAC labels.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator


class PACResidualCandidates(nn.Module):
    """Generate residuals on the original [B,C,band,patch,feature] grid.

    The three alternatives preserve content tokens: coefficient residual,
    content-conditioned residual, and a bounded complex rotation correction.
    The physical analysis bank and measurement are the existing CFC operator.
    """
    def __init__(self, frontend, seed=1103):
        super().__init__()
        self.original = copy.deepcopy(frontend).eval().requires_grad_(False)
        self.operator = CFCOperator().eval().requires_grad_(False)
        d = frontend.hidden_dim if hasattr(frontend, 'hidden_dim') else frontend.amplitude_tokenizer.out_channels*2
        width = self.operator.n_source*self.operator.harmonics*2
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(d, width, generator=generator))
        self.register_buffer('coefficient_projection', q.T.contiguous())
        # A second independent fixed projection for content conditioning.
        q, _ = torch.linalg.qr(torch.randn(d, width, generator=generator))
        self.register_buffer('content_projection', q.T.contiguous())
        target = self.operator.physical_grid()['target_centers']
        fs = self.original.sinc.sample_rate
        t = torch.arange(self.original.sinc.kernel_size)/fs
        self.register_buffer('target_cos', torch.cos(2*torch.pi*t[:, None]*target[None]))
        self.register_buffer('target_sin', torch.sin(2*torch.pi*t[:, None]*target[None]))

    def mapping(self):
        kernel = self.original.sinc.filters().squeeze(1)
        power = (kernel@self.target_cos).square()+(kernel@self.target_sin).square()
        peak = torch.fft.rfft(kernel, n=2048).abs().square().amax(-1, keepdim=True)
        response = (power/peak.clamp_min(1e-12)).clamp(0, 1)
        response = torch.where(response >= .5, response, 0.)
        return response/response.sum(-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def components(self, x):
        base = self.original(x)[0]
        packet = self.operator(x).relation_features(stabilized=False)
        # Pack source and harmonic identity instead of collapsing to one angle.
        coef = packet['features'].flatten(-2)
        rel = torch.einsum('nt,bcpstk->bcpnsk', self.mapping(), coef).flatten(-2)
        rel = rel.permute(0, 1, 3, 2, 4)
        additive = rel@self.coefficient_projection
        conditioning = rel@self.content_projection
        # Fixed normalization prevents any added branch from spending a larger
        # token perturbation budget merely because it has more coordinates.
        additive = additive/(additive.square().mean((-1,-2,-3), keepdim=True).sqrt().clamp_min(1.))
        content = base*torch.tanh(conditioning)
        pair = base.reshape(*base.shape[:-1], -1, 2)
        phase = torch.tanh(conditioning[..., :pair.shape[-2]])
        tangent = torch.stack([-pair[..., 1], pair[..., 0]], -1)*phase[..., None]
        return base, additive, content, tangent.flatten(-2), packet

    @staticmethod
    def fuse(base, residual, gain, scale):
        """Bound actual RMS perturbation per record to gain * original RMS.

        ``scale`` is one fixed train-derived amplitude for the additive branch;
        candidate selection must not use test statistics.
        """
        delta = residual*scale
        axes = tuple(range(1, base.ndim))
        norm = base.square().mean(axes, keepdim=True).sqrt()
        size = delta.square().mean(axes, keepdim=True).sqrt()
        delta = delta*torch.minimum(torch.ones_like(size), norm/size.clamp_min(1e-12))
        return base+gain*delta
