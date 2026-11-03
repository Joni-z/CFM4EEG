"""Single measured-PAC path in the original content grid; research prototypes.

The original frontend supplies its learned filters and content/phase projections,
never its local PAC statistic. No coefficient-only additive output is created.
"""
import copy

import torch
from torch import nn

from paclock_bench.models.paclock.frontend.analytic import hilbert, phase_amplitude
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator
from paclock_bench.models.paclock.frontend.triaxial import _patch_project


class SinglePathPAC(nn.Module):
    MODES = ('aligned', 'strength', 'direct')

    def __init__(self, frontend):
        super().__init__()
        if frontend.sinc.sample_rate != 200 or frontend.patch_len != 50:
            raise ValueError('prototype requires 200 Hz and 50-sample patches')
        if frontend.waveform_modulation != 'quadrature' or frontend.interaction_mode != 'rotation':
            raise ValueError('prototype requires the quadrature rotation frontend')
        self.original = copy.deepcopy(frontend)
        self.operator = CFCOperator()
        target = self.operator.physical_grid()['target_centers']
        time = torch.arange(frontend.sinc.kernel_size) / frontend.sinc.sample_rate
        self.register_buffer('target_cos', torch.cos(2 * torch.pi * time[:, None] * target[None]))
        self.register_buffer('target_sin', torch.sin(2 * torch.pi * time[:, None] * target[None]))

    def mapping(self):
        filters = self.original.sinc.filters().squeeze(1)
        power = (filters @ self.target_cos).square() + (filters @ self.target_sin).square()
        peak = torch.fft.rfft(filters, n=2048).abs().square().amax(-1, keepdim=True)
        response = (power / peak.clamp_min(1e-12)).clamp(0, 1)
        response = torch.where(response >= .5, response, 0.)
        return response / response.sum(-1, keepdim=True).clamp_min(1e-12)

    def prepare(self, x):
        f = self.original
        b, c, length = x.shape
        filtered = f.sinc(x.reshape(b * c, 1, length)).reshape(b, c, f.n_bands, length)
        _, envelope = phase_amplitude(hilbert(filtered))
        a = _patch_project(f.amplitude_tokenizer, torch.log1p(envelope).reshape(-1, length)) * f.amplitude_scale
        w = _patch_project(f.waveform_quadrature, filtered.reshape(-1, length))
        if not f.waveform_content:
            w = w * 0
        content = torch.complex(a, w).reshape(b, c, f.n_bands, length // f.patch_len, -1)
        if content.shape[-1] % 2:
            raise ValueError('complex width must divide into H1 and H2 groups')
        measured = self.operator(x)
        coefficient = torch.view_as_complex(measured.relation_features(stabilized=False)['features'].contiguous())
        phase = measured.source_phase * measured.phase_valid
        source_features = []
        for harmonic in (1, 2):
            def project(signal):
                z = _patch_project(f.phase_tokenizer, signal.reshape(-1, length))
                return z.reshape(b, c, phase.shape[2], length // f.patch_len, -1).permute(0, 1, 3, 2, 4)
            ph = phase ** harmonic
            feature = torch.complex(project(ph.real), project(ph.imag))
            # Matching harmonic occupies its own existing feature coordinates.
            source_features.append(feature.chunk(2, dim=-1)[harmonic - 1])
        return dict(content=content, coefficient=coefficient, source_features=source_features,
                    mapping=self.mapping())

    def transform(self, prepared, mode, coefficient=None):
        if mode not in self.MODES:
            raise ValueError(mode)
        coefficient = prepared['coefficient'] if coefficient is None else coefficient
        # Average sources using their magnitudes; absent/invalid entries are zero.
        magnitude = coefficient.abs()
        weights = magnitude / magnitude.sum(dim=3, keepdim=True).clamp_min(1e-12)
        mapping = prepared['mapping'].to(coefficient.dtype)
        parts = []
        for h, content in enumerate(prepared['content'].chunk(2, dim=-1)):
            beta = coefficient[..., h]
            weight = weights[..., h]
            # b,c,p,n: one complex relation and a separate nonnegative depth.
            aggregate = torch.einsum('nj,bcpsj->bcpn', mapping, weight * beta)
            depth = torch.einsum('nj,bcpsj->bcpn', mapping.real, weight * beta.abs()).clamp(0, 1)
            if mode == 'direct':
                bounded = aggregate / aggregate.abs().clamp_min(1.)
                factor = (1 + .5 * bounded).permute(0, 1, 3, 2).unsqueeze(-1)
            else:
                context = torch.einsum('nj,bcpsj,bcpsd->bcpnd', mapping,
                                       weight * beta.conj(), prepared['source_features'][h])
                size = context.abs()
                reference = context / size.clamp_min(1e-6)
                reference = torch.where(size >= 1e-6, reference, torch.ones_like(reference))
                factor = reference.permute(0, 1, 3, 2, 4)
                if mode == 'strength':
                    factor = factor * (1 + .3 * depth.permute(0, 1, 3, 2).unsqueeze(-1))
            parts.append(content * factor)
        return torch.view_as_real(torch.cat(parts, dim=-1)).flatten(-2)

    def forward(self, x, *, mode='aligned'):
        return self.transform(self.prepare(x), mode)


class DirectPACFrontend(SinglePathPAC):
    """Candidate C adapter for the existing two-axis encoder, from scratch."""
    def __init__(self, frontend):
        super().__init__(frontend)
        self.tokenizer_mode = 'cfc_direct'
        self.n_token_bands = frontend.n_token_bands
        self.n_bands = frontend.n_bands
        self.patch_len = frontend.patch_len

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            tokens = super().forward(x.float(), mode='direct').contiguous()
        b, c, _, p, _ = tokens.shape
        # Unused by the none-frequency-mixer encoder; no second PAC measurement.
        coupling = tokens.new_zeros(b, c, p, self.n_bands, self.n_bands)
        return tokens, coupling, self.original.token_band_hz()
