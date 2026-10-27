"""Measured PAC tokens using the existing envelope/waveform projections.

Physical measurement anchors are fixed at initialization and checkpointed.
Learned content Sinc filters remain separate; do not call their envelopes the
same quantity as the broad measurement envelope.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .analytic import hilbert, phase_amplitude
from .measurement import PhaseAmplitudeMeasurement
from .triaxial import TriAxialFrontend, _patch_project


class MeasuredPACFrontend(TriAxialFrontend):
    def __init__(self, *args, measurement_mode: str = 'measured',
                 measurement_phase_cycles: float = 5., **kwargs):
        kwargs['tokenizer_mode'] = 'pac_interaction'
        kwargs['interaction_mode'] = 'rotation'
        kwargs['waveform_modulation'] = 'quadrature'
        super().__init__(*args, **kwargs)
        if measurement_mode not in ('measured', 'identity'):
            raise ValueError('measurement_mode must be measured or identity')
        self.measurement_mode = measurement_mode
        with torch.no_grad():
            low = self.sinc.min_low_hz+self.sinc.low_hz_.abs().flatten()
            high = (low+self.sinc.min_band_hz+self.sinc.band_hz_.abs().flatten()).clamp_max(self.sinc.sample_rate/2)
            centers = (low+high)/2
        passband = kwargs.get('filter_range') or (.5, min(75., self.sinc.sample_rate/2-1))
        self.measurement = PhaseAmplitudeMeasurement(
            centers, sample_rate=self.sinc.sample_rate, patch_len=self.patch_len,
            passband=tuple(passband), phase_cycles=measurement_phase_cycles)

    def _forward_fp32(self, x: Tensor, return_amp_target: bool = False):
        batch, channels, length = x.shape
        bands, patches = self.n_bands, length//self.patch_len
        filtered = self.sinc(x.reshape(batch*channels, 1, length)).reshape(batch, channels, bands, length)
        _, amplitude = phase_amplitude(hilbert(filtered))
        measured = self.measurement(x)
        flat = (batch*channels*bands, length)

        def project(layer, signal):
            return _patch_project(layer, signal.reshape(flat)).reshape(
                batch, channels, bands, patches, -1).permute(0, 1, 3, 2, 4)

        phase = torch.complex(project(self.phase_tokenizer, measured.phase_unit.real),
                              project(self.phase_tokenizer, measured.phase_unit.imag))
        a = project(self.amplitude_tokenizer, torch.log1p(amplitude))*self.amplitude_scale
        w = project(self.waveform_quadrature, filtered)
        if not self.waveform_content:
            w = w*0
        carrier = torch.complex(a, w)
        if self.measurement_mode == 'identity':
            output = carrier
        else:
            ratio = torch.where(measured.valid,
                measured.coef/measured.baseline.clamp_min(1e-12), torch.zeros_like(measured.coef))
            # Same measured phasor convention as the archived tokenizer, but
            # preserve absolute depth and keep a safe identity reference.
            transported = torch.einsum('bcpij,bcpik->bcpjk', ratio.conj(), phase)
            bound = torch.einsum('bcpij,bcpik->bcpjk', ratio.abs(), phase.abs())
            reference = 1+transported/(1+bound)
            rotation = reference/reference.abs().clamp_min(1e-6)
            output = carrier*rotation
        tokens = torch.view_as_real(output).flatten(-2).permute(0, 1, 3, 2, 4).contiguous()
        result = [tokens, measured.depth, self.token_band_hz()]
        if return_amp_target:
            result.append(torch.log(amplitude[..., :patches*self.patch_len].reshape(
                batch, channels, bands, patches, self.patch_len).mean(-1)+1e-6))
        if self.return_pac_vector:
            result.append(measured.coef)
        return tuple(result)
