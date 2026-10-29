"""New PAC measurement with the original CfFT content and full rotation."""
from __future__ import annotations

import torch
from torch import Tensor

from .analytic import hilbert, phase_amplitude
from .measurement_uncertain import CFCOperator, UncertainPACMeasurement
from .triaxial import TriAxialFrontend, _patch_project, patch_pac_vector


class CFCFrontend(TriAxialFrontend):
    def __init__(self, *args, **kwargs):
        kwargs.update(tokenizer_mode="pac_interaction", interaction_mode="rotation",
                      waveform_modulation="quadrature")
        super().__init__(*args, **kwargs)
        if self.pac_patch_lens != [self.patch_len]:
            raise ValueError("CFC reference uses the original single-patch local fallback")
        if self.pac_token_mode not in ("measured", "identity"):
            raise ValueError("CFC reference supports measured or identity")
        self.measurement = CFCOperator(
            sample_rate=self.sinc.sample_rate, patch_len=self.patch_len,
            passband=tuple(kwargs.get("filter_range") or (.5, 75.)),
            block_cycles=1., shrink=False)
        target = self.measurement.physical_grid()["target_centers"].float()
        time = torch.arange(self.sinc.kernel_size)/self.sinc.sample_rate
        angle = 2*torch.pi*time[:, None]*target[None, :]
        self.register_buffer("target_cos", angle.cos())
        self.register_buffer("target_sin", angle.sin())

    def target_mapping(self) -> Tensor:
        """Power response at physical targets, restricted to the -3dB passband."""
        kernel = self.sinc.filters().squeeze(1)
        power = (kernel@self.target_cos).square()+(kernel@self.target_sin).square()
        peak = torch.fft.rfft(kernel, n=2048).abs().square().amax(-1, keepdim=True)
        response = (power/peak.clamp_min(1e-12)).clamp(0, 1)
        return torch.where(response >= .5, response, 0.)

    def measured_reference(self, measured: UncertainPACMeasurement) -> tuple[Tensor, Tensor]:
        """H1 reference and availability, on the existing content-token grid.

        The joint H1/H2 fit is unchanged. Original CfFT rotation consumes one
        first-harmonic phasor; H2 is not converted into an ambiguous angle.
        """
        packet = measured.relation_features(stabilized=False)
        coef = torch.view_as_complex(packet["features"].contiguous())[..., 0]
        b, c, s, length = measured.source_phase.shape
        phase = measured.source_phase*measured.phase_valid
        def project(signal: Tensor) -> Tensor:
            return _patch_project(self.phase_tokenizer, signal.reshape(-1, length)).reshape(
                b, c, s, length//self.patch_len, -1).permute(0, 1, 3, 2, 4)
        feature = torch.complex(project(phase.real), project(phase.imag))
        mapping = self.target_mapping()
        context = torch.einsum("nt,bcpst,bcpsd->bcpnd",
                               mapping.to(coef.dtype), coef.conj(), feature)
        available = torch.einsum("nt,bcpst->bcpn", mapping,
                                 packet["harmonic_valid"][..., 0].float()) > 0
        available = available & (context.abs().amin(-1) >= 1e-6)
        return context/context.abs().clamp_min(1e-6), available

    def _forward_fp32(self, x: Tensor, return_amp_target: bool = False):
        b, c, length = x.shape
        patches = length//self.patch_len
        filtered = self.sinc(x.reshape(b*c, 1, length)).reshape(b, c, self.n_bands, length)
        phase, amplitude = phase_amplitude(hilbert(filtered))
        local = patch_pac_vector(phase, amplitude, patches, self.normalize)
        tokens = self._interaction_tokens(phase, amplitude, local, filtered=filtered)
        if self.pac_token_mode == "measured":
            reference, available = self.measured_reference(self.measurement(x))
            def project(layer, signal: Tensor) -> Tensor:
                return _patch_project(layer, signal.reshape(-1, length)).reshape(
                    b, c, self.n_bands, patches, -1).permute(0, 1, 3, 2, 4)
            a = project(self.amplitude_tokenizer, torch.log1p(amplitude))*self.amplitude_scale
            w = project(self.waveform_quadrature, filtered)
            carrier = torch.complex(a, w if self.waveform_content else w*0)
            measured = torch.view_as_real(carrier*reference).flatten(-2).permute(0, 1, 3, 2, 4)
            tokens = torch.where(available.permute(0, 1, 3, 2)[..., None], measured, tokens)
        result = [tokens.contiguous(), local.abs(), self.token_band_hz()]
        if return_amp_target:
            result.append(torch.log(amplitude[..., :patches*self.patch_len].reshape(
                b, c, self.n_bands, patches, self.patch_len).mean(-1)+1e-6))
        return tuple(result)
