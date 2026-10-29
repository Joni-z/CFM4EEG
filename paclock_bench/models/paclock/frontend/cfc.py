"""Physical-grid PAC measurement with explicit, versioned token integration.

The original rotation pilot uses physical response mapping. The residual pilot
uses separate harmonic readouts to condition all content bands, retaining the
unrotated content path. Neither interprets harmonic two as a unique phase angle.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .analytic import hilbert, phase_amplitude
from .measurement_uncertain import CFCOperator, UncertainPACMeasurement
from .triaxial import TriAxialFrontend, _patch_project


class CFCFrontend(TriAxialFrontend):
    def __init__(self, *args, cfc_guidance: str = "measured",
                 cfc_stabilized: bool = True, cfc_fusion: str = "rotation", **kwargs):
        kwargs.update(tokenizer_mode="pac_interaction", interaction_mode="rotation",
                      waveform_modulation="quadrature")
        super().__init__(*args, **kwargs)
        if cfc_guidance not in ("measured", "identity"):
            raise ValueError("cfc_guidance must be measured or identity")
        self.cfc_guidance = cfc_guidance
        self.cfc_stabilized = cfc_stabilized
        if cfc_fusion not in ("rotation", "residual"):
            raise ValueError("cfc_fusion must be rotation or residual")
        self.cfc_fusion = cfc_fusion
        passband = kwargs.get("filter_range") or (.5, 75.)
        self.measurement = CFCOperator(
            sample_rate=self.sinc.sample_rate, patch_len=self.patch_len,
            passband=tuple(passband), block_cycles=1., shrink=False)
        frequency = self.measurement.physical_grid()["target_centers"].float()
        time = torch.arange(self.sinc.kernel_size)/self.sinc.sample_rate
        angle = 2*torch.pi*time[:, None]*frequency[None, :]
        self.register_buffer("target_cos", angle.cos())
        self.register_buffer("target_sin", angle.sin())
        if cfc_fusion == "residual":
            # These weights learn how physical PAC context conditions content;
            # they do not pretend each content band is a measured PAC target.
            width = 2*self.phase_tokenizer.out_channels
            self.relation_mix_logits = nn.Parameter(torch.zeros(
                self.n_bands, self.measurement.n_target*self.measurement.harmonics))
            # Preserve encoder/head RNG initialization relative to the old pilot.
            with torch.random.fork_rng(devices=[]):
                self.relation_values = nn.ModuleList([
                    nn.Linear(width, width, bias=False)
                    for _ in range(self.measurement.harmonics)])
            self.residual_gain = nn.Parameter(torch.ones(self.n_bands))

    def target_mapping(self) -> Tensor:
        """[content band, measurement target], nonnegative, row mass <= 1.

        Absolute response relative to each filter's peak is retained: an
        out-of-band filter is not assigned a strong relation by normalization.
        """
        kernel = self.sinc.filters().squeeze(1)
        power = (kernel@self.target_cos).square()+(kernel@self.target_sin).square()
        peak = torch.fft.rfft(kernel, n=2048).abs().square().amax(-1, keepdim=True)
        weights = (power/peak.clamp_min(1e-12)).clamp(0, 1)
        return weights/weights.sum(-1, keepdim=True).clamp_min(1.)

    def relation_context(self, measured: UncertainPACMeasurement) -> tuple[Tensor, Tensor]:
        """Project each measured harmonic with its matching source phase power."""
        packet = measured.relation_features(stabilized=self.cfc_stabilized)
        coefficient = torch.view_as_complex(packet["features"].contiguous())
        batch, channels, sources, length = measured.source_phase.shape
        patches = length//self.patch_len
        contexts, bounds = [], []
        for harmonic in range(self.measurement.harmonics):
            phase = measured.source_phase**(harmonic+1)*measured.phase_valid
            def project(signal: Tensor) -> Tensor:
                return _patch_project(self.phase_tokenizer, signal.reshape(-1, length)).reshape(
                    batch, channels, sources, patches, -1).permute(0, 1, 3, 2, 4)
            feature = torch.complex(project(phase.real), project(phase.imag))
            ratio = coefficient[..., harmonic]
            contexts.append(torch.einsum("bcpst,bcpsd->bcptd", ratio.conj(), feature))
            bounds.append(torch.einsum("bcpst,bcpsd->bcptd", ratio.abs(), feature.abs()))
        mapping = self.target_mapping()
        context = torch.einsum("nt,bcptd->bcpnd", mapping.to(contexts[0].dtype), sum(contexts))
        bound = torch.einsum("nt,bcptd->bcpnd", mapping, sum(bounds))
        return context, bound

    def relation_residual(self, measured: UncertainPACMeasurement) -> Tensor:
        """Gauge-invariant harmonic responses -> dense content conditioning.

        Returns [batch, electrode, patch, content band, feature]. Harmonics
        have independent value projections; coefficients keep their magnitude.
        Zero or unsupported coefficients yield exactly zero, including biases.
        Global coefficients are evaluated against local phase features; this
        does not turn them into independent local PAC estimates.
        """
        packet = measured.relation_features(stabilized=self.cfc_stabilized)
        coefficient = torch.view_as_complex(packet["features"].contiguous())
        batch, channels, sources, length = measured.source_phase.shape
        patches = length//self.patch_len
        weights = self.relation_mix_logits.softmax(-1).reshape(
            self.n_bands, self.measurement.n_target, self.measurement.harmonics)
        residual = None
        for harmonic, projection in enumerate(self.relation_values):
            phase = measured.source_phase**(harmonic+1)*measured.phase_valid
            def project(signal: Tensor) -> Tensor:
                return _patch_project(self.phase_tokenizer, signal.reshape(-1, length)).reshape(
                    batch, channels, sources, patches, -1).permute(0, 1, 3, 2, 4)
            feature = torch.complex(project(phase.real), project(phase.imag))
            response = torch.einsum("bcpst,bcpsd->bcptd",
                                    coefficient[..., harmonic].conj(), feature)
            value = projection(torch.view_as_real(response).flatten(-2))
            contribution = torch.einsum("nt,bcptd->bcpnd", weights[..., harmonic], value)
            residual = contribution if residual is None else residual+contribution
        return residual*self.residual_gain[None, None, None, :, None]

    def _forward_fp32(self, x: Tensor, return_amp_target: bool = False):
        batch, channels, length = x.shape
        patches = length//self.patch_len
        filtered = self.sinc(x.reshape(batch*channels, 1, length)).reshape(
            batch, channels, self.n_bands, length)
        _, amplitude = phase_amplitude(hilbert(filtered))
        measured = self.measurement(x)
        def project(layer, signal: Tensor) -> Tensor:
            return _patch_project(layer, signal.reshape(-1, length)).reshape(
                batch, channels, self.n_bands, patches, -1).permute(0, 1, 3, 2, 4)
        a = project(self.amplitude_tokenizer, torch.log1p(amplitude))*self.amplitude_scale
        w = project(self.waveform_quadrature, filtered)
        carrier = torch.complex(a, w if self.waveform_content else w*0)
        if self.cfc_guidance == "identity":
            output = carrier
        elif self.cfc_fusion == "residual":
            output = carrier+torch.view_as_complex(
                self.relation_residual(measured).reshape(*carrier.shape, 2).contiguous())
        else:
            context, bound = self.relation_context(measured)
            # Zero/unavailable coupling gives the identity exactly. Weak
            # estimates remain weak instead of being promoted to unit magnitude.
            reference = 1+context/(1+bound)
            rotation = reference/reference.abs().clamp_min(1e-6)
            output = carrier*rotation
        tokens = torch.view_as_real(output).flatten(-2).permute(0, 1, 3, 2, 4).contiguous()
        strength = torch.linalg.vector_norm(
            measured.relation_features(stabilized=self.cfc_stabilized)["features"], dim=(-2, -1))
        result = [tokens, strength, self.token_band_hz()]
        if return_amp_target:
            result.append(torch.log(amplitude[..., :patches*self.patch_len].reshape(
                batch, channels, self.n_bands, patches, self.patch_len).mean(-1)+1e-6))
        return tuple(result)
