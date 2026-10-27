"""Explicit first-harmonic PAC measurement with physical support masks.

Regression follows the sine/cosine approach of Penny et al. (2008) and
van Wijk et al. (2015). Separate phase/envelope bandwidths follow Dvorak and
Fenton (2014). This is an estimator, not a neural-causality or significance test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def harmonic_regression(phase: Tensor, envelope: Tensor, weights: Tensor | None = None,
                        ridge: float = 1e-3, detrend: bool = True) -> dict[str, Tensor]:
    """Fit A=b0+bt*time+bc*cos(phi)+bs*sin(phi) along the last axis.

    Leading dimensions broadcast. Angles retain the original equal scaling of
    sine/cosine. Constant/time nuisance is removed analytically; only a 2x2
    solve is needed. The returned depth is unbounded relative harmonic depth.
    """
    phase, envelope = torch.broadcast_tensors(phase, envelope)
    dtype = phase.real.dtype
    if weights is None:
        weights = torch.ones_like(envelope)
    else:
        weights = torch.broadcast_to(weights, envelope.shape).to(dtype)
    count = weights.sum(-1)
    w = weights / count.clamp_min(1)[..., None]
    t = torch.linspace(-1, 1, envelope.shape[-1], device=envelope.device, dtype=dtype)

    def mean(x: Tensor) -> Tensor:
        return (w * x).sum(-1)

    c, s = phase.real, phase.imag
    mc, ms, my, mt = mean(c), mean(s), mean(envelope), mean(t)
    c, s, y, t = c-mc[..., None], s-ms[..., None], envelope-my[..., None], t-mt[..., None]
    vt = mean(t.square()).clamp_min(1e-12)
    ct, st, yt = mean(c*t), mean(s*t), mean(y*t)
    if detrend:
        c = c - (ct/vt)[..., None]*t
        s = s - (st/vt)[..., None]*t
        y = y - (yt/vt)[..., None]*t
    cc, ss, cs = mean(c*c), mean(s*s), mean(c*s)
    cy, sy = mean(c*y), mean(s*y)
    gram = torch.stack([cc, cs, cs, ss], -1).reshape(*cc.shape, 2, 2)
    # Eigenvalues of a symmetric 2x2 matrix without an expensive eigensolver.
    gap = torch.sqrt((cc-ss).square()+4*cs.square()+torch.finfo(dtype).tiny)
    eigmin, eigmax = (cc+ss-gap)/2, (cc+ss+gap)/2
    regularizer = ridge*(cc+ss).clamp_min(1e-6)/2 + 1e-10
    system = gram + regularizer[..., None, None]*torch.eye(2, device=c.device, dtype=dtype)
    beta = torch.linalg.solve(system, torch.stack([cy, sy], -1)[..., None]).squeeze(-1)
    bc, bs = beta.unbind(-1)
    baseline = my-bc*mc-bs*ms
    drift = (yt-bc*ct-bs*st)/vt if detrend else torch.zeros_like(my)
    coef = torch.complex(bc, bs)
    residual = y-bc[..., None]*c-bs[..., None]*s
    mse = mean(residual.square())
    valid = ((count >= 8) & (eigmin >= 0.01) & (eigmin >= 0.05*eigmax)
             & (baseline > 1e-12) & torch.isfinite(coef) & torch.isfinite(mse))
    depth = coef.abs()/baseline.clamp_min(1e-12)
    return dict(coef=coef, depth=depth, baseline=baseline, drift=drift, mse=mse,
                valid=valid, eigmin=eigmin, eigmax=eigmax, count=count)


def _gain(kernel: Tensor, frequency: Tensor, fs: float) -> Tensor:
    t = (torch.arange(kernel.numel(), dtype=torch.float64)-(kernel.numel()-1)/2)/fs
    return (kernel.to(torch.complex128)[None, :]
            * torch.exp(-2j*math.pi*frequency[:, None]*t)).sum(-1).abs()


def _phase_filter(f: float, fs: float, cycles: float) -> Tensor:
    sigma = cycles/(2*math.pi*f)
    radius = math.ceil(3*sigma*fs)
    t = torch.arange(-radius, radius+1, dtype=torch.float64)/fs
    h = torch.exp(-t.square()/(2*sigma*sigma))*(torch.exp(2j*math.pi*f*t)-math.exp(-cycles*cycles/2))
    h = 2*h/_gain(h, torch.tensor([f]), fs)[0]
    return h.to(torch.complex64)


def _amplitude_filter(f: float, half_width: float, fs: float) -> Tensor:
    transition = half_width/4
    radius = math.ceil(2*fs/transition)
    t = torch.arange(-radius, radius+1, dtype=torch.float64)/fs
    cutoff = half_width+transition/2
    lowpass = 2*cutoff/fs*torch.sinc(2*cutoff*t)*torch.hamming_window(2*radius+1, periodic=False, dtype=torch.float64)
    lowpass = lowpass/lowpass.sum()
    return (2*lowpass*torch.exp(2j*math.pi*f*t)).to(torch.complex64)


def _stack_kernels(kernels: list[Tensor]) -> Tensor:
    length = max(k.numel() for k in kernels)
    return torch.stack([F.pad(k, ((length-k.numel())//2, (length-k.numel())//2)) for k in kernels])


def linear_analytic_filter(x: Tensor, kernels: Tensor) -> Tensor:
    """Real [...,T] -> complex [...,band,T], with linear convolution."""
    length, taps = x.shape[-1], kernels.shape[-1]
    nfft = 1 << (length+taps-2).bit_length()
    xf = torch.fft.fft(x.float(), n=nfft)
    hf = torch.fft.fft(kernels, n=nfft)
    full = torch.fft.ifft(xf[..., None, :]*hf, n=nfft)
    guard = taps//2
    return full[..., guard:guard+length]


@dataclass
class PACMeasurement:
    coef: Tensor
    depth: Tensor
    valid: Tensor
    phase_unit: Tensor
    phase_valid: Tensor
    baseline: Tensor
    mse: Tensor
    context_count: Tensor
    observed_cycles: Tensor
    pair_eligible: Tensor


class PhaseAmplitudeMeasurement(nn.Module):
    """Fixed physical analysis anchors; no fitted neural estimator parameters.

    Output edge axes are [B,C,patch,source,target]. Kernel support outside the
    supplied example is never counted as observed data. Analysis anchor metadata
    is checkpointed; it is distinct from later learned content-filter drift.
    """
    def __init__(self, centers: list[float] | Tensor, sample_rate: int = 200,
                 patch_len: int = 50, passband: tuple[float, float] = (0.5, 75),
                 phase_cycles: float = 5, min_cycles: float = 3,
                 min_seconds: float = 1, ridge: float = 1e-3,
                 detrend: bool = True):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.float64)
        if centers.ndim != 1 or centers.numel() < 2 or not torch.isfinite(centers).all():
            raise ValueError('centers must be a finite vector of at least two frequencies')
        lo, hi = map(float, passband)
        if not (0 <= lo < hi < sample_rate/2) or (centers <= lo).any() or (centers >= hi).any():
            raise ValueError('analysis centers and passband must be inside Nyquist')
        if phase_cycles <= 2 or min_cycles <= 0 or min_seconds <= 0 or patch_len <= 0:
            raise ValueError('invalid cycle/context/patch setting')
        self.sample_rate, self.patch_len = sample_rate, patch_len
        self.min_cycles, self.min_seconds, self.ridge = min_cycles, min_seconds, ridge
        self.detrend = detrend
        lower = centers*(1-2/phase_cycles)
        upper = centers*(1+2/phase_cycles)
        width = torch.minimum(0.4*centers, torch.minimum(0.8*(centers-lo), 0.8*(hi-centers)))
        eligible = ((upper[:, None] <= width[None, :])
                    & (upper[:, None] < (centers-1.25*width)[None, :])
                    & (lower[:, None] >= lo))
        phases = [_phase_filter(float(f), sample_rate, phase_cycles) for f in centers]
        amplitudes = [_amplitude_filter(float(f), float(w), sample_rate)
                      if eligible[:, j].any() else torch.zeros(1, dtype=torch.complex64)
                      for j, (f, w) in enumerate(zip(centers, width))]
        gain_ratio = torch.zeros_like(eligible, dtype=torch.float64)
        for i, f_source in enumerate(upper):
            for j, f_target in enumerate(centers):
                if eligible[i, j]:
                    response = _gain(amplitudes[j], torch.stack([f_target-f_source, f_target, f_target+f_source]), sample_rate)
                    gain_ratio[i, j] = torch.minimum(response[0], response[2])/response[1]
                    leakage = _gain(amplitudes[j], torch.stack([centers[i], -f_target]), sample_rate)/response[1]
                    eligible[i, j] &= bool(gain_ratio[i, j] >= .9 and leakage.max() <= .01)
        self.register_buffer('centers', centers.float())
        self.register_buffer('phase_lower', lower.float())
        self.register_buffer('phase_kernels', _stack_kernels(phases))
        self.register_buffer('amplitude_kernels', _stack_kernels(amplitudes))
        self.register_buffer('phase_guard', torch.tensor([p.numel()//2 for p in phases]))
        self.register_buffer('amplitude_guard', torch.tensor([a.numel()//2 for a in amplitudes]))
        self.register_buffer('eligible', eligible)
        self.register_buffer('sideband_gain', gain_ratio.float())

    def forward(self, x: Tensor) -> PACMeasurement:
        if x.ndim != 3:
            raise ValueError('measurement input must have shape [batch,channel,time]')
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x.float())

    def _forward(self, x: Tensor) -> PACMeasurement:
        b, c, length = x.shape
        bands, patches = self.centers.numel(), length//self.patch_len
        if patches < 1:
            raise ValueError('input shorter than one content patch')
        finite = torch.isfinite(x).all(-1)
        clean = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        source = linear_analytic_filter(clean, self.phase_kernels)
        envelope = linear_analytic_filter(clean, self.amplitude_kernels).abs()
        amp = source.abs()
        phase = source/amp.clamp_min(1e-12)
        times = torch.arange(length, device=x.device)
        phase_valid = ((times[None, :] >= self.phase_guard[:, None])
                       & (times[None, :] < length-self.phase_guard[:, None]))
        phase_valid = (phase_valid[None, None] & finite[:, :, None, None]
                       & (amp > (amp.square().mean(-1, keepdim=True).sqrt()*1e-4).clamp_min(1e-12)))
        anchors = (torch.arange(patches, device=x.device)+.5)*self.patch_len
        columns: dict[str, list[Tensor]] = {k: [] for k in ('coef','depth','valid','baseline','mse','count','cycles')}
        for i in range(bands):
            empty_shape = (b, c, patches, bands)
            if not self.eligible[i].any():
                zero = clean.new_zeros(empty_shape)
                for key in columns:
                    columns[key].append(zero.bool() if key == 'valid' else zero.to(torch.complex64) if key == 'coef' else zero)
                continue
            seconds = max(self.min_seconds, self.min_cycles/float(self.phase_lower[i]))
            win = math.ceil(seconds*self.sample_rate)+1
            # Oversized contexts can be marked unavailable without allocating
            # a huge low-frequency gather tensor.
            if win > 2*length+1:
                zero = clean.new_zeros(empty_shape)
                for key in columns:
                    columns[key].append(zero.bool() if key == 'valid' else zero.to(torch.complex64) if key == 'coef' else zero)
                continue
            index = anchors.long()[:, None] + torch.arange(win, device=x.device)[None, :] - win//2
            bounded = index.clamp(0, length-1)
            p = phase[:, :, i, bounded][:, :, :, None, :]
            a = envelope[:, :, :, bounded].permute(0, 1, 3, 2, 4)
            good = phase_valid[:, :, i, bounded][:, :, :, None, :]
            good = (good & (index[None, None, :, None, :] >= self.amplitude_guard[None, None, None, :, None])
                    & (index[None, None, :, None, :] < length-self.amplitude_guard[None, None, None, :, None])
                    & (index[None, None, :, None, :] >= 0) & (index[None, None, :, None, :] < length))
            fit = harmonic_regression(p, a, good, self.ridge, self.detrend)
            steps = p[..., 1:]*p[..., :-1].conj()
            contiguous = good[..., 1:] & good[..., :-1]
            cycles = (torch.atan2(steps.imag, steps.real)*contiguous).sum(-1)/(2*math.pi)
            enough = ((fit['count']-1)*self.phase_lower[i]/self.sample_rate >= self.min_cycles-.02)
            valid = fit['valid'] & enough & (cycles >= self.min_cycles-.02) & self.eligible[i][None, None, None, :]
            for key in ('coef','depth','baseline','mse'):
                columns[key].append(torch.where(valid, fit[key], torch.zeros_like(fit[key])))
            columns['valid'].append(valid)
            columns['count'].append(fit['count'])
            columns['cycles'].append(cycles)
        out = {k: torch.stack(v, dim=-2) for k, v in columns.items()}
        return PACMeasurement(out['coef'], out['depth'], out['valid'], phase, phase_valid,
                              out['baseline'], out['mse'], out['count'], out['cycles'], self.eligible)
