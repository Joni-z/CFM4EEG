"""Fixed-frequency PAC curves: Fourier regression with context shrinkage.

Extends the sine/cosine regression family (Penny 2008, van Wijk 2015),
with modeled source-amplitude nuisance covariates (Nadalin 2019).
This estimates statistical phase/amplitude dependence, not neural causality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .measurement import PhaseAmplitudeMeasurement, _gain, linear_analytic_filter


def curve_regression(phase: Tensor, envelope: Tensor, weights: Tensor,
                     source_envelope: Tensor | None = None, harmonics: int = 2,
                     ridge: float = 1e-3, prior: Tensor | None = None,
                     amplitude_floor: float = .02) -> dict[str, Tensor]:
    """Weighted detrended harmonic ridge fit; only phase coefficients penalized.

    Inputs broadcast along the last (sample) axis. All nuisance columns are
    centered. Source-amplitude covariates are disabled below a relative
    variation floor, so tiny filter ripple is not amplified into a predictor.
    Coefficient layout is [...,harmonic] complex (cosine + i*sine).
    """
    if harmonics not in (1, 2) or ridge < 0:
        raise ValueError('harmonics must be 1 or 2 and ridge nonnegative')
    phase, envelope, weights = torch.broadcast_tensors(phase, envelope, weights)
    count = weights.sum(-1)
    w = weights / count.clamp_min(1)[..., None]
    def mean(v: Tensor) -> Tensor:
        return (v*w).sum(-1)
    t = torch.linspace(-1, 1, envelope.shape[-1], device=envelope.device)
    t = t-mean(t)[..., None]
    columns = [t]
    if source_envelope is not None:
        a = source_envelope.broadcast_to(envelope.shape)
        cv = mean((a-mean(a)[..., None]).square()).sqrt()/mean(a).clamp_min(1e-12)
        a = a.clamp_min(1e-12).log()
        a = a-mean(a)[..., None]
        sd = mean(a.square()).sqrt()
        a = torch.where((cv >= amplitude_floor)[..., None],
                        a/sd.clamp_min(1e-6)[..., None], torch.zeros_like(a))
        columns.extend([a, a.square()-mean(a.square())[..., None]])
    nuisance = torch.stack(columns, -1)
    powers = [phase**k for k in range(1, harmonics+1)]
    design = torch.stack([v for p in powers for v in (p.real, p.imag)], -1)
    mx = (design*w[..., None]).sum(-2)
    my = mean(envelope)
    xc = design-mx[..., None, :]
    yc = envelope-my[..., None]
    # Pseudoinverse handles absent/constant nuisance columns without imposing
    # arbitrary QR directions. Double precision protects small regressions.
    nd, wd = nuisance.double(), w.double()
    ng = nd.transpose(-1, -2) @ (nd*wd[..., None])
    inverse = torch.linalg.pinv(ng, rtol=1e-6, hermitian=True)
    nx = inverse @ (nd.transpose(-1, -2) @ (xc.double()*wd[..., None]))
    ny = inverse @ (nd.transpose(-1, -2) @ (yc.double()*wd)[..., None])
    xr = xc.double()-nd@nx
    yr = yc.double()-(nd@ny).squeeze(-1)
    gram = xr.transpose(-1, -2) @ (xr*wd[..., None])
    rhs = xr.transpose(-1, -2) @ (yr*wd)[..., None]
    dim = 2*harmonics
    penalty = ridge*gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)+1e-10
    if prior is not None:
        prior_vector = torch.view_as_real(prior).flatten(-2).double()
        rhs = rhs+penalty[..., None, None]*prior_vector[..., None]
    beta = torch.linalg.solve(gram+penalty[..., None, None]*torch.eye(dim, device=phase.device), rhs).squeeze(-1)
    residual = yr-(xr@beta[..., None]).squeeze(-1)
    baseline = my.double()-(mx.double()*beta).sum(-1)
    eigen = torch.linalg.eigvalsh(gram)
    finite = torch.isfinite(beta).all(-1)&torch.isfinite(baseline)
    valid = ((count >= max(8, 4*dim)) & (eigen[..., 0] >= .01)
             & (eigen[..., 0] >= .05*eigen[..., -1]) & (baseline > 1e-12) & finite)
    coef = torch.view_as_complex(beta.float().reshape(*beta.shape[:-1], harmonics, 2).contiguous())
    return dict(coef=coef, baseline=baseline.float(), valid=valid,
                mse=(residual.square()*wd).sum(-1).float(), count=count,
                condition=(eigen[..., -1]/eigen[..., 0].clamp_min(1e-12)).float())


@dataclass
class PACCurveMeasurement:
    coef: Tensor               # [B,C,P,source,target,harmonic]
    baseline: Tensor           # [B,C,P,source,target]
    valid: Tensor
    local_valid: Tensor
    global_valid: Tensor       # [B,C,source,target]
    source_phase: Tensor       # [B,C,source,T]
    harmonic_support: Tensor   # [source,target,harmonic]
    observed_cycles: Tensor    # [B,C,source,target]
    condition: Tensor          # [B,C,P,source,target]

    def modulation(self, source: int, target: int, patch: int) -> Tensor:
        """Evaluate one pair's selected coefficient curve over the input."""
        phase = self.source_phase[..., source, :]
        coef = self.coef[..., patch, source, target, :]
        value = torch.zeros_like(phase.real)
        for k in range(coef.shape[-1]):
            value = value+(coef[..., k, None].conj()*phase**(k+1)).real
        return value/self.baseline[..., patch, source, target, None].clamp_min(1e-12)


class PACCurveOperator(nn.Module):
    """PAC measured on fixed physical coordinates, independent of content bank."""
    def __init__(self, source_hz=(2.,3.,4.,5.,6.,8.), target_hz=(18.,24.,32.,40.,48.),
                 sample_rate: int = 200, patch_len: int = 50,
                 passband=(.5,75.), excluded_intervals=((56.,64.),),
                 harmonics: int = 2, control_amplitude: bool = True,
                 local: bool = True, ridge: float = 1e-3,
                 local_ridge: float = .05, amplitude_floor: float = .02,
                 phase_cycles: float = 5.):
        super().__init__()
        if harmonics not in (1,2):
            raise ValueError('harmonics must be 1 or 2')
        if len(set(source_hz)) != len(source_hz) or len(set(target_hz)) != len(target_hz):
            raise ValueError('duplicate analysis centers')
        self.n_source, self.n_target = len(source_hz), len(target_hz)
        self.harmonics, self.control_amplitude = harmonics, control_amplitude
        self.local, self.ridge, self.local_ridge = local, ridge, local_ridge
        self.amplitude_floor = amplitude_floor
        centers = tuple(source_hz)+tuple(target_hz)
        self.analysis = PhaseAmplitudeMeasurement(centers, sample_rate, patch_len,
            passband=passband, excluded_intervals=excluded_intervals,phase_cycles=phase_cycles)
        support = self.analysis.eligible[:self.n_source, self.n_source:, None].expand(-1,-1,harmonics).clone()
        for i in range(self.n_source):
            for j in range(self.n_target):
                kernel = self.analysis.amplitude_kernels[self.n_source+j].to(torch.complex128)
                f = float(self.analysis.centers[self.n_source+j])
                for k in range(1,harmonics+1):
                    if not support[i,j,k-1]:
                        continue
                    edge = k*float(self.analysis.phase_upper[i])
                    response = _gain(kernel, torch.tensor([f-edge,f,f+edge]), sample_rate)
                    ok = bool(response[1]>0 and torch.minimum(response[0],response[2])/response[1]>=.9)
                    for low,high in excluded_intervals:
                        ok = ok and not (f-edge <= high and f+edge >= low)
                    support[i,j,k-1] = ok
        self.register_buffer('harmonic_support',support)

    def forward(self, x: Tensor) -> PACCurveMeasurement:
        if x.ndim != 3 or x.shape[-1] < self.analysis.patch_len:
            raise ValueError('expected [batch,channel,time] with at least one patch')
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x.float())

    def _forward(self, x: Tensor) -> PACCurveMeasurement:
        b,c,length=x.shape; ns,nt=self.n_source,self.n_target
        patches=length//self.analysis.patch_len
        finite=torch.isfinite(x).all(-1)
        clean=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        source=linear_analytic_filter(clean,self.analysis.phase_kernels[:ns])
        envelope=linear_analytic_filter(clean,self.analysis.amplitude_kernels[ns:]).abs()
        amplitude=source.abs(); phase=source/amplitude.clamp_min(1e-12)
        out=clean.new_zeros(b,c,patches,ns,nt,self.harmonics,dtype=torch.complex64)
        baseline=clean.new_zeros(b,c,patches,ns,nt)
        valid=torch.zeros_like(baseline,dtype=torch.bool); local_valid=valid.clone()
        global_valid=valid[:,:,0].clone(); cycles_out=baseline[:,:,0].clone()
        condition=baseline.clone(); times=torch.arange(length,device=x.device)
        rms=clean.square().mean(-1).sqrt().clamp_min(1e-8)
        for i in range(ns):
            for j in range(nt):
                order=int(self.harmonic_support[i,j].sum())
                if order==0:
                    continue
                guard=max(int(self.analysis.phase_guard[i]),int(self.analysis.amplitude_guard[ns+j]))
                mask=((times>=guard)&(times<length-guard))[None,None,:]&finite[...,None]
                mask=mask&(amplitude[:,:,i]>rms[...,None]*1e-4)
                if not bool(mask.any()):
                    continue
                ph=phase[:,:,i]; amp=envelope[:,:,j]
                source_amp=amplitude[:,:,i] if self.control_amplitude else None
                fit=curve_regression(ph,amp,mask.float(),source_amp,order,self.ridge,
                                     amplitude_floor=self.amplitude_floor)
                steps=ph[...,1:]*ph[...,:-1].conj()
                cycles=(torch.angle(steps)*(mask[...,1:]&mask[...,:-1])).sum(-1)/(2*math.pi)
                freq=cycles*self.analysis.sample_rate/(fit['count']-1).clamp_min(1)
                enough=(fit['count']-1)*self.analysis.phase_lower[i]/self.analysis.sample_rate>=2.98
                good=fit['valid']&enough&(cycles>=2.98)&(freq>=self.analysis.phase_lower[i])&(freq<=self.analysis.phase_upper[i])
                global_valid[:,:,i,j]=good;cycles_out[:,:,i,j]=cycles
                global_coef=torch.where(good[...,None],fit['coef'],torch.zeros_like(fit['coef']))
                out[:,:,:,i,j,:order]=global_coef[:,:,None,:]
                baseline[:,:,:,i,j]=torch.where(good,fit['baseline'],0)[:,:,None]
                valid[:,:,:,i,j]=good[:,:,None]
                condition[:,:,:,i,j]=fit['condition'][:,:,None]
                if not self.local:
                    continue
                win=math.ceil(max(1.,3/float(self.analysis.phase_lower[i]))*self.analysis.sample_rate)+1
                for p in range(patches):
                    center=int((p+.5)*self.analysis.patch_len)
                    local_mask=mask&(times>=center-win//2)&(times<center-win//2+win)
                    local_fit=curve_regression(ph,amp,local_mask.float(),source_amp,order,
                        self.local_ridge,global_coef, self.amplitude_floor)
                    local_cycles=(torch.angle(steps)*(local_mask[...,1:]&local_mask[...,:-1])).sum(-1)/(2*math.pi)
                    local_enough=(local_fit['count']-1)*self.analysis.phase_lower[i]/self.analysis.sample_rate>=2.98
                    local_good=local_fit['valid']&local_enough&(local_cycles>=2.98)&good
                    local_valid[:,:,p,i,j]=local_good
                    out[:,:,p,i,j,:order]=torch.where(local_good[...,None],local_fit['coef'],global_coef)
                    baseline[:,:,p,i,j]=torch.where(local_good,local_fit['baseline'],baseline[:,:,p,i,j])
                    condition[:,:,p,i,j]=torch.where(local_good,local_fit['condition'],condition[:,:,p,i,j])
        return PACCurveMeasurement(out,baseline,valid,local_valid,global_valid,phase,
                                   self.harmonic_support,cycles_out,condition)
