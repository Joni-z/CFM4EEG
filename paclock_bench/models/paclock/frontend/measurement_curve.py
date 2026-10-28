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
        cv = mean((a-mean(a)[..., None]).square()).clamp_min(1e-12).sqrt()/mean(a).clamp_min(1e-12)
        a = a.clamp_min(1e-12).log()
        a = a-mean(a)[..., None]
        sd = mean(a.square()).clamp_min(1e-12).sqrt()
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
    # A tiny nuisance ridge keeps absent columns differentiable. A truncated
    # pseudoinverse has undefined derivatives at repeated zero eigenvalues.
    nd, wd = nuisance.double(), w.double()
    ng = nd.transpose(-1, -2) @ (nd*wd[..., None])
    nscale=ng.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-4)
    inverse = torch.linalg.inv(ng+1e-8*nscale[...,None,None]*torch.eye(ng.shape[-1],device=phase.device))
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
    nuisance_coef=ny.squeeze(-1)-(nx@beta[...,None]).squeeze(-1)
    prediction=baseline[...,None]+(design.double()@beta[...,None]).squeeze(-1)+(nd@nuisance_coef[...,None]).squeeze(-1)
    null_prediction=my.double()[...,None]+(nd@ny).squeeze(-1)
    return dict(coef=coef, baseline=baseline.float(), valid=valid,
                mse=(residual.square()*wd).sum(-1).float(), count=count,
                condition=(eigen[..., -1]/eigen[..., 0].clamp_min(1e-12)).float(),
                prediction=prediction.float(),null_prediction=null_prediction.float())


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
    predictive_gain: Tensor    # [B,C,source,target], held-out folds vs nuisance model
    source_frequency: Tensor   # [B,C,source], raw-signal refinement or fixed center

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
                 harmonics: int = 2, control_amplitude: bool = False,
                 local: bool = False, ridge: float = 1e-3,
                 local_ridge: float = .05, amplitude_floor: float = .02,
                 phase_cycles: float = 9., refine_phase: bool = True,
                 confidence_folds: str = 'halves'):
        super().__init__()
        if harmonics not in (1,2):
            raise ValueError('harmonics must be 1 or 2')
        if confidence_folds not in ('halves','cycles'):
            raise ValueError('confidence_folds must be halves or cycles')
        if len(set(source_hz)) != len(source_hz) or len(set(target_hz)) != len(target_hz):
            raise ValueError('duplicate analysis centers')
        if not source_hz or not target_hz or any(a>=b for a,b in zip(source_hz,source_hz[1:])):
            raise ValueError('nonempty banks and increasing source centers required')
        self.configuration=dict(source_hz=list(source_hz),target_hz=list(target_hz),
            sample_rate=sample_rate,patch_len=patch_len,passband=list(passband),
            excluded_intervals=[list(v) for v in excluded_intervals],harmonics=harmonics,
            control_amplitude=control_amplitude,local=local,ridge=ridge,
            local_ridge=local_ridge,amplitude_floor=amplitude_floor,
            phase_cycles=phase_cycles,refine_phase=refine_phase,confidence_folds=confidence_folds)
        self.n_source, self.n_target = len(source_hz), len(target_hz)
        self.harmonics, self.control_amplitude = harmonics, control_amplitude
        self.local, self.ridge, self.local_ridge = local, ridge, local_ridge
        self.amplitude_floor = amplitude_floor
        self.phase_cycles,self.refine_phase=phase_cycles,refine_phase
        self.confidence_folds=confidence_folds
        centers = tuple(source_hz)+tuple(target_hz)
        self.analysis = PhaseAmplitudeMeasurement(centers, sample_rate, patch_len,
            passband=passband, excluded_intervals=excluded_intervals,phase_cycles=phase_cycles)
        sc=self.analysis.centers[:self.n_source]
        search_low=torch.cat([self.analysis.phase_lower[:1],(sc[:-1]+sc[1:])/2])
        search_high=torch.cat([(sc[:-1]+sc[1:])/2,self.analysis.phase_upper[self.n_source-1:self.n_source]])
        self.register_buffer('search_low',search_low)
        self.register_buffer('search_high',search_high)
        support = self.analysis.eligible[:self.n_source, self.n_source:, None].expand(-1,-1,harmonics).clone()
        for i in range(self.n_source):
            for j in range(self.n_target):
                kernel = self.analysis.amplitude_kernels[self.n_source+j].to(torch.complex128)
                f = float(self.analysis.centers[self.n_source+j])
                for k in range(1,harmonics+1):
                    if not support[i,j,k-1]:
                        continue
                    edge = k*float(search_high[i]*(1+2/phase_cycles) if refine_phase else self.analysis.phase_upper[i])
                    width=min(.4*f,.8*(f-passband[0]),.8*(passband[1]-f))
                    response = _gain(kernel, torch.tensor([f-edge,f,f+edge]), sample_rate)
                    ok = bool(response[1]>0 and torch.minimum(response[0],response[2])/response[1]>=.9)
                    ok = ok and edge<=width
                    source_upper=float(search_high[i]*(1+2/phase_cycles) if refine_phase else self.analysis.phase_upper[i])
                    source_lower=float(search_low[i]*(1-2/phase_cycles) if refine_phase else self.analysis.phase_lower[i])
                    ok = ok and source_upper<f-1.25*width and source_lower>=passband[0]
                    for low,high in excluded_intervals:
                        ok = ok and not (f-edge <= high and f+edge >= low)
                        ok = ok and not (source_lower<=high and source_upper>=low)
                    support[i,j,k-1] = ok
        self.register_buffer('harmonic_support',support)

    def get_extra_state(self) -> dict:
        return dict(version=1,configuration=self.configuration)

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError('operator checkpoint configuration differs from constructor')

    def _source_analysis(self, x: Tensor) -> tuple[Tensor,Tensor,Tensor]:
        """Estimate a peak inside each fixed cell, then apply steered Morlet.

        The peak is estimated from raw input, with no generator parameters.
        Disjoint search cells prevent multiple anchors from chasing one peak.
        """
        b,c,length=x.shape;fs=self.analysis.sample_rate
        centers=self.analysis.centers[:self.n_source]
        if not self.refine_phase:
            source=linear_analytic_filter(x,self.analysis.phase_kernels[:self.n_source])
            return (source,centers.view(1,1,-1).expand(b,c,-1),
                    self.analysis.phase_guard[:self.n_source].view(1,1,-1).expand(b,c,-1))
        with torch.no_grad():
            frequency=torch.fft.rfftfreq(length,1/fs,device=x.device)
            spectrum=torch.fft.rfft(x*torch.hann_window(length,device=x.device),dim=-1).abs().square()
            estimates=[]
            for i,center in enumerate(centers):
                low,high=self.search_low[i],self.search_high[i]
                allowed=(frequency>=low)&(frequency<=high)
                if not bool(allowed.any()):
                    estimates.append(center.expand(b,c));continue
                peak=torch.where(allowed,spectrum,-1.).argmax(-1).clamp(1,len(frequency)-2)
                logpower=spectrum.clamp_min(1e-20).log()
                left=logpower.gather(-1,(peak-1)[...,None]).squeeze(-1)
                mid=logpower.gather(-1,peak[...,None]).squeeze(-1)
                right=logpower.gather(-1,(peak+1)[...,None]).squeeze(-1)
                curvature=left-2*mid+right
                offset=torch.where(curvature.abs()>1e-8,.5*(left-right)/curvature,0.).clamp(-.5,.5)
                estimates.append(((peak+offset)*fs/length).clamp(low,high))
            refined=torch.stack(estimates,-1)
            # Eligibility uses the worst upper support across the entire
            # fixed search cell. No artificial long kernel is needed.
            cycles=torch.full_like(refined,self.phase_cycles)
            guards=torch.ceil(3*cycles/(2*math.pi*refined)*fs).long()
        radius=int(guards.max());t=torch.arange(-radius,radius+1,device=x.device)/fs
        sigma=cycles/(2*math.pi*refined)
        gaussian=torch.exp(-t.square()/(2*sigma[...,None].square()))
        gaussian=gaussian*(torch.arange(-radius,radius+1,device=x.device).abs()<=guards[...,None])
        wave=torch.exp(2j*math.pi*refined[...,None]*t)
        kernel=gaussian*(wave-torch.exp(-cycles.square()/2)[...,None])
        gain=(kernel*wave.conj()).sum(-1).abs().clamp_min(1e-12)
        kernel=2*kernel/gain[...,None]
        source=linear_analytic_filter(x,kernel)
        return source,refined,guards

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
        source,source_frequency,phase_guards=self._source_analysis(clean)
        envelope=linear_analytic_filter(clean,self.analysis.amplitude_kernels[ns:]).abs()
        amplitude=source.abs(); phase=source/amplitude.clamp_min(1e-12)
        out=clean.new_zeros(b,c,patches,ns,nt,self.harmonics,dtype=torch.complex64)
        baseline=clean.new_zeros(b,c,patches,ns,nt)
        valid=torch.zeros_like(baseline,dtype=torch.bool); local_valid=valid.clone()
        global_valid=valid[:,:,0].clone(); cycles_out=baseline[:,:,0].clone()
        condition=baseline.clone(); times=torch.arange(length,device=x.device)
        predictive_gain=cycles_out.clone()
        rms=clean.square().mean(-1).sqrt().clamp_min(1e-8)
        for i in range(ns):
            for j in range(nt):
                order=int(self.harmonic_support[i,j].sum())
                if order==0:
                    continue
                guard=phase_guards[:,:,i].clamp_min(int(self.analysis.amplitude_guard[ns+j]))
                mask=(times>=guard[...,None])&(times<length-guard[...,None])&finite[...,None]
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
                lower=source_frequency[:,:,i]*(1-2/self.phase_cycles)
                upper=source_frequency[:,:,i]*(1+2/self.phase_cycles)
                enough=(fit['count']-1)*lower/self.analysis.sample_rate>=2.98
                good=fit['valid']&enough&(cycles>=2.98)&(freq>=lower)&(freq<=upper)
                global_valid[:,:,i,j]=good;cycles_out[:,:,i,j]=cycles
                if bool(good.any()):
                    # Out-of-half prediction is diagnostic confidence, not a
                    # p-value. Record-level calibration happens independently.
                    with torch.no_grad():
                        error=amp.new_zeros(b,c);null_error=error.clone();cv_good=good.clone()
                        if self.confidence_folds=='cycles':
                            phase_path=torch.nn.functional.pad(torch.angle(steps).cumsum(-1),(1,0))
                            fold_id=torch.floor(phase_path/(2*math.pi)).long().remainder(2)
                        else:
                            fold_id=(times>=length//2).long().expand_as(ph.real)
                        for half in (0,1):
                            training=mask&(fold_id==half)
                            testing=mask&~training
                            fold=curve_regression(ph.detach(),amp.detach(),training.float(),
                                None if source_amp is None else source_amp.detach(),order,self.ridge,
                                amplitude_floor=self.amplitude_floor)
                            error+=((amp-fold['prediction']).square()*testing).sum(-1)
                            null_error+=((amp-fold['null_prediction']).square()*testing).sum(-1)
                            cv_good&=fold['valid']
                        gain=(1-error/null_error.clamp_min(1e-12)).clamp(0,1)
                        predictive_gain[:,:,i,j]=torch.where(cv_good,gain,0.)
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
                                   self.harmonic_support,cycles_out,condition,predictive_gain,source_frequency)
