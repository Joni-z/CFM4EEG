"""Cycle-block uncertainty and phase-preserving shrinkage for measured PAC.

Cluster-sandwich covariance is a finite-record diagnostic, not a calibrated
confidence interval. Shrinkage is a signal/noise-power rule with a zero-centered
prior; raw estimates remain available. It does not introduce a learned fusion.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from .measurement import linear_analytic_filter
from .measurement_curve import PACCurveOperator, PACCurveMeasurement, curve_regression


def coefficient_uncertainty(phase: Tensor, envelope: Tensor, mask: Tensor,
                            fit: dict, order: int, block_cycles: float = 2.,
                            min_blocks: int = 4, ridge: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Cluster sandwich for detrended harmonic ridge regression, no covariates.

    Regressor residualization matches curve_regression with amplitude control
    disabled. A block needs at least one observed phase cycle. Covariance stays
    in amplitude units squared and rotates with the harmonic coordinates.
    """
    count = mask.sum(-1)
    w = mask.double()/count.clamp_min(1)[...,None]
    t = torch.linspace(-1,1,phase.shape[-1],device=phase.device)
    t = t-(t*w).sum(-1)[...,None]
    design = torch.stack([v for k in range(1,order+1)
                          for v in ((phase**k).real,(phase**k).imag)],-1).double()
    design = design-(design*w[...,None]).sum(-2)[...,None,:]
    scale=(t.square()*w).sum(-1).clamp_min(1e-12)
    slope=(design*t[...,None]*w[...,None]).sum(-2)/scale[...,None]
    residualized=design-t[...,None]*slope[...,None,:]
    gram=residualized.transpose(-1,-2)@(residualized*w[...,None])
    penalty=ridge*gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-6)+1e-10
    inverse=torch.linalg.inv(gram+penalty[...,None,None]*torch.eye(2*order,device=phase.device))
    residual=(envelope-fit['prediction']).double()
    score=residualized*residual[...,None]*w[...,None]
    with torch.no_grad():
        step=torch.angle(phase[...,1:]*phase[...,:-1].conj())
        path=torch.nn.functional.pad(step.cumsum(-1),(1,0))
        groups=torch.floor(path/(2*torch.pi*block_cycles)).long().clamp_min(0)
        n=int(groups.max())+1
    buckets=torch.zeros(*score.shape[:-2],n,2*order,device=phase.device,dtype=torch.float64)
    buckets.scatter_add_(-2,groups[...,None].expand_as(score),score)
    # An observed cycle is measured from phase increments, not a nominal rate.
    observed=torch.zeros(*mask.shape[:-1],n,device=phase.device,dtype=torch.float64)
    observed.scatter_add_(-1,groups[...,1:],step.clamp_min(0).double()*
                         mask[...,1:]*mask[...,:-1]/(2*torch.pi))
    complete=observed>=.98
    buckets=buckets*complete[...,None]
    number=complete.sum(-1)
    meat=buckets.transpose(-1,-2)@buckets
    cov=inverse@meat@inverse
    cov=cov*(number/number.sub(1).clamp_min(1))[...,None,None]
    variance=cov.diagonal(dim1=-2,dim2=-1).reshape(*count.shape,order,2).sum(-1).float()
    good=(number>=min_blocks)&torch.isfinite(variance).all(-1)&fit['valid']
    return torch.where(good[...,None],variance.clamp_min(0),0),good


@dataclass
class UncertainPACMeasurement(PACCurveMeasurement):
    raw_coef: Tensor
    coefficient_variance: Tensor
    uncertainty_valid: Tensor
    reliability: Tensor

    @property
    def harmonic_valid(self):
        return self.valid[...,None]&self.harmonic_support

    def relation_features(self,stabilized: bool=True):
        """Dimensionless coefficients and separate quality masks, no hard PAC gate.

        real/imag is the final axis of features [B,C,P,S,T,K,2]. Unsupported
        harmonics are zero; quality masks remain explicit. Global mode repeats
        one record estimate across P, not P independent local measurements.
        """
        coef=self.raw_coef*self.reliability if stabilized else self.raw_coef
        normalized=torch.where(self.harmonic_valid,
            coef/self.baseline[...,None].clamp_min(1e-12),0.)
        variance=self.coefficient_variance/self.baseline[...,None].square().clamp_min(1e-24)
        return dict(features=torch.view_as_real(normalized),
                    harmonic_valid=self.harmonic_valid,
                    uncertainty_valid=self.uncertainty_valid[...,None]&self.harmonic_support,
                    variance=torch.where(self.harmonic_valid,variance,0.),
                    reliability=torch.where(self.harmonic_valid,self.reliability,0.))


class UncertainPACOperator(PACCurveOperator):
    """Experimental global PAC measurement with explicit uncertainty outputs."""
    def __init__(self,*args,block_cycles: float=1.,shrink: bool=False,**kwargs):
        super().__init__(*args,**kwargs)
        if self.local or self.control_amplitude:
            raise ValueError('uncertainty prototype currently requires global mode without amplitude covariates')
        if block_cycles<1:
            raise ValueError('a block must span at least one cycle')
        self.block_cycles,self.shrink=block_cycles,shrink
        self.configuration['uncertainty']=dict(block_cycles=block_cycles,shrink=shrink)

    def physical_grid(self):
        """Fixed measurement coordinates; these are not content-token indices."""
        return dict(source_centers=self.analysis.centers[:self.n_source].clone(),
                    source_cells=torch.stack([self.search_low,self.search_high],-1),
                    target_centers=self.analysis.centers[self.n_source:].clone(),
                    harmonic_support=self.harmonic_support.clone())

    def forward(self,x):
        if x.device.type not in ('cpu','cuda'):
            raise ValueError('PAC regression requires CPU or CUDA float64 linear algebra')
        return super().forward(x)

    def _forward(self,x):
        out=super()._forward(x)
        envelope=linear_analytic_filter(torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.),
                                       self.analysis.amplitude_kernels[self.n_source:]).abs()
        variance=torch.zeros_like(out.coef.real)
        uncertainty_valid=torch.zeros_like(out.valid)
        reliability=torch.ones_like(out.coef.real)
        times=torch.arange(x.shape[-1],device=x.device)
        for i in range(self.n_source):
            guard=torch.ceil(3*self.phase_cycles/(2*torch.pi*out.source_frequency[:,:,i])*self.analysis.sample_rate).long()
            for j in range(self.n_target):
                order=int(out.harmonic_support[i,j].sum())
                if not order or not bool(out.global_valid[:,:,i,j].any()):
                    continue
                edge=guard.clamp_min(int(self.analysis.amplitude_guard[self.n_source+j]))
                mask=((times>=edge[...,None])&(times<x.shape[-1]-edge[...,None])).float()
                mask=mask*out.phase_valid[:,:,i]
                ph=out.source_phase[:,:,i];amp=envelope[:,:,j]
                fit=curve_regression(ph,amp,mask,harmonics=order,ridge=self.ridge)
                v,good=coefficient_uncertainty(ph,amp,mask,fit,order,self.block_cycles,ridge=self.ridge)
                good=good&out.global_valid[:,:,i,j]
                power=out.coef[:,:,0,i,j,:order].abs().square()
                r=torch.where(good[...,None],power/(power+v).clamp_min(1e-20),1.)
                variance[:,:,:,i,j,:order]=v[:,:,None,:]
                uncertainty_valid[:,:,:,i,j]=good[:,:,None]
                reliability[:,:,:,i,j,:order]=r[:,:,None,:]
        raw=out.coef
        return UncertainPACMeasurement(**(vars(out)|dict(
            coef=raw*reliability if self.shrink else raw,raw_coef=raw,
            coefficient_variance=variance,uncertainty_valid=uncertainty_valid,
            reliability=reliability)))
