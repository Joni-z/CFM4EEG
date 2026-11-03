"""PAC-conditioned content transforms on the original 20-position grid.

No independent PAC token or content-independent additive PAC vector. The old
rotation and projections are preserved. This research module is not frozen by
construction; callers freeze it explicitly for representation experiments.
"""
import copy
import math
import torch
from torch import nn
from paclock_bench.models.paclock.frontend.analytic import hilbert,phase_amplitude
from paclock_bench.models.paclock.frontend.triaxial import patch_pac_vector,_patch_project
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator


class PACContentFusion(nn.Module):
    def __init__(self,frontend):
        super().__init__()
        if frontend.sinc.sample_rate!=200 or frontend.patch_len!=50:
            raise ValueError('this prototype targets the original 200-Hz / 50-sample frontend')
        if frontend.waveform_modulation!='quadrature' or frontend.interaction_mode!='rotation':
            raise ValueError('content fusion requires the quadrature rotation frontend')
        self.original=copy.deepcopy(frontend)
        self.operator=CFCOperator()
        target=self.operator.physical_grid()['target_centers']
        t=torch.arange(frontend.sinc.kernel_size)/frontend.sinc.sample_rate
        self.register_buffer('target_cos',torch.cos(2*torch.pi*t[:,None]*target[None]))
        self.register_buffer('target_sin',torch.sin(2*torch.pi*t[:,None]*target[None]))
        self.register_buffer('residual_scale',torch.ones(1,1,frontend.n_bands,1,1))

    def mapping(self):
        f=self.original.sinc.filters().squeeze(1)
        power=(f@self.target_cos).square()+(f@self.target_sin).square()
        peak=torch.fft.rfft(f,n=2048).abs().square().amax(-1,keepdim=True)
        response=(power/peak.clamp_min(1e-12)).clamp(0,1)
        response=torch.where(response>=.5,response,0.)
        return response/response.sum(-1,keepdim=True).clamp_min(1e-12)

    def prepare(self,x):
        f=self.original;b,c,t=x.shape
        filtered=f.sinc(x.reshape(b*c,1,t)).reshape(b,c,f.n_bands,t)
        phase,amplitude=phase_amplitude(hilbert(filtered))
        z=patch_pac_vector(phase,amplitude,t//f.patch_len,f.normalize)
        baseline=f._interaction_tokens(phase,amplitude,z,filtered=filtered)
        measured=self.operator(x)
        return dict(baseline=baseline,filtered=filtered,phase=phase,amplitude=amplitude,
                    z=z,measured=measured,coefficient=self.coefficients(measured))

    @staticmethod
    def coefficients(measured):
        return torch.view_as_complex(measured.relation_features(stabilized=False)['features'].contiguous())

    def modulation_trace(self,prepared,coefficient):
        ph=prepared['measured'].source_phase
        ph=ph*prepared['measured'].phase_valid
        beta=coefficient[:,:,0] # one global estimate, not independent patches
        power=beta.abs().square().sum(-1).sqrt()
        weight=power/power.sum(-2,keepdim=True).clamp_min(1e-12)
        curve=ph.real.new_zeros(*beta.shape[:2],beta.shape[3],ph.shape[-1])
        for h in range(beta.shape[-1]):
            curve+=torch.einsum('bcsj,bcst->bcjt',weight*beta[...,h].conj(),ph**(h+1)).real
        return torch.einsum('nj,bcjt->bcnt',self.mapping(),curve)

    def carrier(self,prepared):
        f=self.original;amplitude=prepared['amplitude'];filtered=prepared['filtered']
        b,c,nb,t=amplitude.shape
        a=_patch_project(f.amplitude_tokenizer,torch.log1p(amplitude).reshape(-1,t))
        # Remove the projection bias from the correction branch. PAC cannot
        # produce a correction through a constant learned content bias alone.
        if f.amplitude_tokenizer.bias is not None:a=a-f.amplitude_tokenizer.bias
        a=a*f.amplitude_scale
        w=_patch_project(f.waveform_quadrature,filtered.reshape(-1,t))
        if not f.waveform_content:w=w*0
        return torch.complex(a,w).reshape(b,c,nb,t//f.patch_len,-1)

    def fuse(self,prepared,*,mode,gain,coefficient=None):
        if not math.isfinite(gain) or gain<0:
            raise ValueError('gain must be finite and nonnegative')
        base=prepared['baseline']
        if gain==0:return base
        coefficient=prepared['coefficient']if coefficient is None else coefficient
        if mode=='trace_gain':
            gate=1+gain*torch.tanh(self.modulation_trace(prepared,coefficient))
            return self.original._interaction_tokens(prepared['phase'],prepared['amplitude']*gate,
                prepared['z'],filtered=prepared['filtered']*gate)
        if mode not in ('carrier_binding','aligned_binding','radial_binding'):raise ValueError(mode)
        relation=torch.einsum('nj,bcpsjh->bcpnsh',self.mapping().to(coefficient.dtype),coefficient).flatten(-2)
        relation=relation.permute(0,1,3,2,4)
        old=torch.view_as_complex(base.reshape(*base.shape[:-1],-1,2).contiguous())
        relation=relation.repeat(1,1,1,1,(old.shape[-1]+relation.shape[-1]-1)//relation.shape[-1])[...,:old.shape[-1]]
        if mode=='radial_binding':
            # Keep the original complex content, but use a nonnegative local
            # content magnitude to carry the relation correction. Multiplying
            # by the signed/rotated carrier can cancel preferred phase when
            # the readout pools over time. No content -> no correction.
            content=self.carrier(prepared).abs()
        else:
            content=self.carrier(prepared)if mode=='carrier_binding'else old
        delta=content*relation*self.residual_scale
        power=old.abs().square().mean((-1,-2),keepdim=True)
        # Unmapped bands have exactly zero correction. Clamp before sqrt so
        # their zero-gradient path never evaluates the derivative of sqrt(0).
        norm=torch.where(power>0,power.clamp_min(1e-24).sqrt(),0.)
        size=delta.abs().square().mean((-1,-2),keepdim=True).clamp_min(1e-24).sqrt()
        delta=delta*torch.minimum(torch.ones_like(size),norm/size)
        return torch.view_as_real(old+gain*delta).flatten(-2)

    def forward(self,x,*,mode='trace_gain',gain=.1):
        return self.fuse(self.prepare(x),mode=mode,gain=gain)
