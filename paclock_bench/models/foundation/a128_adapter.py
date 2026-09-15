"""Duplex-A128 at its native 50-sample resolution, adapted to a host's exact grid.

The 8 fused rows followed by 8 coupling rows are concatenated; one linear map changes feature
width. Four consecutive 50-sample embeddings are averaged inside each native
200-sample host window. BIOT advances by its native hop; CBraMod by 200.
Only the input stem changes. Native encoder/position/classifier initialization
is preserved by building the complete native model before replacing its stem.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ..paclock.frontend.triaxial import TriAxialFrontend

class A128HostTokens(nn.Module):
    def __init__(self, width, hop, window=200):
        super().__init__()
        if window != 200 or hop not in (100, 200):
            raise ValueError('Audited grids: window200, hop100/200 only')
        self.hop, self.window = hop, window
        self.frontend = TriAxialFrontend(n_bands=8, hidden_dim=128,
            sample_rate=200, kernel_size=201, patch_len=50, pac_patch_len=50,
            tokenizer_mode='duplex', pac_token_mode='measured',
            interaction_mode='rotation')
        self.projection = nn.Sequential(nn.Linear(16*128, width), nn.LayerNorm(width))

    def _channel(self, x):
        t = self.frontend(x)[0]  # B,1,16,P,128
        t = t[:,0].permute(0,2,1,3).flatten(2)  # B,P,2048
        # Sliding windows are indexed at exactly the native host boundaries.
        t = t.unfold(1,4,self.hop//50).mean(-1)
        return self.projection(t)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] % 50 or x.shape[-1] < 200:
            raise ValueError(f'Unexpected raw input {tuple(x.shape)}')
        parts=[]
        for c in range(x.shape[1]):
            channel=x[:,c:c+1]
            # Avoid multiplying the native host's batch size by all 16 bands
            # in retained activations. This does not change effective batch.
            fn=lambda a: self._channel(a)
            part=checkpoint(fn,channel,use_reentrant=False) if self.training and torch.is_grad_enabled() else fn(channel)
            parts.append(part)
        return torch.stack(parts,1)

class A128CBraEmbedding(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.positional_encoding=native.positional_encoding
        self.quad=A128HostTokens(200,200)
    def forward(self,x,mask=None):
        if mask is not None:
            raise ValueError('Supervised host adapter does not implement host SSL masks')
        b,c,p,w=x.shape
        assert w==200
        t=self.quad(x.reshape(b,c,p*w))
        assert t.shape==(b,c,p,200)
        return t+self.positional_encoding(t.permute(0,3,1,2)).permute(0,2,3,1)

class A128BIOTEncoder(nn.Module):
    def __init__(self,native):
        super().__init__()
        self.native=native
        self.quad=A128HostTokens(256,native.hop_length,native.n_fft)
        self.native.patch_embedding=nn.Identity()
    def forward(self,x,n_channel_offset=0,perturb=False):
        if perturb:
            raise ValueError('Supervised host experiment only')
        t=self.quad(x)
        seq=[]
        # Preserve upstream per-channel positional dropout call order.
        for c in range(x.shape[1]):
            v=t[:,c]+self.native.channel_tokens(self.native.index[c+n_channel_offset])[None,None,:]
            seq.append(self.native.positional_encoding(v))
        return self.native.transformer(torch.cat(seq,1)).mean(1)

def build_a128_host(cfg,input_shape):
    from ..build import build_model
    native_cfg=dict(cfg)
    host=cfg['model'].removesuffix('_a128')
    native_cfg['model']=host
    if cfg.get('pretrained') or cfg.get('checkpoint'):
        raise ValueError('This arm is a scratch host; pretrained transfer is a separate experiment')
    native=build_model(native_cfg,input_shape)
    # Fork RNG: replacing a stem must not change training/dropout RNG state.
    with torch.random.fork_rng(devices=[]):
        if host=='cbramod':
            native.backbone.patch_embedding=A128CBraEmbedding(native.backbone.patch_embedding)
        elif host=='biot':
            native.biot=A128BIOTEncoder(native.biot)
        else: raise ValueError(host)
    return native
