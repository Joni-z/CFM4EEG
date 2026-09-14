"""Source-separated duplex encoding and late, non-destructive interaction."""
import torch
from torch import nn


def encode_sources(encoder, tokens, coupling, pac_vector=None):
    """Frontend order is waveform first, coupling second. Share all encoder weights.

    Source groups become independent batch items, so no attention/normalization
    spans sources. Joint gradients still update shared parameters.
    """
    if tokens.shape[2] != 2 * coupling.shape[-1]:
        raise ValueError('duplex routing requires two complete band groups')
    wave, cpl = tokens.chunk(2, dim=2)
    grouped = torch.cat((wave, cpl), dim=0)
    paired_coupling = torch.cat((coupling, coupling), dim=0)
    paired_pac = None if pac_vector is None else torch.cat((pac_vector, pac_vector), dim=0)
    encoded = encoder(grouped, paired_coupling, paired_pac)
    return encoded.chunk(2, dim=0)


class DuplexReadout(nn.Module):
    """One task head reads intact C, intact W, and C-query/W-value interaction."""
    def __init__(self, d_model, n_heads, num_classes):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.content_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.pool_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.proj = nn.Linear(3 * d_model, num_classes)

    def forward(self, wave, cpl):
        if wave.shape != cpl.shape:
            raise ValueError('duplex streams must have matching grids')
        b, c, bands, patches, d = cpl.shape
        q = self.query_norm(cpl).permute(0, 1, 3, 2, 4).reshape(-1, bands, d)
        kv = self.content_norm(wave).permute(0, 1, 3, 2, 4).reshape(-1, bands, d)
        joint, _ = self.cross(q, kv, kv, need_weights=False)
        joint = joint.reshape(b, c, patches, bands, d).permute(0, 1, 3, 2, 4)
        pooled = [norm(x.mean(dim=(1, 2, 3)))
                  for norm, x in zip(self.pool_norms, (cpl, wave, joint))]
        return self.proj(torch.cat(pooled, dim=-1))
