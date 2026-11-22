"""TFM-Tokenizer as a replacement frontend inside BIOT.

Same surgery as quad_range_adapter.QuadBIOTEncoder: the native BIOT model is
built first, its patch embedding is replaced, and the transformer, channel
tokens, positional encoding and classifier stay as released. The frontend is
TFM's released tokenizer, frozen exactly as upstream uses it: STFT -> tokenize
-> quantised 64-d code -> Linear(64, width) + LayerNorm. TFM's STFT is
n_fft 200 / hop 100 / center False, the grid BIOT's own spectrogram stem uses,
so the host sees the same number of windows per channel as with its native
stem. Compared with the CroFreMo arm this frontend is *favoured*: its codebook
was pretrained on the corpus (single setting) or on four corpora (multiple),
while CroFreMo's frontend trains from scratch inside the host.
"""
import torch
from torch import nn
from einops import rearrange

from .tfm_adapter import (_import_tfm, weight_paths, CODE_BOOK_SIZE, EMB_SIZE,
                          RESAMPLING_RATE)


class TFMHostTokens(nn.Module):
    def __init__(self, width, dataset, setting='single'):
        super().__init__()
        mk_tok, _, stft_fn = _import_tfm()
        self.tokenizer = mk_tok(code_book_size=CODE_BOOK_SIZE, emb_size=EMB_SIZE)
        tok_path, _ = weight_paths(dataset, setting)
        state = torch.load(tok_path, map_location='cpu', weights_only=False)
        missing, unexpected = self.tokenizer.load_state_dict(state, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if not k.startswith('decoder')]
        if bad:
            raise RuntimeError(f'TFM tokenizer weights mismatch: {bad[:5]}')
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)
        self._stft = stft_fn
        self.projection = nn.Sequential(nn.Linear(EMB_SIZE, width), nn.LayerNorm(width))
        print(f'  TFM host frontend: tokenizer {tok_path} (frozen), projection {EMB_SIZE}->{width}', flush=True)

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()          # never leaves eval, as in TFMClassifier
        return self

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f'Unexpected raw input {tuple(x.shape)}')
        B, C, T = x.shape
        with torch.no_grad():
            s = rearrange(self._stft(x, resampling_rate=RESAMPLING_RATE), 'B C F T -> (B C) F T')
            q, _, _ = self.tokenizer.tokenize(s, rearrange(x, 'B C T -> (B C) T'))   # (B C) W 64
        return rearrange(self.projection(q), '(B C) W D -> B C W D', B=B)


class TFMBIOTEncoder(nn.Module):
    """Mirror of QuadBIOTEncoder with the TFM frontend in place of ours."""

    def __init__(self, native, dataset, setting):
        super().__init__()
        self.native = native
        self.quad = TFMHostTokens(256, dataset, setting)
        self.native.patch_embedding = nn.Identity()
        self.expected_windows = None

    def forward(self, x, n_channel_offset=0, perturb=False):
        if perturb:
            raise ValueError('Supervised host experiment only')
        t = self.quad(x)
        w = (x.shape[-1] - self.native.n_fft) // self.native.hop_length + 1
        if t.shape[2] != w:
            raise RuntimeError(f'TFM frontend gives {t.shape[2]} windows, BIOT stem would give {w}')
        seq = []
        for c in range(x.shape[1]):
            v = t[:, c] + self.native.channel_tokens(self.native.index[c + n_channel_offset])[None, None, :]
            seq.append(self.native.positional_encoding(v))
        return self.native.transformer(torch.cat(seq, 1)).mean(1)


def build_tfm_host(cfg, input_shape):
    from ..build import build_model
    native_cfg = dict(cfg)
    host = cfg['model'].removesuffix('_tfm_host')
    if host != 'biot':
        raise ValueError(host)
    native_cfg['model'] = host
    if cfg.get('pretrained') or cfg.get('checkpoint'):
        raise ValueError('This arm is a scratch host; pretrained transfer is a separate experiment')
    native = build_model(native_cfg, input_shape)
    with torch.random.fork_rng(devices=[]):
        native.biot = TFMBIOTEncoder(native.biot, cfg['dataset'], cfg.get('tfm_setting', 'single'))
    return native
