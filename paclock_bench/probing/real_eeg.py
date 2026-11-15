"""Real-EEG PAC/AAC targets and bounded-memory ridge probing.

Protocol: Tang et al., arXiv:2605.11410v2, Appendix A (Family X),
C.3--C.5 and D. The original audit code is not publicly linked there.
NumPy/PyTorch implement the stated FFT/Tort definition; ridge sufficient
statistics below solve the same standardized affine ridge objective as sklearn.
"""
from __future__ import annotations

import numpy as np
import torch

BANDS = ((.5, 4.), (4., 8.), (8., 13.), (13., 30.), (30., 45.))
PAIRS = ((0, 3), (1, 4), (2, 4))
FEATURES = ("X001_delta_beta", "X002_theta_gamma", "X003_alpha_gamma")
AAC_PAIRS = ((1, 4), (0, 4))
CFC_FEATURES = FEATURES + ("X004_aac_theta_gamma", "X005_aac_delta_gamma")
AAC_FEATURES = CFC_FEATURES[3:]
SUMMARIES = ("mean", "std", "median", "q25", "q75", "max")
ALPHAS = (.1, 1., 10., 100.)


@torch.inference_mode()
def pac_targets(signals: torch.Tensor, sample_rate: int = 200, *,
                gamma_high: float = 45.) -> tuple[torch.Tensor, dict]:
    """[N,C,T] -> [N,3,6], using whole epochs and 18 equal phase bins.

    Rectangular masks are [low,high); analytic FFT doubles positive frequencies.
    Empty phase bins receive zero mean amplitude. Entirely zero-energy pairs
    receive MI=0; these cases and empty bins are counted rather than hidden.
    Positive channel rescaling leaves these MI targets unchanged.
    Gamma defaults to 30--45 Hz; Appendix C.3 uses 30--35 Hz for ISRUC.
    """
    if signals.ndim != 3 or not torch.isfinite(signals).all():
        raise ValueError("Expected finite [N,C,T] EEG")
    if not 30 < gamma_high <= 45:
        raise ValueError("Gamma upper edge must lie in (30,45] Hz")
    x = signals.to(torch.float64)
    length = x.shape[-1]
    freq = torch.fft.fftfreq(length, 1 / sample_rate, device=x.device)
    spectrum = torch.fft.fft(x, dim=-1)
    bands = (*BANDS[:-1], (30., gamma_high))
    analytic = [torch.fft.ifft(spectrum * (2 * ((freq >= lo) & (freq < hi))), dim=-1)
                for lo, hi in bands]
    results, empty, degenerate = [], 0, 0
    for i, j in PAIRS:
        phase, amplitude = analytic[i].angle(), analytic[j].abs()
        bins = (((phase + torch.pi) * (18 / (2 * torch.pi))).floor().long() % 18)
        shape = (*x.shape[:2], 18)
        sums, counts = x.new_zeros(shape), x.new_zeros(shape)
        sums.scatter_add_(-1, bins, amplitude)
        counts.scatter_add_(-1, bins, torch.ones_like(amplitude))
        means = sums / counts.clamp_min(1)
        total = means.sum(-1, keepdim=True)
        valid = (analytic[i].abs().square().mean(-1) > 0) & (total.squeeze(-1) > 0)
        probability = means / total.clamp_min(torch.finfo(x.dtype).tiny)
        terms = probability * probability.clamp_min(torch.finfo(x.dtype).tiny).log()
        mi = 1 + terms.sum(-1) / np.log(18)
        mi = torch.where(valid, mi.clamp(0, 1), torch.zeros_like(mi))
        quantiles = torch.quantile(mi, x.new_tensor([.25, .5, .75]), dim=1)
        results.append(torch.stack((mi.mean(1), mi.std(1, correction=0), quantiles[1],
                                    quantiles[0], quantiles[2], mi.max(1).values), -1))
        empty += int((counts == 0).sum())
        degenerate += int((~valid).sum())
    return torch.stack(results, 1), dict(empty_phase_bins=empty, degenerate_channel_pairs=degenerate)


@torch.inference_mode()
def cfc_targets(signals: torch.Tensor, sample_rate: int = 200, *,
                gamma_high: float = 45.) -> tuple[torch.Tensor, dict]:
    """Appendix A family X: three Tort PAC and two signed envelope correlations."""
    pac, quality = pac_targets(signals, sample_rate, gamma_high=gamma_high)
    aac, aac_quality = aac_targets(signals, sample_rate, gamma_high=gamma_high)
    return torch.cat((pac, aac), dim=1), {**quality, **aac_quality}


@torch.inference_mode()
def aac_targets(signals: torch.Tensor, sample_rate: int = 200, *,
                gamma_high: float = 45.) -> tuple[torch.Tensor, dict]:
    """AAC-only targets let completed PAC readouts remain frozen and reusable."""
    if signals.ndim != 3 or not torch.isfinite(signals).all():
        raise ValueError("Expected finite [N,C,T] EEG")
    if not 30 < gamma_high <= 45:
        raise ValueError("Gamma upper edge must lie in (30,45] Hz")
    x = signals.to(torch.float64)
    freq = torch.fft.fftfreq(x.shape[-1], 1 / sample_rate, device=x.device)
    spectrum = torch.fft.fft(x, dim=-1)
    bands = (*BANDS[:-1], (30., gamma_high))
    envelopes = {i: torch.fft.ifft(spectrum * (2 * ((freq >= bands[i][0]) &
                 (freq < bands[i][1]))), dim=-1).abs() for i in (0, 1, 4)}
    results, degenerate = [], 0
    for i, j in AAC_PAIRS:
        a, b = envelopes[i], envelopes[j]
        a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
        denominator = (a.square().sum(-1) * b.square().sum(-1)).sqrt()
        valid = denominator > 0
        corr = torch.where(valid, (a * b).sum(-1) /
                           denominator.clamp_min(torch.finfo(x.dtype).tiny), 0.).clamp(-1, 1)
        quantiles = torch.quantile(corr, x.new_tensor([.25, .5, .75]), dim=1)
        results.append(torch.stack((corr.mean(1), corr.std(1, correction=0), quantiles[1],
                                    quantiles[0], quantiles[2], corr.max(1).values), -1))
        degenerate += int((~valid).sum())
    return torch.stack(results, 1), {"degenerate_aac_channel_pairs": degenerate}


def select_readouts(grid: dict, features: tuple, encoder_names: list,
                    *, layer_profiles: bool = False) -> list[dict]:
    """Select ridge strengths and feature peaks using validation scores only."""
    scopes = [("tokenizer", ["tokenizer"])]
    if encoder_names:
        scopes.append(("encoder_peak", encoder_names))
    if layer_profiles:
        scopes.extend(("family_layer/" + name, [name]) for name in encoder_names)
    selected = []
    for feature, name in enumerate(features):
        for scope, names in scopes:
            choices = [(float(grid[n][a, 0, feature]), n, a)
                       for n in names for a in range(len(ALPHAS))]
            score, endpoint, index = max(choices, key=lambda item: item[0])
            selected.append(dict(label=scope + "/" + name, scope=scope, feature=name,
                feature_index=feature, endpoint=endpoint, alpha=ALPHAS[index], alpha_index=index,
                validation_r2=score, validation_controls={c: float(grid[endpoint][:, i, feature].max())
                    for i, c in ((1, "shuffled"), (2, "gaussian"))}))
    return selected


def summarize_family(entries: list[dict], features: tuple) -> dict:
    """Median of feature scores within a common layer, never across peak layers."""
    layers = list(dict.fromkeys(e["endpoint"] for e in entries
                  if e["scope"].startswith("family_layer/")))
    profile = []
    for depth, name in enumerate(["tokenizer", *layers]):
        scope = "tokenizer" if name == "tokenizer" else "family_layer/" + name
        rows = [next(e for e in entries if e["scope"] == scope and e["feature"] == f)
                for f in features]
        profile.append(dict(endpoint=name,
            normalized_depth=depth / len(layers) if layers else 0.,
            validation_feature_r2=[e["validation_r2"] for e in rows],
            test_feature_r2=[e["test_r2"] for e in rows],
            validation_median_r2=float(np.median([e["validation_r2"] for e in rows])),
            test_median_r2=float(np.median([e["test_r2"] for e in rows]))))
    encoder = profile[1:]
    return dict(features=features, aggregation="median of five feature R2 scores at the same layer",
                tokenizer=profile[0], encoder_layer_profile=encoder,
                validation_selected_encoder=max(encoder, key=lambda e: e["validation_median_r2"])
                    if encoder else None,
                descriptive_test_peak=max(encoder, key=lambda e: e["test_median_r2"])
                    if encoder else None)


def robust_scale(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Appendix C.3: train-only median/IQR, mean/std fallback."""
    median = np.median(train, axis=0)
    iqr = np.quantile(train, .75, axis=0) - np.quantile(train, .25, axis=0)
    mean, std = train.mean(0), train.std(0)
    use_iqr = iqr > 1e-8
    scale = np.where(use_iqr, iqr, np.where(std > 1e-8, std, 1.))
    return np.where(use_iqr, median, mean), scale


class RidgeMoments:
    """Exact sufficient statistics; storage depends on width, not epoch count."""
    def __init__(self, width: int, outputs: int):
        self.n = 0
        self.sx = np.zeros(width)
        self.sy = np.zeros(outputs)
        self.xx = np.zeros((width, width))
        self.xy = np.zeros((width, outputs))

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("Expected nonempty aligned feature/target matrices")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite readout data")
        self.n += len(x)
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.xx += x.T @ x
        self.xy += x.T @ y

    def fit(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Return raw-X affine coefficients after train-only StandardScaler."""
        from scipy.linalg import solve
        if self.n < 2 or alpha <= 0:
            raise ValueError("Ridge requires at least two rows and positive alpha")
        mean, ymean = self.sx / self.n, self.sy / self.n
        centered = self.xx - self.n * np.outer(mean, mean)
        variance = np.maximum(np.diag(centered) / self.n, 0.)
        scale = np.sqrt(variance)
        # Match StandardScaler's numerical constant-feature criterion.
        eps = np.finfo(np.float64).eps
        constant = variance <= self.n * eps * variance + (self.n * mean * eps) ** 2
        scale[constant] = 1.
        gram = centered / np.outer(scale, scale)
        rhs = (self.xy - self.n * np.outer(mean, ymean)) / scale[:, None]
        coef = solve((gram + gram.T) / 2 + alpha * np.eye(len(mean)), rhs, assume_a="pos")
        coef /= scale[:, None]
        return coef, ymean - mean @ coef


class R2Moments:
    def __init__(self, outputs: int):
        self.n = 0
        self.sy, self.yy, self.sse = (np.zeros(outputs) for _ in range(3))

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target, prediction = np.asarray(target, np.float64), np.asarray(prediction, np.float64)
        self.n += len(target)
        self.sy += target.sum(0)
        self.yy += (target ** 2).sum(0)
        self.sse += ((target - prediction) ** 2).sum(0)

    def raw(self) -> np.ndarray:
        if self.n < 2:
            raise ValueError("R2 requires two rows")
        variance = np.maximum(self.yy - self.sy ** 2 / self.n, 0.)
        result = np.zeros_like(variance)
        nonconstant = variance > 0
        result[nonconstant] = 1 - self.sse[nonconstant] / variance[nonconstant]
        result[~nonconstant] = (self.sse[~nonconstant] == 0).astype(float)
        return result

    def groups(self) -> np.ndarray:
        """Clip per output before averaging six channel summaries, as in C.5."""
        return np.maximum(0, self.raw()).reshape(-1, 6).mean(-1)
