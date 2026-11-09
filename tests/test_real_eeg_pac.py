import numpy as np
import torch
from scipy.signal import hilbert
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from paclock_bench.probing.real_eeg import (
    BANDS, PAIRS, R2Moments, RidgeMoments, pac_targets, robust_scale,
)


def reference(signal):
    freq = np.fft.fftfreq(len(signal), 1 / 200)
    bands = [hilbert(np.fft.ifft(np.fft.fft(signal) *
              ((np.abs(freq) >= lo) & (np.abs(freq) < hi))).real) for lo, hi in BANDS]
    result = []
    for i, j in PAIRS:
        phase, amplitude = np.angle(bands[i]), np.abs(bands[j])
        ids = (np.floor((phase + np.pi) * 18 / (2 * np.pi)).astype(int) % 18)
        means = np.array([amplitude[ids == b].mean() if (ids == b).any() else 0 for b in range(18)])
        p = means / means.sum()
        result.append(1 + (p[p > 0] * np.log(p[p > 0])).sum() / np.log(18))
    return result


def test_fft_tort_matches_independent_scipy_and_channel_scaling():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 5, 1000))
    actual, _ = pac_targets(torch.from_numpy(x))
    per_channel = np.array([[reference(c) for c in row] for row in x])
    expected = np.stack((per_channel.mean(1), per_channel.std(1), np.median(per_channel, 1),
                         np.quantile(per_channel, .25, 1), np.quantile(per_channel, .75, 1),
                         per_channel.max(1)), -1)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10)
    scaled, _ = pac_targets(torch.from_numpy(x * rng.uniform(.01, 100, (4, 5, 1))))
    np.testing.assert_allclose(scaled.numpy(), actual.numpy(), atol=1e-10)


def test_known_amplitude_modulation_and_zero_inputs():
    t = np.arange(2000) / 200
    null = np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 37.5 * t)
    coupled = np.cos(2 * np.pi * 5 * t) + (1 + .8 * np.cos(2 * np.pi * 5 * t)) * np.cos(2 * np.pi * 37.5 * t)
    target, _ = pac_targets(torch.tensor(np.stack((null, coupled))[:, None]))
    assert target[1, 1, 0] > target[0, 1, 0] + .04
    zero, qc = pac_targets(torch.zeros(2, 3, 1000))
    assert torch.isfinite(zero).all() and not zero.any()
    assert qc['degenerate_channel_pairs'] == 18


def test_streaming_standardized_ridge_matches_sklearn():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(237, 13)) * np.geomspace(.01, 100, 13) + 3
    x[:, -1] = 5
    y = rng.normal(size=(237, 18)) + x[:, :1] * .3
    stats = RidgeMoments(13, 18)
    for a in range(0, len(x), 7):
        stats.update(x[a:a+7], y[a:a+7])
    scaler = StandardScaler().fit(x)
    for alpha in (.1, 1, 10, 100):
        coef, intercept = stats.fit(alpha)
        expected = Ridge(alpha=alpha).fit(scaler.transform(x), y).predict(scaler.transform(x))
        np.testing.assert_allclose(x @ coef + intercept, expected, atol=2e-10)


def test_r2_accumulator_and_train_only_target_scaling():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(130, 18))
    p = y + rng.normal(size=y.shape) * 1.2
    score = R2Moments(18)
    for start in range(0, len(y), 9):
        score.update(y[start:start+9], p[start:start+9])
    raw = r2_score(y, p, multioutput='raw_values')
    np.testing.assert_allclose(score.raw(), raw)
    np.testing.assert_allclose(score.groups(), np.maximum(0, raw).reshape(3, 6).mean(1))
    train = np.column_stack((np.arange(20), np.zeros(20)))
    center, scale = robust_scale(train)
    np.testing.assert_equal(center, [9.5, 0])
    np.testing.assert_equal(scale, [9.5, 1])


def test_frozen_hooks_pool_hidden_axis_and_return_no_grad():
    from paclock_bench.probing.real_eeg_models import FrozenReadout
    stub = object.__new__(FrozenReadout)
    stub.features = {}
    x = torch.randn(2, 4, 3, 8)
    stub._hook('layer')(None, None, (x, torch.zeros(1)))
    np.testing.assert_allclose(stub.features['layer'], x.mean((1, 2)).double().numpy())
