import numpy as np
import torch
from scipy.signal import hilbert
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from paclock_bench.probing.real_eeg import (
    BANDS, PAIRS, CFC_FEATURES, R2Moments, RidgeMoments, pac_targets,
    cfc_targets, robust_scale, select_readouts, summarize_family,
)


def test_sleep_streaming_rows_keep_epoch_order_across_sequences(tmp_path):
    import pytest
    from scripts.experiments.pac_real_eeg_probe import Rows

    stored = np.arange(3 * 20 * 2 * 10, dtype=np.float32).reshape(3, 20, 2, 10)
    path = tmp_path / 'sleep.npy'
    np.save(path, stored)
    with pytest.raises(ValueError, match='contiguous float32'):
        Rows(path)
    rows = Rows(path, flatten_sequences=True)
    expected = stored.reshape(60, 2, 10)
    assert rows.shape == expected.shape
    for start, stop in [(0, 2), (19, 23), (38, 42), (58, 60)]:
        np.testing.assert_array_equal(rows.read(start, stop), expected[start:stop])
    rows.close()


def test_ordinary_streaming_rows_are_unchanged(tmp_path):
    from scripts.experiments.pac_real_eeg_probe import Rows

    expected = np.arange(7 * 3 * 11, dtype=np.float32).reshape(7, 3, 11)
    path = tmp_path / 'epochs.npy'
    np.save(path, expected)
    for flatten in (False, True):
        rows = Rows(path, flatten_sequences=flatten)
        assert rows.shape == expected.shape
        np.testing.assert_array_equal(rows.read(1, 6), expected[1:6])
        rows.close()


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


def test_family_x_aac_matches_independent_scipy_and_preserves_pac():
    rng = np.random.default_rng(105)
    x = rng.normal(size=(3, 4, 1000))
    freq = np.fft.fftfreq(x.shape[-1], 1 / 200)
    for gamma_high in (35., 45.):
        actual, _ = cfc_targets(torch.from_numpy(x), gamma_high=gamma_high)
        pac, _ = pac_targets(torch.from_numpy(x), gamma_high=gamma_high)
        assert torch.equal(actual[:, :3], pac)
        envelopes = {}
        for i, (low, high) in enumerate((*BANDS[:-1], (30., gamma_high))):
            band = np.fft.ifft(np.fft.fft(x, axis=-1) *
                              ((np.abs(freq) >= low) & (np.abs(freq) < high)), axis=-1).real
            envelopes[i] = np.abs(hilbert(band, axis=-1))
        per_channel = np.array([[[np.corrcoef(envelopes[i][n, c], envelopes[j][n, c])[0, 1]
                                  for i, j in ((1, 4), (0, 4))]
                                 for c in range(x.shape[1])] for n in range(len(x))])
        expected = np.stack((per_channel.mean(1), per_channel.std(1),
                            np.median(per_channel, 1), np.quantile(per_channel, .25, 1),
                            np.quantile(per_channel, .75, 1), per_channel.max(1)), -1)
        np.testing.assert_allclose(actual[:, 3:].numpy(), expected, atol=1e-10)


def test_aac_keeps_negative_correlations_and_handles_zero_inputs():
    t = np.arange(2000) / 200
    envelope = .8 * np.cos(2 * np.pi * .5 * t)
    theta = (1 + envelope) * np.cos(2 * np.pi * 6 * t)
    opposite = theta + (1 - envelope) * np.cos(2 * np.pi * 32.5 * t)
    together = theta + (1 + envelope) * np.cos(2 * np.pi * 32.5 * t)
    x = torch.tensor(np.stack((opposite, together))[:, None])
    result, _ = cfc_targets(x, gamma_high=35.)
    np.testing.assert_allclose(result[:, 3, 0], [-1, 1], atol=1e-10)
    scaled, _ = cfc_targets(x * 17.2, gamma_high=35.)
    np.testing.assert_allclose(result[:, 3], scaled[:, 3], atol=1e-10)
    zero, quality = cfc_targets(torch.zeros(2, 3, 1000), gamma_high=35.)
    assert not zero.any() and torch.isfinite(zero).all()
    assert quality['degenerate_aac_channel_pairs'] == 12


def test_family_median_uses_one_layer_and_validation_selection():
    grid = {name: np.zeros((4, 3, 5)) for name in ['tokenizer', 'layer_01', 'layer_02']}
    grid['layer_01'][0, 0] = [.8, .2, .8, .2, .8]
    grid['layer_02'][1, 0] = [.2, .7, .2, .7, .2]
    selected = select_readouts(grid, CFC_FEATURES, ['layer_01', 'layer_02'], layer_profiles=True)
    test = {'tokenizer': [.05, .15, .25, .35, .45],
            'layer_01': [.1, .9, .2, .8, .3], 'layer_02': [.9, .1, .8, .2, .7]}
    for e in selected:
        e['test_r2'] = test[e['endpoint']][e['feature_index']]
    family = summarize_family(selected, CFC_FEATURES)
    assert family['validation_selected_encoder']['endpoint'] == 'layer_01'
    assert family['validation_selected_encoder']['test_median_r2'] == .3
    assert family['descriptive_test_peak']['endpoint'] == 'layer_02'
    assert family['descriptive_test_peak']['test_median_r2'] == .7
    assert family['tokenizer']['test_median_r2'] == .25


def test_five_feature_readout_pipeline_has_no_three_feature_dimensions(tmp_path, monkeypatch):
    import json
    from scripts.experiments import pac_real_eeg_probe as runner

    class FrozenFixture:
        def __init__(self, *_args):
            self.endpoints = ['tokenizer', 'layer_01', 'layer_02']
            self.source_hashes = {}

        def __call__(self, x, _scales):
            content = x.numpy().mean(-1).astype(np.float64)
            return {name: content for name in self.endpoints}

    rng = np.random.default_rng(7)
    for split, count in [('train', 12), ('val', 8), ('test', 8)]:
        np.save(tmp_path / f'{split}_signals.npy', rng.normal(size=(count, 3, 1000)).astype('float32'))
    job = dict(kind='cfft', dataset='isruc', data_root=str(tmp_path),
               targets_root=str(tmp_path / 'targets'), out=str(tmp_path / 'readout'),
               checkpoint_sha256='0' * 64, feature_family='X', gamma_high=35., batch_size=4)
    monkeypatch.setattr(runner, 'dataset_identity', lambda _job: {'fixture': True})
    monkeypatch.setattr(runner, 'headroom', lambda *_args: {})
    monkeypatch.setattr(runner, 'FrozenReadout', FrozenFixture)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    runner.targets(job, 'cpu', False)
    runner.run_model(job, 'cpu', False)
    result = json.loads((tmp_path / 'readout/result.json').read_text())
    assert np.load(tmp_path / 'targets/test_targets.npy').shape == (8, 30)
    assert len(result['endpoints']) == 20
    assert result['features'] == list(CFC_FEATURES)
    assert len(result['family_summary']['encoder_layer_profile']) == 2


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


def test_declared_seed2_tokenizer_readout_matches_full_forward(tmp_path, monkeypatch):
    from paclock_bench.models import build
    from paclock_bench.probing.real_eeg_models import FrozenReadout, sha256

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = torch.nn.Linear(10, 8)
            self.encoder = torch.nn.Module()
            self.encoder.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

        def forward(self, x):
            x = self.frontend(x)
            for block in self.encoder.blocks:
                x = block(x)
            return x

    model = Model()
    checkpoint = tmp_path / 'seed2.pt'
    torch.save({'config': {'seed': 2, 'model': 'paclock', 'dataset': 'tuev',
                          'model_kwargs': {'tokenizer_mode': 'pac_interaction'}},
                'model': model.state_dict()}, checkpoint)
    monkeypatch.setattr(build, 'build_model', lambda cfg, shape: Model())
    job = dict(kind='cfft', dataset='tuev', seed=2, checkpoint=str(checkpoint),
               checkpoint_sha256=sha256(checkpoint))
    full = FrozenReadout(job, (4, 10), 'cpu')
    token = FrozenReadout({**job, 'readout_scope': 'tokenizer'}, (4, 10), 'cpu')
    x = torch.randn(3, 4, 10)
    np.testing.assert_array_equal(full(x)['tokenizer'], token(x)['tokenizer'])
    assert token.endpoints == ['tokenizer']
    assert len(full.endpoints) == 3
    assert not any(p.requires_grad for p in token.model.parameters())


def test_readout_rejects_checkpoint_seed_mismatch(tmp_path, monkeypatch):
    import pytest
    from paclock_bench.probing.real_eeg_models import FrozenReadout, sha256
    checkpoint = tmp_path / 'seed2.pt'
    torch.save({'config': {'seed': 2, 'model': 'paclock'}, 'model': {}}, checkpoint)
    with pytest.raises(ValueError, match='declared job seed'):
        FrozenReadout(dict(kind='cfft', seed=0, checkpoint=str(checkpoint),
                           checkpoint_sha256=sha256(checkpoint)), (4, 10), 'cpu')
