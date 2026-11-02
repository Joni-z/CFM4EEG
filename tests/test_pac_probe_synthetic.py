import io

import numpy as np
import pytest

from paclock_bench.probing.synthetic import curve_from_coefficients, generate_dataset


def test_analytic_bin_integral_matches_quadrature_and_preserves_depth():
    coefficients = np.array([[0.3, -0.2, 0.1, 0.25], [0.5, 0., 0., 0.]])
    bins, resolution = 24, 10000
    phase = -np.pi + (np.arange(bins * resolution) + 0.5) * 2 * np.pi / (bins * resolution)
    basis = np.array([np.cos(phase), np.sin(phase), np.cos(2 * phase), np.sin(2 * phase)])
    quadrature = (coefficients @ basis).reshape(2, bins, resolution).mean(-1)
    analytic = curve_from_coefficients(coefficients, bins)
    np.testing.assert_allclose(analytic, quadrature, atol=1e-10)
    np.testing.assert_allclose(analytic.mean(-1), 0, atol=1e-15)
    np.testing.assert_allclose(curve_from_coefficients(2 * coefficients), 2 * analytic)


def test_families_splits_determinism_and_safe_roundtrip():
    data = generate_dataset(13, 9, 8, seed=42)
    again = generate_dataset(13, 9, 8, seed=42)
    assert data["signals"].shape == (30, 1, 1000)
    assert data["signals"].dtype == np.float32
    for key, value in data.items():
        assert isinstance(value, np.ndarray)
        assert value.dtype != object
        np.testing.assert_array_equal(value, again[key])
    for split in ("train", "val", "test"):
        labels, counts = np.unique(data["family"][data["split"] == split], return_counts=True)
        assert set(labels) == {"h1", "h2", "mixed", "null"}
        assert np.ptp(counts) <= 1
    assert len(np.unique(data["sample_id"])) == len(data["signals"])
    assert len(np.unique(data["group_id"])) == len(data["signals"])
    assert len({x.tobytes() for x in data["signals"]}) == len(data["signals"])
    changed_train = generate_dataset(16, 9, 8, seed=42)
    np.testing.assert_array_equal(data["signals"][data["split"] != "train"],
                                  changed_train["signals"][changed_train["split"] != "train"])
    # Family reassignment must not change draws of gain, SNR, or phase nuisance.
    for key in ("snr_db", "low_amplitude", "expected_high_power", "source_hz", "source_phase0"):
        np.testing.assert_array_equal(data[key][:13], changed_train[key][:13])
    assert not np.array_equal(data["signals"], generate_dataset(13, 9, 8, seed=43)["signals"])
    buffer = io.BytesIO()
    np.savez(buffer, **data)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        for key, value in data.items():
            np.testing.assert_array_equal(saved[key], value)


def test_null_population_truth_and_independent_nonstationary_phase():
    data = generate_dataset(64, 0, 0, seed=7)
    null = ~data["coupled"]
    assert null.sum() == 16
    assert np.all(data["coefficients"][null] == 0)
    assert np.all(data["curves"][null] == 0)
    assert np.all(np.linalg.norm(data["envelope_coefficients"][null], axis=-1) > 0.25)
    for key in ("hz", "phase0", "drift_hz", "drift_phase0"):
        assert np.all(data[f"source_{key}"][null] != data[f"envelope_{key}"][null])
        np.testing.assert_array_equal(data[f"source_{key}"][~null], data[f"envelope_{key}"][~null])
    assert float(data["phase_drift"]) > 0
    assert np.all(data["source_drift_hz"] > 0)
    for family, forbidden in (("h1", slice(2, 4)), ("h2", slice(0, 2))):
        assert np.all(data["coefficients"][data["family"] == family, forbidden] == 0)
    # Uniform independent latent phase has zero conditional modulation, even
    # though one short realized signal need not produce a flat fitted curve.
    phase = 2 * np.pi * (np.arange(4096) + 0.5) / 4096
    basis = np.array([np.cos(phase), np.sin(phase), np.cos(2 * phase), np.sin(2 * phase)])
    np.testing.assert_allclose((data["envelope_coefficients"][null] @ basis).mean(-1), 0, atol=1e-15)


def test_positive_envelopes_sideband_guards_and_actual_power_diagnostics():
    data = generate_dataset(1024, 0, 0, seed=12, sample_rate=100)
    assert np.all(data["envelope_lower_bound"] > 0.29)
    assert np.all(data["minimum_envelope"] >= data["envelope_lower_bound"])
    assert np.all(data["guarded_sideband_low"] > data["source_guard_hz"])
    assert np.all(data["guarded_sideband_high"] < data["sample_rate"] / 2)
    np.testing.assert_allclose(10 * np.log10(data["clean_power"] / data["noise_power"]),
                               data["snr_db"], atol=1e-12)
    for family in ("h1", "h2", "mixed", "null"):
        mask = data["family"] == family
        for component in ("low", "high"):
            ratio = data[f"{component}_power"][mask].mean() / data[f"expected_{component}_power"][mask].mean()
            assert abs(ratio - 1) < 0.02
    # Match expectations mathematically and check actual finite-window powers;
    # this does not establish indistinguishable spectra or a fair full benchmark.
    coefficients = data["envelope_coefficients"]
    expected = data["high_gain"]**2 * (1 + np.sum(coefficients**2, axis=1) / 2) / 2
    np.testing.assert_allclose(expected, data["expected_high_power"], atol=1e-15)
    # A tapered FFT checks weak tails empirically; the guards are not claimed
    # to be exact support for frequency-modulated finite-duration signals.
    t = np.arange(data["signals"].shape[-1]) / float(data["sample_rate"])
    phase = 2 * np.pi * data["envelope_hz"][:, None] * t + data["envelope_phase0"][:, None]
    phase += float(data["phase_drift"]) * np.sin(
        2 * np.pi * data["envelope_drift_hz"][:, None] * t + data["envelope_drift_phase0"][:, None])
    envelope = 1 + sum(coefficients[:, k, None] * wave for k, wave in enumerate(
        (np.cos(phase), np.sin(phase), np.cos(2 * phase), np.sin(2 * phase))))
    high = data["high_gain"][:, None] * envelope * np.cos(
        2 * np.pi * data["target_hz"][:, None] * t + data["carrier_phase0"][:, None])
    power = abs(np.fft.rfft(high * np.hanning(len(t)), axis=-1))**2
    hz = np.fft.rfftfreq(len(t), 1 / float(data["sample_rate"]))
    outside = (hz[None] < data["guarded_sideband_low"][:, None]) | (
        hz[None] > data["guarded_sideband_high"][:, None])
    assert np.max((power * outside).sum(-1) / power.sum(-1)) < 1e-6


def test_empty_and_invalid_parameters():
    empty = generate_dataset(0, 0, 0)
    assert empty["signals"].shape == (0, 1, 1000)
    assert empty["curves"].shape == (0, 24)
    assert empty["coefficients"].shape == (0, 4)
    for kwargs in ({"n_train": -1}, {"n_val": 1.5}, {"sample_rate": 99},
                   {"seconds": 0}, {"seconds": np.nan}, {"n_bins": 4}, {"seed": -1}):
        with pytest.raises(ValueError):
            generate_dataset(**kwargs)
    with pytest.raises(ValueError):
        curve_from_coefficients([1, 2])
