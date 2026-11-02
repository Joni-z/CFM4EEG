"""Small analytic PAC correctness fixture, not a validated physiological benchmark.

Adapts the harmonic envelope, low + envelope * carrier construction and colored
noise in scripts/experiments/cfc_operator_capabilities.py:61-102 (especially
78, 94-98), also cfc_operator_benchmark.py:26-61. Those generators are untouched.
Phase-conditioned amplitude follows Tort et al. (2010),
https://pmc.ncbi.nlm.nih.gov/articles/PMC2941206/; interpretational limits follow
Aru et al. (2015), https://pubmed.ncbi.nlm.nih.gov/25212583/.

Our adaptations are analytic bin integrals, independent split RNGs, equal family
quotas, smooth phase drift, an independent latent null oscillator, and expected
carrier-power normalization. Source phase is that of cos(phi), including its
random initial phase and drift, rather than time zero. The target is the bin
average of E[A | phi] / E[A] - 1 under uniform population phase, preserving
modulation depth; no fitted estimate is used as truth. Null phase has independent
initial phase, frequency, and drift with the same marginal law as source phase.
Its population target is zero;
finite windows can nevertheless exhibit accidental dependence.

The smooth frequency modulation has weak infinite sideband tails. Reported
guarded bands include six drift sidebands around H2, not strict spectral support.
Power matching holds in expectation; actual component powers are also returned.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


FAMILIES = np.array(["h1", "h2", "mixed", "null"])
PHASE_DRIFT = 0.5


def curve_from_coefficients(coeffs: np.ndarray, n_bins: int = 24) -> np.ndarray:
    """Exact bin means on [-pi, pi), for [..., cos1/sin1/cos2/sin2].

    Output is centered relative amplitude, so flat/null is zero. Multiplication
    by sinc(h / n_bins) integrates each harmonic instead of sampling its center.
    """
    if not isinstance(n_bins, Integral) or n_bins < 5:
        raise ValueError("n_bins must be an integer >= 5 to resolve H2")
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim < 1 or coeffs.shape[-1] != 4 or not np.isfinite(coeffs).all():
        raise ValueError("coeffs must be finite with final dimension 4")
    centers = -np.pi + (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins
    harmonics = np.array([1, 1, 2, 2])
    basis = np.stack([np.cos(centers), np.sin(centers),
                      np.cos(2 * centers), np.sin(2 * centers)])
    return coeffs @ (basis * np.sinc(harmonics / n_bins)[:, None])


def _phase(t: np.ndarray, hz: float, phase0: float, drift_hz: float,
           drift_phase0: float) -> np.ndarray:
    return 2 * np.pi * hz * t + phase0 + PHASE_DRIFT * np.sin(
        2 * np.pi * drift_hz * t + drift_phase0)


def _record(rng: np.random.Generator, family: str, t: np.ndarray,
            sample_rate: float) -> dict:
    source_hz, independent_hz = rng.uniform(3.2, 4.3, 2)
    source_phase0, independent_phase0, carrier_phase0 = rng.uniform(-np.pi, np.pi, 3)
    source_drift_hz, independent_drift_hz = rng.uniform(0.3, 0.6, 2)
    source_drift_phase0, independent_drift_phase0 = rng.uniform(-np.pi, np.pi, 2)
    target_hz = rng.uniform(30.2, 33.8)
    low_amplitude, high_amplitude = rng.uniform(0.8, 1.2, 2)
    snr_db = rng.uniform(5, 20)
    radius, mixing = rng.uniform(0.25, 0.5), rng.uniform(0.35, np.pi / 2 - 0.35)
    theta1, theta2 = rng.uniform(-np.pi, np.pi, 2)
    null_shape = str(rng.choice(FAMILIES[:3]))
    shape = null_shape if family == "null" else family
    d1 = radius if shape == "h1" else radius * np.cos(mixing) if shape == "mixed" else 0.
    d2 = radius if shape == "h2" else radius * np.sin(mixing) if shape == "mixed" else 0.
    envelope_coefficients = np.array([d1 * np.cos(theta1), d1 * np.sin(theta1),
                                      d2 * np.cos(theta2), d2 * np.sin(theta2)])
    phi = _phase(t, source_hz, source_phase0, source_drift_hz, source_drift_phase0)
    null = family == "null"
    envelope_hz = independent_hz if null else source_hz
    envelope_phase0 = independent_phase0 if null else source_phase0
    envelope_drift_hz = independent_drift_hz if null else source_drift_hz
    envelope_drift_phase0 = independent_drift_phase0 if null else source_drift_phase0
    psi = _phase(t, envelope_hz, envelope_phase0, envelope_drift_hz, envelope_drift_phase0)
    envelope = 1 + d1 * np.cos(psi - theta1) + d2 * np.cos(2 * psi - theta2)
    # This scalar preserves E[A | phi] / E[A] - 1 while matching E[high**2].
    high_gain = high_amplitude / np.sqrt(1 + 0.5 * radius**2)
    low = low_amplitude * np.cos(phi)
    high = high_gain * envelope * np.cos(2 * np.pi * target_hz * t + carrier_phase0)
    clean = low + high
    noise = rng.normal(size=t.size)
    frequencies = np.fft.rfftfreq(t.size, 1 / sample_rate)
    noise = np.fft.irfft(np.fft.rfft(noise) / np.sqrt(1 + frequencies), n=t.size)
    noise -= noise.mean()
    clean_power = np.mean(clean**2)
    noise *= np.sqrt(clean_power / 10**(snr_db / 10)) / noise.std()
    return dict(signals=(clean + noise)[None].astype(np.float32),
                coefficients=np.zeros(4) if null else envelope_coefficients,
                envelope_coefficients=envelope_coefficients, envelope_family=shape,
                source_hz=source_hz, target_hz=target_hz, source_phase0=source_phase0,
                source_drift_hz=source_drift_hz, source_drift_phase0=source_drift_phase0,
                envelope_hz=envelope_hz, envelope_phase0=envelope_phase0,
                envelope_drift_hz=envelope_drift_hz, envelope_drift_phase0=envelope_drift_phase0,
                carrier_phase0=carrier_phase0, low_amplitude=low_amplitude, high_gain=high_gain,
                expected_low_power=low_amplitude**2 / 2,
                expected_high_power=high_amplitude**2 / 2,
                low_power=np.mean(low**2), high_power=np.mean(high**2),
                clean_power=clean_power, noise_power=np.mean(noise**2), snr_db=snr_db,
                minimum_envelope=envelope.min(),
                envelope_lower_bound=1 - d1 - d2,
                guarded_sideband_low=target_hz - 2 * envelope_hz - 6 * envelope_drift_hz,
                guarded_sideband_high=target_hz + 2 * envelope_hz + 6 * envelope_drift_hz,
                source_guard_hz=source_hz + 6 * source_drift_hz)


def generate_dataset(n_train: int = 128, n_val: int = 64, n_test: int = 64,
                     seed: int = 0, sample_rate: float = 200,
                     seconds: float = 5., n_bins: int = 24) -> dict[str, np.ndarray]:
    """Generate disjoint independent records, with shuffled near-equal families.

    All values are NumPy arrays, including scalar metadata; np.savez outputs
    reload with allow_pickle=False. IDs encode split and local record index.
    RNGs for other splits are unaffected by changing one split's sample count.
    Parameters such as noise level, gain, phase and frequency use the same
    distribution and draw order for every family. Null envelope shapes have the
    positive families' mixture distribution. This is not exact spectral matching.
    """
    counts = (n_train, n_val, n_test)
    if any(not isinstance(n, Integral) or n < 0 or n >= 2**32 for n in counts):
        raise ValueError("split counts must be nonnegative integers < 2**32")
    if not isinstance(seed, Integral) or not 0 <= seed <= np.iinfo(np.int64).max:
        raise ValueError("seed must be a nonnegative int64")
    if not np.isfinite(sample_rate) or sample_rate < 100:
        raise ValueError("sample_rate must be >= 100 Hz for guarded H2 sidebands")
    if not np.isfinite(seconds) or seconds <= 0 or round(seconds * sample_rate) < 16:
        raise ValueError("seconds must be positive and provide at least 16 samples")
    curve_from_coefficients(np.zeros(4), n_bins)  # Validate even an empty dataset.
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    records, families, splits, ids = [], [], [], []
    for split_index, (split, count) in enumerate(zip(("train", "val", "test"), counts)):
        assignment = np.resize(FAMILIES, count).copy()
        np.random.default_rng(np.random.SeedSequence([seed, split_index, 0])).shuffle(assignment)
        for index, family in enumerate(assignment):
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, 1, index]))
            records.append(_record(rng, str(family), t, sample_rate))
            families.append(family)
            splits.append(split)
            ids.append((split_index << 32) + index)
    # The prototype supplies shapes/dtypes even when all requested counts are zero.
    prototype = records[0] if records else _record(np.random.default_rng(seed), "h1", t, sample_rate)
    data = {key: np.stack([r[key] for r in records]) if records else
            np.empty((0,) + np.asarray(value).shape, dtype=np.asarray(value).dtype)
            for key, value in prototype.items()}
    data.update(curves=curve_from_coefficients(data["coefficients"], n_bins),
                family=np.asarray(families, dtype="U5"), split=np.asarray(splits, dtype="U5"),
                coupled=np.asarray(families, dtype="U5") != "null",
                group_id=np.asarray(ids, dtype=np.int64), sample_id=np.asarray(ids, dtype=np.int64),
                phase_bins=-np.pi + (np.arange(n_bins) + 0.5) * 2 * np.pi / n_bins,
                sample_rate=np.asarray(sample_rate), seconds=np.asarray(seconds), seed=np.asarray(seed),
                phase_drift=np.asarray(PHASE_DRIFT), format_version=np.asarray("analytic-pac-v1"),
                target_definition=np.asarray("E[A|phase-bin]/E[A]-1; analytic population bin mean"))
    return data
