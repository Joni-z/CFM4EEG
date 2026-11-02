"""Audit finite-window PAC recovery against analytic population ground truth.

The independent signal reference uses scipy SOS filtering/Hilbert and harmonic
least squares. Both analysis bands are fixed across records. The CFC operator
is unmodified, scored at its predeclared 4 Hz / 32 Hz cell. Neither estimator
receives per-record latent frequencies, phases, family labels or coefficients.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
import torch

from paclock_bench.probing.synthetic import curve_from_coefficients
from paclock_bench.probing.scoring import score_curves
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator


def signal_reference(signals, fs=200., source_band=(2.5,5.5), target_band=(20.,45.)):
    """Fixed broad carrier band includes H2 sidebands for this fixture.

    This is a signal-derived reference, not a gold-standard physiological PAC
    estimator. Discard .75 s at both edges for the finite-window regression.
    """
    x = np.asarray(signals, dtype=np.float64)[:, 0]
    phase = np.angle(hilbert(sosfiltfilt(butter(4, source_band, fs=fs,
                        btype='bandpass', output='sos'), x, axis=-1), axis=-1))
    amplitude = np.abs(hilbert(sosfiltfilt(butter(4, target_band, fs=fs,
                        btype='bandpass', output='sos'), x, axis=-1), axis=-1))
    edge = round(.75*fs)
    phase, amplitude = phase[:, edge:-edge], amplitude[:, edge:-edge]
    coeff = []
    for p, a in zip(phase, amplitude):
        design = np.stack([np.ones_like(p), np.linspace(-1, 1, len(p)),
                           np.cos(p), np.sin(p), np.cos(2*p), np.sin(2*p)], -1)
        beta = np.linalg.lstsq(design, a, rcond=None)[0]
        coeff.append(beta[2:]/max(beta[0], 1e-12))
    return np.array(coeff)


def summarize(data, coefficients):
    curves = curve_from_coefficients(coefficients)
    result = dict(overall=score_curves(data['curves'], curves, data['coupled']), families={})
    for f in np.unique(data['family']):
        rows = data['family'] == f
        result['families'][str(f)] = score_curves(data['curves'][rows], curves[rows], data['coupled'][rows])
        result['families'][str(f)]['predicted_curve_rms_mean'] = float(np.sqrt(np.mean(curves[rows]**2, axis=1)).mean())
    return result, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with np.load(args.dataset, allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    n = min(args.limit, len(data['signals']))
    data = {k: v[:n] if v.ndim and len(v) == len(data['signals']) else v for k, v in data.items()}
    start = time.monotonic()
    reference = signal_reference(data['signals'])
    report = {'signal_reference': summarize(data, reference)[0]}
    operator = CFCOperator().eval()
    estimates, valid, support = [], [], []
    with torch.no_grad():
        for first in range(0, n, 32):
            out = operator(torch.from_numpy(data['signals'][first:first+32]))
            packet = out.relation_features(stabilized=False)
            estimates.append(packet['features'][:, 0, 0, 2, 2].flatten(-2).numpy())
            valid.append(packet['harmonic_valid'][:, 0, 0, 2, 2].numpy())
            support.append(packet['reliability'][:, 0, 0, 2, 2].numpy())
            if first % 128 == 0:
                print('operator', first, '/', n, round(time.monotonic()-start, 2), flush=True)
    estimate = np.concatenate(estimates)
    report['cfc_operator'] = summarize(data, estimate)[0]
    report['cfc_operator']['harmonic_coverage'] = np.concatenate(valid).mean(0).tolist()
    null = data['family'] == 'null'
    report['null_identifiability'] = dict(
        n=int(null.sum()),
        within_one_fourier_bin_fraction=float((np.abs(data['source_hz'][null]-data['envelope_hz'][null]) < 1/float(data['seconds'])).mean()),
        frequency_difference_hz=np.quantile(np.abs(data['source_hz'][null]-data['envelope_hz'][null]), [.1,.5,.9]).tolist(),
        definition='Independent latent oscillators may remain nearly phase-locked in one finite record; this is not estimator false positivity by itself.')
    report.update(n=n, seconds=float(data['seconds']), elapsed_seconds=time.monotonic()-start,
                  analysis_bands=[2.5,5.5,20,45], operator_cell=[4,32],
                  scope='legacy fixture diagnostic; no tokenizer model selection')
    np.savez_compressed(args.output/'predictions.npz', sample_id=data['sample_id'],
                        target=data['curves'], reference_coefficients=reference,
                        operator_coefficients=estimate, harmonic_valid=np.concatenate(valid),
                        reliability=np.concatenate(support))
    (args.output/'scores.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
