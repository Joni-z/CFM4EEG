"""Frozen legacy CfFT rotation checks; no tokenizer or encoder optimization.

Reuses the analytic curve fixture, spectrum-matched pairing generator, strict
checkpoint loader, and existing train/validation-selected kernel ridge probes.
Identity and wrong-record arms are interventions on the SAME fitted frontend.
They are not independently retrained clinical ablations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import torch
from threadpoolctl import threadpool_limits

from paclock_bench.models.paclock.frontend.analytic import hilbert, phase_amplitude
from paclock_bench.models.paclock.frontend.triaxial import patch_pac_vector
from paclock_bench.probing.features import load_project_tokenizer, normalize_project_signals
from paclock_bench.probing.synthetic import generate_dataset, curve_from_coefficients
from paclock_bench.probing.scoring import score_curves
from scripts.experiments.pac_candidate_search import ALPHAS, probe
from scripts.experiments.pac_singlepath_search import evaluate
from scripts.experiments.pac_tokenizer_learning import cross_pair_dataset

ARMS = ('measured', 'identity', 'wrong_record')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pairing_data(counts, seed):
    parts = {key: [] for key in ('signals', 'labels', 'split', 'group', 'sample_id')}
    errors = []
    for index, (split, count) in enumerate(zip(('train', 'val', 'test'), counts)):
        if count % 4 or count < 8:
            raise ValueError('pairing split counts must be multiples of four and >=8')
        signals, labels, error = cross_pair_dataset(seed + index * 10000, count // 2, (4.3, 5.0))
        # The reused generator returns physical-amplitude signals already scaled by20.
        parts['signals'].append(signals.numpy())
        parts['labels'].append(labels.numpy())
        parts['split'].append(np.repeat(split, count))
        parts['group'].append(np.repeat(np.arange(count // 4) + index * 100000, 4))
        parts['sample_id'].append(np.arange(count) + index * 1000000)
        errors.append(error)
    data = {key: np.concatenate(values) for key, values in parts.items()}
    data['curves'] = data['labels'][:, None].astype(float)
    data['coupled'] = np.ones(len(data['labels']), dtype=bool)
    data['generator_spectrum_error'] = np.array(errors)
    return data


def wrong_donors(split, group, seed):
    """Label-blind random donors: same split, different group, no batch dependence."""
    rng = np.random.default_rng(seed)
    donor = np.empty(len(split), dtype=np.int64)
    for name in np.unique(split):
        ids = np.flatnonzero(split == name)
        groups = np.unique(group[ids])
        if len(groups) < 2:
            raise ValueError('need two independent groups per split')
        order = rng.permutation(groups)
        next_group = dict(zip(order, np.roll(order, 1)))
        for current in groups:
            target = ids[group[ids] == current]
            source = ids[group[ids] == next_group[current]]
            if len(source) != len(target):
                raise ValueError('donor groups must have the same cardinality')
            donor[target] = rng.permutation(source)
    assert np.all(split[donor] == split)
    assert np.all(group[donor] != group)
    return donor


@torch.no_grad()
def extract(frontend, data, output, normalization, gain, seed, batch_size):
    n, channels, length = data['signals'].shape
    patches = length // frontend.patch_len
    nb = frontend.n_bands
    group = data.get('group', np.arange(n))
    donor = wrong_donors(data['split'], group, seed)
    np.save(output / 'donor_indices.npy', donor)
    moments = np.lib.format.open_memmap(output / 'moments.npy', mode='w+', dtype=np.complex64,
                                       shape=(n, channels, patches, nb, nb))
    # First pass caches each sample's own relation; donors never cross split/group.
    for start in range(0, n, batch_size):
        x = normalize_project_signals(torch.from_numpy(data['signals'][start:start+batch_size]) * gain, normalization)
        b = len(x)
        filtered = frontend.sinc(x.reshape(b * channels, 1, length)).reshape(b, channels, nb, length)
        phase, amplitude = phase_amplitude(hilbert(filtered))
        moments[start:start+b] = patch_pac_vector(phase, amplitude, patches, frontend.normalize).numpy()
    moments.flush()
    arrays = {}
    norm_error, parity_error = 0., 0.
    diagnostic = []
    for start in range(0, n, batch_size):
        x = normalize_project_signals(torch.from_numpy(data['signals'][start:start+batch_size]) * gain, normalization)
        b = len(x)
        filtered = frontend.sinc(x.reshape(b * channels, 1, length)).reshape(b, channels, nb, length)
        phase, amplitude = phase_amplitude(hilbert(filtered))
        own = torch.from_numpy(np.array(moments[start:start+b]))
        wrong = torch.from_numpy(np.array(moments[donor[start:start+b]]))
        tokens = {}
        try:
            for arm in ARMS:
                frontend.pac_token_mode = 'identity' if arm == 'identity' else 'measured'
                value = frontend._interaction_tokens(phase, amplitude, wrong if arm == 'wrong_record' else own,
                                                     filtered=filtered)
                tokens[arm] = value.numpy()
                if arm not in arrays:
                    arrays[arm] = np.lib.format.open_memmap(output / (arm + '.npy'), mode='w+',
                        dtype=np.float32, shape=(n, value[0].numel()))
                arrays[arm][start:start+b] = value.numpy().reshape(b, -1)
        finally:
            frontend.pac_token_mode = 'measured'
        if start == 0:
            reference = frontend(x)[0].numpy()
            parity_error = float(np.max(np.abs(reference - tokens['measured'])))
            np.testing.assert_allclose(reference, tokens['measured'], rtol=1e-5, atol=1e-6)
        reference_norm = np.square(tokens['identity'].reshape(b, -1, 2)).sum(-1)
        for arm in ('measured', 'wrong_record'):
            value_norm = np.square(tokens[arm].reshape(b, -1, 2)).sum(-1)
            norm_error = max(norm_error, float(np.max(abs(value_norm-reference_norm)) /
                                             max(float(np.max(reference_norm)), 1e-12)))
        if 'source_hz' in data:
            # Known frequencies choose diagnostic filter indices ONLY; never token inputs.
            # Report the unchanged local-moment path pooled over time. This is not a
            # calibrated estimator or a claim that 0.25s spans a physiological cycle.
            kernel = frontend.sinc.filters().squeeze(1).double()
            t = torch.arange(kernel.shape[-1], dtype=torch.float64) / 200
            gains = []
            for key in ('source_hz', 'target_hz'):
                f = torch.from_numpy(data[key][start:start+b]).double()
                arg = 2 * torch.pi * f[:, None] * t
                gains.append(((arg.cos() @ kernel.T)**2 + (arg.sin() @ kernel.T)**2).argmax(-1))
            source, target = gains
            for k in range(b):
                i, j = int(source[k]), int(target[k])
                z = own[k, 0, :, i, j].mean() * (100. if frontend.normalize else 1.)
                coeff = 2 * z / amplitude[k, 0, j].mean().clamp_min(1e-12)
                diagnostic.append([float(coeff.real), float(coeff.imag), 0., 0., i, j, int(i < j)])
        if start % 256 == 0:
            print('tokens', output.name, min(start+b, n), '/', n, flush=True)
    for value in arrays.values():
        value.flush()
    if diagnostic:
        np.save(output / 'native_h1_diagnostic.npy', np.asarray(diagnostic))
    write_json(output / 'extraction.json', {'normalization': normalization, 'input_gain': gain,
        'native_features': arrays['measured'].shape[1], 'complete_native_tokens': True,
        'frontend_parity_max_abs': parity_error, 'pair_norm_max_relative_error': norm_error,
        'donor_seed': seed, 'donor_same_group_count': int(np.sum(group[donor] == group)),
        'intervention': 'same frozen weights; identity/wrong relation, no retraining',
        'native_h1_diagnostic': '2*pooled original local Z / mean target envelope; known-frequency filter matching is diagnostic only'})
    return arrays


def pairing_scores(features, data, output):
    test = data['split'] == 'test'
    rows, predictions = [], {}
    for degree in (1, 2):
        score, pred = probe(features, data, degree)
        value = pred[:, 0]
        score.update(accuracy=float(accuracy_score(data['labels'][test], value >= .5)),
                     balanced_acc=float(balanced_accuracy_score(data['labels'][test], value >= .5)),
                     auroc=float(roc_auc_score(data['labels'][test], value)))
        rows.append(score)
        predictions[f'degree{degree}'] = value
    np.savez_compressed(output, **predictions, labels=data['labels'][test],
                        group=data['group'][test], sample_id=data['sample_id'][test])
    return {'linear_primary': rows[0], 'secondary_validation_selected': min(rows, key=lambda r:r['validation_mse']),
            'readouts': rows, 'fixed_decision_threshold': .5}


def paired_accuracy_interval(a, b, labels, group, seed=1105, resamples=2000):
    difference = ((a >= .5) == labels).astype(float) - ((b >= .5) == labels).astype(float)
    groups = np.unique(group)
    values = np.array([difference[group == g].mean() for g in groups])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    return {'difference': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'groups': len(groups), 'bootstrap_seed': seed, 'resamples': resamples}


def pairing_power_controls(data, normalization, output):
    """Check shortcuts both before and after the checkpoint's input normalization."""
    raw = data['signals']
    normalized = normalize_project_signals(torch.from_numpy(raw.copy()), normalization).numpy()
    frequency = np.fft.rfftfreq(raw.shape[-1], 1/200)
    results = {}
    for name, signal in [('physical', raw), ('normalized', normalized)]:
        spectrum = abs(np.fft.rfft(signal[:, 0].astype(float)))**2
        powers = np.stack([spectrum[:, frequency <= 12].sum(1),
                           spectrum[:, frequency >= 16].sum(1)], axis=1)
        results[name] = pairing_scores(powers, data, output/(name+'-power-predictions.npz'))
    write_json(output/'power-controls.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--normalization', choices=('div100', 'q95'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--counts', type=int, nargs=3, default=[1024,256,512])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    started = time.monotonic()
    model, metadata = load_project_tokenizer(args.config, args.checkpoint, (16,1000),
                                              normalization=args.normalization)
    frontend = model.frontend
    cfg = metadata['training_config']['model_kwargs']
    assert cfg['n_bands'] == 16 and cfg['d_model'] == 128 and cfg['depth'] == 6
    assert cfg['freq_mixer'] == 'attention' and not cfg.get('space_over_bands', False)
    assert cfg['interaction_mode'] == 'rotation' and cfg['waveform_modulation'] == 'quadrature'
    assert cfg.get('pac_patch_len', 50) == cfg['patch_len'] == 50
    assert frontend.pac_token_mode == 'measured'
    write_json(args.output/'checkpoint.json', metadata)
    source_paths = [Path(__file__), Path(__file__).with_name('pac_singlepath_search.py'),
        Path(__file__).with_name('pac_candidate_search.py'), Path(__file__).with_name('pac_tokenizer_learning.py'),
        Path(__file__).resolve().parents[2]/'paclock_bench/probing/synthetic.py']
    write_json(args.output/'protocol.json', {'seed':args.seed,'counts':args.counts,'arms':ARMS,
        'readouts':{'primary_degree':1,'secondary_degrees':[1,2],'alphas':ALPHAS,'selection':'validation MSE'},
        'input':'200Hz,5s,one physical synthetic electrode; checkpoint restored with original16 electrodes',
        'scope':'frozen clinical frontend; simulated PAC/content recovery and selective pairing',
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}})
    datasets = {'curve': generate_dataset(*args.counts,seed=args.seed),
                'pairing': pairing_data(args.counts,args.seed+100000)}
    report = {}
    for name, data in datasets.items():
        path = args.output/name
        path.mkdir()
        np.savez_compressed(path/'data.npz', **data)
        arrays = extract(frontend,data,path,args.normalization,20. if name=='curve' else 1.,
                         args.seed+200000,args.batch_size)
        report[name] = {}
        for arm, feature in arrays.items():
            if name=='curve':
                metrics, predictions = evaluate(feature,data)
                np.savez_compressed(path/(arm+'-predictions.npz'), **predictions,
                    sample_id=data['sample_id'][data['split']=='test'])
                report[name][arm] = metrics
                brief = {target: round(row['readouts'][0]['test']['curve_r2'],4) for target,row in metrics.items()}
            else:
                metrics = pairing_scores(feature,data,path/(arm+'-predictions.npz'))
                report[name][arm] = metrics
                brief = {k:metrics['linear_primary'][k] for k in ('accuracy','auroc')}
            write_json(path/(arm+'-scores.json'),metrics)
            print(name,arm,'linear primary',brief,'seconds',round(time.monotonic()-started,1),flush=True)
        if name=='curve':
            diagnostic = np.load(path/'native_h1_diagnostic.npy')
            test = data['split']=='test'
            pred = curve_from_coefficients(diagnostic[test,:4])
            h1 = np.array(data['coefficients']);h1[:,2:]=0
            h1_curves = curve_from_coefficients(h1)
            # H1-only population targets are stated separately from the full curve.
            h1_nonzero = np.linalg.norm(h1[test],axis=1)>0
            report[name]['native_moment_diagnostic'] = {
                'h1_population_curve':score_curves(h1_curves[test],pred,h1_nonzero),
                'full_population_curve':score_curves(data['curves'][test],pred,data['coupled'][test]),
                'eligible_selected_pair_fraction':float(diagnostic[test,6].mean())}
            write_json(path/'native-moment-scores.json',report[name]['native_moment_diagnostic'])
        else:
            report[name]['power_controls'] = pairing_power_controls(data,args.normalization,path)
            saved = {arm:np.load(path/(arm+'-predictions.npz')) for arm in ARMS}
            report[name]['paired_intervals'] = {other:paired_accuracy_interval(saved['measured']['degree1'],
                saved[other]['degree1'],saved['measured']['labels'],saved['measured']['group'])
                for other in ('identity',)}
            # Wrong-record donor cycles connect different groups. An ordinary
            # group bootstrap would incorrectly treat those groups as independent.
            a=saved['measured'];b=saved['wrong_record']
            report[name]['wrong_record_difference'] = {
                'difference':float(np.mean((a['degree1']>=.5)==a['labels'])-
                                   np.mean((b['degree1']>=.5)==b['labels'])),
                'ci95':None,'reason':'cross-group donor dependence; descriptive difference only'}
            write_json(path/'paired-intervals.json',report[name]['paired_intervals'])
    report['seconds'] = time.monotonic()-started
    write_json(args.output/'scores.json',report)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
