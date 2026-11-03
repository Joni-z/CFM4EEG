"""Bounded frozen-token screen; no downstream training or metric changes.

The kernels/readouts reproduce pac_candidate_search.probe. Kernel decompositions
are reused across independent targets, each with its own validation selection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
import torch
from threadpoolctl import threadpool_limits

from paclock_bench.probing.features import linear_token_budget, load_project_tokenizer
from paclock_bench.probing.scoring import score_curves
from paclock_bench.probing.synthetic import curve_from_coefficients, generate_dataset
from scripts.experiments.pac_candidate_search import ALPHAS, cfft_sequence, dump
from scripts.experiments.pac_singlepath import SinglePathPAC


def content_targets(data):
    """Generator powers and fixed random waveform projections, never token inputs."""
    power = np.stack([data['expected_low_power'], data['expected_high_power']], axis=-1)
    rng = np.random.default_rng(7104)
    signal = data['signals'].reshape(len(power), -1)
    projection = rng.normal(size=(signal.shape[1], 32)) / np.sqrt(signal.shape[1])
    waveform = signal @ projection
    train = data['split'] == 'train'
    return {name: StandardScaler().fit(value[train]).transform(value)
            for name, value in [('power', power), ('waveform', waveform)]}


def evaluate(features, data):
    tr, va, te = (data['split'] == s for s in ('train', 'val', 'test'))
    x = StandardScaler().fit(features[tr]).transform(features).astype(np.float64)
    gram = (x @ x[tr].T) / features.shape[1]
    del x
    targets = dict(pac=data['curves'], **content_targets(data))
    results = {key: [] for key in targets}
    predictions = {}
    for degree in (1, 2):
        kernel = gram.copy() if degree == 1 else (1 + gram) ** 2
        col = kernel[tr].mean(0)
        kernel = kernel - kernel.mean(1, keepdims=True) - col[None] + col.mean()
        values, vectors = eigh(kernel[tr], check_finite=False)
        values = np.maximum(values, 0.)
        for name, target in targets.items():
            mean = target[tr].mean(0)
            projected = vectors.T @ (target[tr] - mean)
            best, grid = None, []
            for alpha in ALPHAS:
                weights = vectors @ (projected / (values[:, None] + alpha))
                predicted = kernel[va] @ weights + mean
                error = float(np.mean((predicted - target[va]) ** 2))
                grid.append(dict(alpha=alpha, mse=error))
                if best is None or error < best[0]:
                    best = error, alpha, weights, predicted
            predicted = kernel[te] @ best[2] + mean
            labels = data['coupled'] if name == 'pac' else np.ones(len(target), dtype=bool)
            row = dict(degree=degree, alpha=best[1], validation_mse=best[0], validation_grid=grid,
                       validation=score_curves(target[va], best[3], labels[va]),
                       test=score_curves(target[te], predicted, labels[te]))
            if name == 'pac':
                row['families'] = {family: score_curves(target[te][data['family'][te] == family],
                    predicted[data['family'][te] == family], labels[te][data['family'][te] == family])
                    for family in ('h1', 'h2', 'mixed', 'null')}
            results[name].append(row)
            predictions[f'{name}_degree{degree}'] = predicted
            predictions[f'{name}_target'] = target[te]
        del kernel
    return {name: dict(readouts=rows, selected=min(rows, key=lambda r: r['validation_mse']))
            for name, rows in results.items()}, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--train', type=int, default=1024)
    parser.add_argument('--val', type=int, default=256)
    parser.add_argument('--test', type=int, default=512)
    parser.add_argument('--variants', nargs='+', choices=SinglePathPAC.MODES, default=list(SinglePathPAC.MODES))
    parser.add_argument('--wrong', action='store_true')
    parser.add_argument('--diagnostic-cell', nargs=2, type=int, default=[2, 2],
                        help='fixed physical cell for operator-only scoring, never token input')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    if args.dataset:
        with np.load(args.dataset, allow_pickle=False) as archive:
            data = dict(archive)
    else:
        data = generate_dataset(args.train, args.val, args.test, seed=args.seed)
    assert int(data['seed']) == args.seed
    assert [int((data['split'] == s).sum()) for s in ('train', 'val', 'test')] == [args.train, args.val, args.test]
    np.savez_compressed(args.output / 'data.npz', **data)
    sources = [Path(__file__), Path(__file__).with_name('pac_singlepath.py')]
    protocol = dict(seed=args.seed, variants=args.variants, wrong=args.wrong, input_gain=.2,
        diagnostic_cell=args.diagnostic_cell,
        train=args.train, val=args.val, test=args.test, alphas=ALPHAS, degrees=[1, 2],
        selection='validation MSE independently for each target; no test-based readout choice',
        parameters='frozen original checkpoint; no new learned parameters',
        content='train-standardized true generating powers and 32 fixed waveform projections',
        scope='synthetic screen only; no clinical training',
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    dump(args.output / 'protocol.json', protocol)
    model, meta = load_project_tokenizer(args.reference / 'original-training.yaml',
        args.reference / 'original-best.pt', (16, 1000), normalization='div100')
    candidate = SinglePathPAC(model.frontend).eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    dump(args.output / 'checkpoint.json', meta)
    torch.save(dict(state_dict=candidate.state_dict(), modes=args.variants, protocol=protocol), args.output / 'prototype.pt')
    n = len(data['signals'])
    names = ['original', 'unrotated'] + args.variants
    if args.wrong:
        names += ['wrong_' + mode for mode in args.variants]
    arrays = {name: np.lib.format.open_memmap(args.output / (name + '.npy'), mode='w+',
                                           dtype=np.float32, shape=(n, 40960)) for name in names}
    coefficients, coverage = [], []
    with torch.inference_mode():
        for first in range(0, n, 32):
            x = torch.from_numpy(data['signals'][first:first + 32]) * .2
            p = candidate.prepare(x)
            values = dict(original=model.frontend(x)[0], unrotated=torch.view_as_real(p['content']).flatten(-2))
            values.update({mode: candidate.transform(p, mode) for mode in args.variants})
            for name, value in values.items():
                arrays[name][first:first + len(x)] = value.numpy().reshape(len(x), -1)
            coefficients.append(p['coefficient'].numpy())
            coverage.append(p['mapping'].numpy())
            if first % 256 == 0:
                print('extracted', first, '/', n, 'seconds', round(time.monotonic() - start, 1), flush=True)
    coefficient = np.concatenate(coefficients)
    np.save(args.output / 'coefficients.npy', coefficient)
    np.save(args.output / 'mapping.npy', coverage[0])
    if args.wrong:
        indices = np.arange(n)
        rng = np.random.default_rng(91104)
        for split in ('train', 'val', 'test'):
            ids = np.flatnonzero(data['split'] == split)
            indices[ids] = np.roll(ids, int(rng.integers(1, len(ids))))
        np.save(args.output / 'wrong_indices.npy', indices)
        with torch.inference_mode():
            for first in range(0, n, 32):
                x = torch.from_numpy(data['signals'][first:first + 32]) * .2
                p = candidate.prepare(x)
                wrong = torch.from_numpy(coefficient[indices[first:first + len(x)]])
                for mode in args.variants:
                    value = candidate.transform(p, mode, wrong)
                    arrays['wrong_' + mode][first:first + len(x)] = value.numpy().reshape(len(x), -1)
                if first % 512 == 0:
                    print('wrong-record extraction', first, '/', n, flush=True)
    for array in arrays.values():
        array.flush()
    te = data['split'] == 'test'
    # The known generator cell is used only for operator diagnosis, never in token construction.
    source_cell, target_cell = args.diagnostic_cell
    queried = coefficient[:, 0, 0, source_cell, target_cell]
    queried = np.stack([queried.real, queried.imag], axis=-1).reshape(n, 4)
    report = dict(operator=score_curves(data['curves'][te], curve_from_coefficients(queried[te]), data['coupled'][te]), rows={})
    dump(args.output / 'measurement.json', report['operator'])
    for name in names:
        report['rows'][name] = {}
        with np.errstate(divide='raise', invalid='raise'):
            delta = np.asarray(arrays[name], dtype=np.float64) - arrays['original']
            change = np.sqrt(np.mean(delta * delta, axis=1)) / np.sqrt(np.mean(np.square(arrays['original'], dtype=np.float64), axis=1)).clip(1e-12)
        dump(args.output / (name + '-change.json'), dict(mean_relative_change=float(change.mean()), max_relative_change=float(change.max())))
        del delta
        for budget in ('native', 'common'):
            feature = np.asarray(arrays[name])
            if budget == 'common':
                feature = linear_token_budget(cfft_sequence(feature), time_bins=9, width=64, seed=0)
            metrics, predictions = evaluate(feature, data)
            dump(args.output / (name + '-' + budget + '.json'), metrics)
            np.savez_compressed(args.output / (name + '-' + budget + '-predictions.npz'),
                                **predictions, sample_id=data['sample_id'][te], coupled=data['coupled'][te])
            report['rows'][name][budget] = metrics
            brief = {target: round(r['selected']['test']['curve_r2'], 4) for target, r in metrics.items()}
            print(name, budget, brief, 'seconds', round(time.monotonic() - start, 1), flush=True)
    report['elapsed_seconds'] = time.monotonic() - start
    dump(args.output / 'scores.json', report)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
