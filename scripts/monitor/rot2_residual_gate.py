"""Local-only validation screen; reports eligibility, never submits jobs."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def review(runs, seed=0):
    records = {}
    issues = []
    for dataset in ('tuev', 'sleepedf'):
        for arm in ('control', 'band'):
            name = f'{dataset}-rot2_residual_{arm}'
            path = runs / name / f'seed{seed}' / 'result.json'
            if not path.exists():
                issues.append(f'missing: {path}')
                continue
            data = path.read_bytes()
            r = json.loads(data)
            cfg = r['config']
            selected = r.get('selected_validation', {})
            metrics = selected.get('metrics', {})
            checks = {
                'identity': r.get('name') == name and r.get('dataset') == dataset and r.get('seed') == seed,
                'complete': r.get('epochs_run') == 20 and r.get('stopped_by') == 'epochs' and cfg.get('epochs') == 20,
                'selection': selected.get('selection_metric') == 'cohen_kappa' and (cfg.get('select_metric') or r.get('primary_metric')) == 'cohen_kappa',
                'manifest': bool(r.get('data_manifest_created')),
                'counts': bool(r.get('class_counts', {}).get('val')),
                'finite_metrics': all(isinstance(metrics.get(k), (float, int)) and math.isfinite(metrics[k]) for k in ('cohen_kappa', 'balanced_acc')),
                'correct_arm': cfg.get('model_kwargs', {}).get('local_residual') is (arm == 'band'),
                'no_subset': not cfg.get('train_subsample') and not cfg.get('val_subsample'),
            }
            if checks['finite_metrics']:
                checks['selected_matches_best'] = math.isclose(metrics['cohen_kappa'], r['best_val'], abs_tol=1e-10, rel_tol=0)
            issues.extend(f'{name}: {k}' for k, ok in checks.items() if not ok)
            records[(dataset, arm)] = (r, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), metrics=metrics))
    if issues:
        return dict(state='not_ready', issues=issues, seed=seed)
    deltas = {}
    for dataset in ('tuev', 'sleepedf'):
        a, _ = records[(dataset, 'control')]
        b, _ = records[(dataset, 'band')]
        ca, cb = copy.deepcopy(a['config']), copy.deepcopy(b['config'])
        for c in (ca, cb):
            c.pop('name', None)
            c['model_kwargs'].pop('local_residual', None)
        if ca != cb:
            issues.append(f'{dataset}: paired recipes differ beyond run name and residual flag')
        if any(a.get(k) != b.get(k) for k in ('data_manifest_created', 'class_counts')):
            issues.append(f'{dataset}: paired cohorts differ')
        ma, mb = a['selected_validation']['metrics'], b['selected_validation']['metrics']
        deltas[dataset] = {k: mb[k] - ma[k] for k in ('cohen_kappa', 'balanced_acc')}
    if issues:
        return dict(state='not_ready', issues=issues, seed=seed)
    conditions = {
        'tuev_kappa_non_degradation': deltas['tuev']['cohen_kappa'] >= 0,
        'tuev_balanced_acc_non_degradation': deltas['tuev']['balanced_acc'] >= 0,
        'sleepedf_kappa_gain_at_least_001': deltas['sleepedf']['cohen_kappa'] >= .01,
        'sleepedf_balanced_acc_non_degradation': deltas['sleepedf']['balanced_acc'] >= 0,
    }
    return dict(state='validation_screen_passed' if all(conditions.values()) else 'validation_screen_failed',
                seed=seed, conditions=conditions, deltas=deltas,
                evidence=[v[1] for v in records.values()],
                limitations=['Not a final candidate decision. No automatic submissions.',
                             'Passing requires class-wise TUEV review and independent seed replication before promotion.',
                             'Source/runtime and admission receipts still require review; config equality alone is insufficient.',
                             'A failed seed does not eliminate the entire model family. Test metrics are not read.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.runs)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['state'])
