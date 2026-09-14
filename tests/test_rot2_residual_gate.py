import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('gate', Path(__file__).resolve().parents[1] / 'scripts/monitor/rot2_residual_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = {}
        for ds in ('tuev', 'sleepedf'):
            for arm in ('control', 'band'):
                name = f'{ds}-rot2_residual_{arm}'
                kappa = .5 if arm == 'control' else .52
                self.rows[(ds, arm)] = dict(name=name, dataset=ds, seed=0, epochs_run=20, stopped_by='epochs',
                    primary_metric='cohen_kappa', best_val=kappa, data_manifest_created='same', class_counts={'val': [4, 5]},
                    selected_validation={'selection_metric': 'cohen_kappa', 'metrics': {'cohen_kappa': kappa, 'balanced_acc': .6}},
                    config={'name': name, 'epochs': 20, 'model_kwargs': {'local_residual': arm == 'band'}})

    def evaluate(self):
        for (ds, arm), r in self.rows.items():
            p = self.root / r['name'] / 'seed0' / 'result.json'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
        return gate.review(self.root)

    def test_complete_validation_pass_does_not_need_test_scores(self):
        self.assertEqual(self.evaluate()['state'], 'validation_screen_passed')

    def test_reject_incomplete_mismatched_or_missing_evidence(self):
        for mutation in ('cap', 'manifest', 'metrics', 'recipe', 'best'):
            with self.subTest(mutation=mutation):
                saved = copy.deepcopy(self.rows)
                r = self.rows[('tuev', 'band')]
                if mutation == 'cap': r['stopped_by'] = 'max_hours'
                if mutation == 'manifest': r['data_manifest_created'] = 'other'
                if mutation == 'metrics': r['selected_validation']['metrics'].pop('balanced_acc')
                if mutation == 'recipe': r['config']['lr'] = .02
                if mutation == 'best': r['best_val'] = .9
                self.assertEqual(self.evaluate()['state'], 'not_ready')
                self.rows = saved

    def test_balanced_accuracy_regression_fails_despite_kappa_gain(self):
        self.rows[('tuev', 'band')]['selected_validation']['metrics']['balanced_acc'] = .59
        self.assertEqual(self.evaluate()['state'], 'validation_screen_failed')

    def test_missing_results_not_ready(self):
        self.assertEqual(gate.review(self.root)['state'], 'not_ready')


if __name__ == '__main__':
    unittest.main()
