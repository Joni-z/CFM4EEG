"""Selection is fixed by validation metadata before any checkpoint inference."""
import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / 'scripts/rebuild/evaluate_biot_validation_best.py'
_SPEC = importlib.util.spec_from_file_location('biot_validation_evaluation', _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def records(scores):
    return [dict(epoch=i, step=(i + 1) * 10, metrics={'val_cohen': score})
            for i, score in enumerate(scores)]


@pytest.mark.parametrize('scores,epoch', [
    ([.8, .5, .4], 0),
    ([.4, .8, .8, .5], 1),
    ([.4, .5, .8], 2),
    ([-.4, -.2, -.3], 1),
])
def test_uses_first_validation_maximum(scores, epoch):
    rows = records(scores)
    # Deliberately misleading test values must not affect selection.
    for row in rows:
        row['test_cohen'] = 1.0 if row['epoch'] != epoch else 0.0
    assert _MODULE.select_validation_record(rows) is rows[epoch]


@pytest.mark.parametrize('rows', [
    [],
    records([.5, float('nan')]),
    records([.5, float('inf')]),
    records([1.01]),
    list(reversed(records([.4, .8]))),
    [dict(epoch=0, step=10, metrics={'val_cohen': .5}),
     dict(epoch=1, step=10, metrics={'val_cohen': .6})],
])
def test_rejects_incomplete_or_invalid_history(rows):
    with pytest.raises(ValueError):
        _MODULE.select_validation_record(rows)
