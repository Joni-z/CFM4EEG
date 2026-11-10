"""CHB selection must follow validation AUROC without renaming accuracy."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("labram_selection_entry", ROOT / "scripts/rebuild/run_labram_original.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

# Selection statements from the pinned author's main; the surrounding model
# updates are a tiny deterministic stand-in for the expensive EEG training.
# https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/run_class_finetuning.py
NATIVE_LOOP = '''
def main(args, ds_init):
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    for epoch, val_stats in enumerate(args.curve):
        args.model_weight = epoch + 10
        test_stats = {"accuracy": args.test_accuracy[epoch]}
        if max_accuracy < val_stats["accuracy"]:
            max_accuracy = val_stats["accuracy"]
            args.saved = (epoch, args.model_weight)
            max_accuracy_test = test_stats["accuracy"]
        print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
'''


def native_module(tmp_path):
    path = tmp_path / "native_selection.py"
    path.write_text(NATIVE_LOOP)
    spec = importlib.util.spec_from_file_location("native_selection_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auc_checkpoint_beats_accuracy_and_test_maxima_and_keeps_first_tie(tmp_path):
    native = native_module(tmp_path)
    curve = [{"accuracy": .999, "roc_auc": .6},
             {"accuracy": .991, "roc_auc": .9},
             {"accuracy": .998, "roc_auc": .9},
             {"accuracy": .995, "roc_auc": .8}]
    args = SimpleNamespace(curve=curve, test_accuracy=[.5, .4, 1., .9])
    native.main(args, None)
    assert args.saved == (0, 10)  # Unchanged upstream selects accuracy.
    patch = entry.adapt_chb_selection(native)
    native.main(args, None)
    assert args.saved == (1, 11)  # AUROC, first tie; neither final nor test-best.
    assert args.model_weight == 13  # Restoring saved, not terminal, is necessary.
    assert curve[0]["accuracy"] == .999  # Logged metrics were not relabeled.
    assert '+        if max_accuracy < val_stats["roc_auc"]:' in patch
    assert (tmp_path / "native_selection.py").read_text() == NATIVE_LOOP


def test_zero_auc_still_retains_the_first_checkpoint(tmp_path):
    native = native_module(tmp_path)
    entry.adapt_chb_selection(native)
    args = SimpleNamespace(curve=[{"accuracy": .99, "roc_auc": 0.}], test_accuracy=[1.])
    native.main(args, None)
    assert args.saved == (0, 10)


def test_changed_native_selection_anchor_is_rejected(tmp_path):
    native = native_module(tmp_path)
    path = Path(native.__file__)
    path.write_text(NATIVE_LOOP.replace('max_accuracy = 0.0', 'max_accuracy = .0'))
    with pytest.raises(ValueError, match="anchor changed"):
        entry.adapt_chb_selection(native)


def test_deferred_test_is_never_iterated_and_validation_is_unchanged():
    calls = []
    scores = {"accuracy": .999, "roc_auc": .6, "loss": .1}

    def evaluate(loader, model, device, **kwargs):
        calls.append(kwargs["header"])
        return scores

    wrapped = entry.defer_test_evaluation(evaluate, ["accuracy", "roc_auc"], "roc_auc")
    assert wrapped(object(), None, None, header="Test:") == {"accuracy": 0., "roc_auc": 0., "loss": 0.}
    assert not calls
    assert wrapped(None, None, None, header="Val:") is scores
    scores["roc_auc"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite validation"):
        wrapped(None, None, None, header="Val:")


@pytest.mark.parametrize("pretrained", [False, True])
def test_chb_configs_keep_global512_and_explicit_auc(pretrained):
    name = "chbmit-pretrained.json" if pretrained else "chbmit-scratch.json"
    cfg = json.loads((ROOT / "configs/baselines/labram" / name).read_text())
    assert entry.selection_metric(cfg) == "roc_auc"
    assert cfg["batch_size"] == 64 and cfg["update_freq"] == 8
    assert (cfg["lr"], cfg["epochs"], cfg["warmup_epochs"]) == (5e-4, 50, 5)
    assert cfg["layer_decay"] == (.65 if pretrained else 1.)
    cfg["update_freq"] = 1
    with pytest.raises(ValueError, match="effective batch512"):
        entry.selection_metric(cfg)


def test_non_chb_retains_native_accuracy():
    cfg = json.loads((ROOT / "configs/baselines/labram/tuev-shared16-scratch.json").read_text())
    assert entry.selection_metric(cfg) == "accuracy"
    cfg["selection_metric"] = "roc_auc"
    with pytest.raises(ValueError, match="native validation accuracy"):
        entry.selection_metric(cfg)
