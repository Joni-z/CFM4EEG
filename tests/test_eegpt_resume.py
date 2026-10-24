"""Recover the schedule and keep the validation winner across interrupted segments."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "eegpt_resume", Path(__file__).parents[1] / "scripts/rebuild/run_eegpt_chb_official.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Scaler:
    def load_state_dict(self, value):
        self.value = value


def test_resume_restores_optimizer_and_continues_saved_epoch(tmp_path):
    source = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(source.parameters(), lr=.001)
    source(torch.ones(3, 2)).square().mean().backward()
    optimizer.step()
    checkpoint = dict(model=source.state_dict(), optimizer=optimizer.state_dict(),
                      scaler={"scale": 1024.}, epoch=17,
                      args=argparse.Namespace(epochs=50))
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    restored = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=.9)
    args, scaler = argparse.Namespace(epochs=50, start_epoch=0), Scaler()
    assert entry.restore_training_checkpoint(path, args, restored, resumed_optimizer, scaler) == 17
    assert args.start_epoch == 18
    assert scaler.value == {"scale": 1024.}
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    a, b = optimizer.state_dict(), resumed_optimizer.state_dict()
    assert a["param_groups"] == b["param_groups"]
    for key, state in a["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["state"][key][name], rtol=0, atol=0)
    # The first resumed update also follows the uninterrupted Adam state.
    for model, opt in [(source, optimizer), (restored, resumed_optimizer)]:
        opt.zero_grad()
        model(torch.tensor([[.5, 2.]])).square().mean().backward()
        opt.step()
    for left, right in zip(source.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["best_epoch", "optimizer_missing", "nonfinite"])
def test_invalid_resume_is_rejected_before_model_changes(tmp_path, problem):
    model = torch.nn.Linear(2, 1)
    checkpoint = dict(model=copy.deepcopy(model.state_dict()),
                      optimizer=torch.optim.AdamW(model.parameters()).state_dict(),
                      scaler={"scale": 1024.}, epoch=17)
    if problem == "best_epoch":
        checkpoint["epoch"] = "best"
    elif problem == "optimizer_missing":
        checkpoint.pop("optimizer")
    else:
        checkpoint["model"]["weight"].fill_(float("nan"))
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        entry.restore_training_checkpoint(path, argparse.Namespace(epochs=50), model,
                                          torch.optim.AdamW(model.parameters()), Scaler())
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)


def test_resume_preserves_global_validation_winner(tmp_path):
    log = tmp_path / "log.txt"
    records = [dict(epoch=0, val_accuracy=.8, test_accuracy=.99),
               dict(epoch=14, val_accuracy=.9, test_accuracy=.1),
               dict(epoch=17, val_accuracy=.85, test_accuracy=1.)]
    log.write_text("\n".join(json.dumps(x) for x in records))
    best = entry.previous_best_accuracy(log, 17)
    assert best == .9
    selected, native_best = tmp_path / "selected_model.pt", tmp_path / "checkpoint-best.pth"
    selected.write_bytes(b"previous winner")
    native_best.write_bytes(b"previous native winner")
    def save():
        selected.write_bytes(b"new winner")
        native_best.write_bytes(b"new native winner")
    for score in [.81, .89, .9]:
        best = entry.save_if_validation_improves(score, best, save)
        assert selected.read_bytes() == b"previous winner"
        assert native_best.read_bytes() == b"previous native winner"
    best = entry.save_if_validation_improves(.91, best, save)
    assert best == .91
    assert selected.read_bytes() == b"new winner"
    assert native_best.read_bytes() == b"new native winner"


def test_resume_rejects_checkpoint_older_than_preserved_selection(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text(json.dumps(dict(epoch=19, val_accuracy=.95)))
    with pytest.raises(ValueError, match="predates preserved validation history"):
        entry.previous_best_accuracy(path, 17)


def test_bf16_logits_bridge_supports_native_numpy_metrics_and_backward():
    layer = torch.nn.Linear(2, 1, dtype=torch.bfloat16)
    x = torch.tensor([[1., 2.], [.5, 1.]], dtype=torch.bfloat16)
    expected = layer(x).detach().float()
    layer.register_forward_hook(entry.fp32_logits)
    actual = layer(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
    assert torch.sigmoid(actual.detach()).numpy().shape == (2, 1)
    torch.nn.functional.binary_cross_entropy_with_logits(actual, torch.ones_like(actual)).backward()
    assert torch.isfinite(layer.weight.grad).all()
