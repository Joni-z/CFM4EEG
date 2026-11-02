"""CLI wiring with temporary random-weight fixtures, never research results."""
from argparse import Namespace
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.probing.benchmark import save_npz
from paclock_bench.probing.synthetic import generate_dataset
from scripts.experiments.pac_token_probe import evaluate, extract


def setup(tmp_path, row="wavepatch", sample_rate=200):
    dataset = tmp_path / "data.npz"
    if not dataset.exists():
        save_npz(dataset, **generate_dataset(8, 4, 8))
    cfg = dict(model="paclock" if row == "cfft" else "wave_patch_twoaxis",
               dataset="synthetic", sample_rate=sample_rate, num_classes=2,
               model_kwargs=dict(d_model=64, depth=1, n_heads=4, dropout=.1,
                                 patch_len=50, spatial_pe="index", augmentations=[]))
    if row == "cfft":
        cfg["model_kwargs"].update(arch="triaxial", n_bands=4, kernel_size=201,
                                   tokenizer_mode="cfc", freq_mixer="none",
                                   waveform_modulation="quadrature")
    config = tmp_path / f"{row}.yaml"
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / f"{row}-random-fixture.pt"
    torch.save({"model": build_model(cfg, (2, 1000)).state_dict(), "config": cfg,
                "purpose": "random test fixture, not trained"}, checkpoint)
    return Namespace(dataset=dataset, output=tmp_path / f"{row}.npz", row=row,
                     config=config, checkpoint=checkpoint, normalization="none",
                     training_channels=2, device="cpu", input_scale=20., batch_size=4,
                     time_bins=9, width=64, projection_seed=0, vendor_root=None)


def test_project_extraction_evaluation_and_budget_guard(tmp_path):
    torch.set_num_threads(1)
    args = setup(tmp_path)
    extract(args)
    cfft = setup(tmp_path, "cfft")
    extract(cfft)
    # Extraction must not poison the shared Hilbert cache for later training.
    training_model = build_model(yaml.safe_load(cfft.config.read_text()), (2, 1000))
    training_model.frontend(torch.randn(1, 2, 1000))[0].square().mean().backward()
    assert any(p.grad is not None for p in training_model.frontend.parameters())
    power = Namespace(**{**vars(args), "row": "bandpower", "output": tmp_path / "power.npz"})
    extract(power)
    evaluation = Namespace(dataset=args.dataset, cache=[f"wave={args.output}",
                           f"cfft={cfft.output}", f"power={power.output}"],
                           output=tmp_path / "scores", bootstrap=20, seed=0)
    evaluate(evaluation)
    scores = json.loads((evaluation.output / "scores.json").read_text())
    assert set(scores["rows"]) == {"wave", "cfft", "power"}
    assert scores["rows"]["wave"]["source"]["native_token_shape"] == [20, 64]
    assert scores["rows"]["cfft"]["source"]["native_token_shape"] == [20, 256]
    assert scores["rows"]["wave"]["fit"]["n_features"] == 576
    assert len(scores["paired_errors"]) == 3
    args.output = tmp_path / "padded.npz"
    args.width = 128
    with pytest.raises(ValueError, match="must not upsample"):
        extract(args)
    assert not args.output.exists()


def test_mismatched_sampling_rate_rejected(tmp_path):
    args = setup(tmp_path, sample_rate=100)
    with pytest.raises(ValueError, match="checkpoint sample rate"):
        extract(args)
    assert not args.output.exists()


@pytest.mark.skipif(not os.environ.get("TFM_VENDOR_ROOT"), reason="optional pinned author checkout")
def test_tfm_cli_quantized_cache(tmp_path):
    args = setup(tmp_path)
    root = Path(os.environ["TFM_VENDOR_ROOT"])
    spec = importlib.util.spec_from_file_location("tfm_cli_fixture", root / "models/tfm_token.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "tfm-random-fixture.pt"
    torch.save(module.get_tfm_tokenizer_2x2x8().state_dict(), checkpoint)
    args.row, args.vendor_root, args.checkpoint = "tfm", root, checkpoint
    args.output = tmp_path / "tfm.npz"
    extract(args)
    evaluation = Namespace(dataset=args.dataset, cache=[f"tfm={args.output}"],
                           output=tmp_path / "tfm-scores", bootstrap=20, seed=0)
    evaluate(evaluation)
    row = json.loads((evaluation.output / "scores.json").read_text())["rows"]["tfm"]
    assert row["source"]["native_token_shape"] == [9, 64]
    assert row["source"]["endpoint"] == "tokenize.quant_out"
