"""Bounded CPU controls for the readout implementation, not model performance."""
from __future__ import annotations

import numpy as np

from .benchmark import evaluate_cache
from .scoring import fit_probe, score_curves
from .synthetic import curve_from_coefficients, generate_dataset


def validate_protocol(seed: int = 0) -> dict:
    data = generate_dataset(256, 128, 256, seed=seed)
    coefficients = data["coefficients"]
    rng = np.random.default_rng(seed + 200)
    # The truth representation is intentionally confined to validation controls.
    invertible = np.array([[2., 1., 0., 0.], [0., 1., 1., 0.],
                           [0., 0., 2., 1.], [1., 0., 0., 1.]])
    controls = {
        "oracle": coefficients,
        "invertible_oracle": coefficients @ invertible,
        "h1_only": coefficients[:, :2],
        "constant": np.zeros((len(coefficients), 4)),
        "independent_noise": rng.normal(size=coefficients.shape),
        "oracle_plus_noise": coefficients + rng.normal(scale=.4, size=coefficients.shape),
    }
    reports = {name: evaluate_cache(data, features)[0]["overall"]
               for name, features in controls.items()}
    train, val, test = (data["split"] == name for name in ("train", "val", "test"))
    permutation_scores = []
    for _ in range(20):
        prediction, _ = fit_probe(coefficients[train], rng.permutation(data["curves"][train]),
                                  coefficients[val], data["curves"][val], coefficients[test])
        permutation_scores.append(score_curves(data["curves"][test], prediction,
                                               data["coupled"][test])["curve_r2"])
    # The oracle ceiling should survive innocuous discretization changes.
    bin_checks = {}
    for bins in (18, 24, 36):
        variant = {**data, "curves": curve_from_coefficients(coefficients, bins)}
        bin_checks[str(bins)] = evaluate_cache(variant, coefficients)[0]["overall"]["curve_r2"]
    exact = score_curves(data["curves"][test], data["curves"][test], data["coupled"][test])
    checks = {
        "exact_prediction": exact["curve_r2"] == 1. and exact["detection_auroc"] == 1.,
        "oracle_recovered": reports["oracle"]["curve_r2"] > 1 - 1e-10,
        "linear_coordinates_recovered": reports["invertible_oracle"]["curve_r2"] > 1 - 1e-10,
        "known_h2_loss_detected": reports["h1_only"]["curve_r2"] < reports["oracle"]["curve_r2"] - .1,
        "added_noise_detected": reports["oracle_plus_noise"]["curve_r2"] < reports["oracle"]["curve_r2"] - .1,
        "constant_at_mean_baseline": reports["constant"]["curve_r2"] <= 1e-12,
        "random_not_above_label_null": reports["independent_noise"]["curve_r2"] <= max(permutation_scores),
        "bin_integrals_readable": min(bin_checks.values()) > 1 - 1e-10,
    }
    return {
        "passed": bool(all(checks.values())), "checks": checks, "controls": reports,
        "label_permutation_r2": permutation_scores, "oracle_r2_by_bins": bin_checks, "seed": seed,
        "scope": "implementation and known-information controls; no tokenizer ranking or clinical result",
        "remaining_protocol_checks": [
            "power/spectrum shortcut audit on intended formal dataset",
            "frequency/noise generalization conditions and readout-budget sensitivity",
            "operator-in-tokenizer consistency and measured-reference coverage",
        ],
    }
