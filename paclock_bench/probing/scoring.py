"""Frozen-feature PAC readout using scikit-learn implementations.

Train-only scaling and ridge fitting with validation-only regularization follow
the probing setup of Tang et al., https://arxiv.org/html/2605.11410v2, Sec. 3.2
and App. C.5. Generator-defined PAC-curve targets, validation MSE selection,
unclipped pooled R² and RMS-based detection AUROC are benchmark adaptations;
this module is not an implementation supplied by that paper.

Implementation references:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _matrix(value, name: str, *, allow_empty: bool = False) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_features), with n_features > 0")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is non-finite; check the input magnitude")
    return result


def score_curves(target, prediction, coupled) -> dict:
    """Score aligned [N, K] baseline-subtracted curves and binary [N] labels.

    Curve metrics use coupled rows only. R² is 1 - total SSE / total SST,
    with each phase bin centered across those rows, not per-curve R² or a
    training-mean-relative score. Centered flattening lets sklearn retain
    prediction errors in constant columns without a ``NaN * 0`` aggregation.
    Detection uses predicted curve RMS on all rows; inputs should therefore
    have the null curve at zero. Undefined metrics are None with a reason.
    """
    target = _matrix(target, "target", allow_empty=True)
    prediction = _matrix(prediction, "prediction", allow_empty=True)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes")
    labels = np.asarray(coupled)
    if labels.shape != (len(target),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("coupled must be a binary vector with shape (n_samples,)")
    positive = labels == 1
    n_positive = int(positive.sum())
    reasons = {}
    result = {
        "curve_r2": None, "curve_rmse": None, "detection_auroc": None,
        "n": len(target), "n_coupled": n_positive, "missing_reasons": reasons,
    }
    if n_positive:
        truth, predicted = target[positive], prediction[positive]
        result["curve_rmse"] = _finite_float(
            np.sqrt(mean_squared_error(truth, predicted)), "curve_rmse"
        )
    else:
        reasons["curve_rmse"] = "no_coupled_rows"

    if n_positive < 2:
        reasons["curve_r2"] = "fewer_than_two_coupled_rows"
    elif np.all(truth == truth[:1]):
        reasons["curve_r2"] = "constant_coupled_targets"
    else:
        mean = truth.mean(axis=0)
        result["curve_r2"] = _finite_float(
            r2_score((truth - mean).ravel(), (predicted - mean).ravel(),
                     force_finite=False), "curve_r2"
        )

    if n_positive == 0 or n_positive == len(target):
        reasons["detection_auroc"] = "both_coupled_and_null_rows_required"
    else:
        rms = np.sqrt(np.mean(np.square(prediction), axis=1))
        result["detection_auroc"] = _finite_float(
            roc_auc_score(labels, rms), "detection_auroc"
        )
    return result


def fit_probe(train_x, train_y, val_x, val_y, test_x, *,
              alphas=(0., .01, .1, 1., 10., 100.)) -> tuple[np.ndarray, dict]:
    """Fit affine Ridge probes on training rows; choose by all-row val MSE.

    StandardScaler is fit once on train_x. Neither validation nor test rows
    enter any fit, including after selection. SVD handles singular/duplicated
    features deterministically. Exact MSE ties keep the first supplied alpha.
    Targets retain their physical scale; no null rows are removed. The returned
    report contains finite JSON-native values and ordered validation errors.
    """
    train_x = _matrix(train_x, "train_x")
    train_y = _matrix(train_y, "train_y")
    val_x = _matrix(val_x, "val_x")
    val_y = _matrix(val_y, "val_y")
    test_x = _matrix(test_x, "test_x")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("feature and target row counts must match within each split")
    if train_x.shape[1] != val_x.shape[1] or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("feature counts must match across train, validation and test")
    if train_y.shape[1] != val_y.shape[1]:
        raise ValueError("target output counts must match across train and validation")
    alpha_values = np.asarray(alphas, dtype=np.float64)
    if (alpha_values.ndim != 1 or not len(alpha_values)
            or not np.isfinite(alpha_values).all() or (alpha_values < 0).any()):
        raise ValueError("alphas must be a nonempty sequence of finite nonnegative values")

    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    val_scaled = scaler.transform(val_x)
    validation_errors = []
    best_model, best_error = None, float("inf")
    for alpha in alpha_values:
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
        model.fit(train_scaled, train_y)
        error = _finite_float(mean_squared_error(val_y, model.predict(val_scaled)),
                              "validation_mse")
        validation_errors.append({"alpha": float(alpha), "mse": error})
        if error < best_error:
            best_model, best_error = model, error

    predictions = best_model.predict(scaler.transform(test_x))
    if not np.isfinite(predictions).all():
        raise ValueError("test predictions are non-finite; check the input magnitude")
    report = {
        "selected_alpha": float(best_model.alpha),
        "selected_validation_mse": best_error,
        "validation_mse": validation_errors,
        "selection": "minimum_validation_mse_first_alpha_on_tie",
        "scaler_fit_split": "train", "probe_fit_split": "train",
        "n_train": len(train_x), "n_validation": len(val_x), "n_test": len(test_x),
        "n_features": train_x.shape[1], "n_outputs": train_y.shape[1],
    }
    return predictions, report
