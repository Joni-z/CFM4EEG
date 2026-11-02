"""Numerical definitions and split-isolation checks for the PAC readout."""

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from paclock_bench.probing import scoring


@pytest.fixture
def curves():
    target = np.array([[1., -1.], [2., -2.], [4., -4.], [0., 0.]])
    return target, np.array([1, 1, 1, 0])


def test_perfect_curves_and_detection(curves):
    target, coupled = curves
    result = scoring.score_curves(target, target, coupled)
    assert result == {"curve_r2": 1., "curve_rmse": 0., "detection_auroc": 1.,
                      "n": 4, "n_coupled": 3, "missing_reasons": {}}
    json.dumps(result, allow_nan=False)


def test_mean_predictor_is_zero_and_wrong_predictor_stays_negative(curves):
    target, coupled = curves
    prediction = np.broadcast_to(target[:3].mean(axis=0), target.shape)
    assert scoring.score_curves(target, prediction, coupled)["curve_r2"] == pytest.approx(0.)
    result = scoring.score_curves(target, -target, coupled)
    expected = 1 - np.square(2 * target[:3]).sum() / np.square(
        target[:3] - target[:3].mean(axis=0)).sum()
    assert result["curve_r2"] == pytest.approx(expected)
    assert result["curve_r2"] < 0


def test_pooled_r2_matches_sklearn_with_nonconstant_outputs():
    target = np.array([[1., 10.], [3., 5.], [2., -2.]])
    prediction = target + np.array([[.1, 1.], [-.5, 2.], [.2, 1.]])
    expected = r2_score(target, prediction, multioutput="variance_weighted")
    assert scoring.score_curves(target, prediction, [1, 1, 1])["curve_r2"] == pytest.approx(expected)


def test_constant_output_column_prediction_errors_count():
    target = np.array([[1., 8.], [2., 8.], [3., 8.]])
    prediction = target + [0., 2.]
    result = scoring.score_curves(target, prediction, [1, 1, 1])
    assert result["curve_r2"] == pytest.approx(-5.)  # SSE=12; SST=2.
    assert result["curve_rmse"] == pytest.approx(np.sqrt(2.))


@pytest.mark.parametrize("offset", [0., 1.])
def test_all_constant_targets_have_no_r2(offset):
    target = np.full((3, 4), .1)
    result = scoring.score_curves(target, target + offset, [1, 1, 1])
    assert result["curve_r2"] is None
    assert result["missing_reasons"]["curve_r2"] == "constant_coupled_targets"
    assert result["curve_rmse"] == pytest.approx(offset)
    assert result["detection_auroc"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_rows_are_explicitly_missing(n):
    result = scoring.score_curves(np.zeros((n, 4)), np.zeros((n, 4)), np.ones(n))
    assert result["curve_r2"] is None
    assert result["missing_reasons"]["curve_r2"] == "fewer_than_two_coupled_rows"
    assert result["curve_rmse"] == (0. if n else None)
    json.dumps(result, allow_nan=False)


def test_null_only_rmse_missing_and_auc_ties(curves):
    target, coupled = curves
    assert scoring.score_curves(target, target, np.zeros(4))["curve_rmse"] is None
    result = scoring.score_curves(target, np.zeros_like(target), coupled)
    assert result["detection_auroc"] == .5
    prediction = np.zeros_like(target)
    prediction[-1] = 100
    result = scoring.score_curves(target, prediction, coupled)
    assert result["detection_auroc"] == 0.
    assert result["curve_rmse"] == pytest.approx(np.sqrt(np.square(target[:3]).mean()))


@pytest.mark.parametrize("target,prediction,coupled", [
    ([[1., np.nan]], [[1., 2.]], [1]),
    ([[1., 2.]], [[1., np.inf]], [1]),
    ([[1., 2.]], [[1.]], [1]),
    ([1., 2.], [1., 2.], [1]),
    ([[1., 2.]], [[1., 2.]], [[1]]),
    ([[1., 2.]], [[1., 2.]], [2]),
    ([[1., 2.]], [[1., 2.]], [np.nan]),
    ([[1j, 2.]], [[1., 2.]], [1]),
])
def test_score_rejects_invalid_inputs(target, prediction, coupled):
    with pytest.raises(ValueError):
        scoring.score_curves(target, prediction, coupled)


def test_train_only_scaling_selection_and_no_refit(monkeypatch):
    train_x = np.array([[0., 0.], [1., 20.], [2., -10.], [4., 40.]])
    train_y = np.array([[1., -1.], [2., -2.], [0., 0.], [3., -3.]])
    val_x = np.array([[10., 200.], [20., -300.]])
    val_y = np.array([[0., 0.], [1., -1.]])  # Null rows participate in selection.
    test_x = np.array([[1000., 9000.], [-2000., 8000.]])
    alphas = [0., .1, 10., 1000.]
    scaler = StandardScaler().fit(train_x)
    expected_models = [Ridge(alpha=a, solver="svd").fit(scaler.transform(train_x), train_y)
                       for a in alphas]
    errors = [mean_squared_error(val_y, m.predict(scaler.transform(val_x)))
              for m in expected_models]
    best = int(np.argmin(errors))
    scaler_fits, ridge_fits = [], []
    original_scaler_fit, original_ridge_fit = StandardScaler.fit, Ridge.fit

    def record_scaler(self, x, *args, **kwargs):
        scaler_fits.append(np.array(x))
        return original_scaler_fit(self, x, *args, **kwargs)

    def record_ridge(self, x, y, *args, **kwargs):
        ridge_fits.append((np.array(x), np.array(y)))
        return original_ridge_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_scaler)
    monkeypatch.setattr(Ridge, "fit", record_ridge)
    prediction, report = scoring.fit_probe(train_x, train_y, val_x, val_y, test_x, alphas=alphas)
    assert len(scaler_fits) == 1 and len(ridge_fits) == len(alphas)
    np.testing.assert_array_equal(scaler_fits[0], train_x)
    for x, y in ridge_fits:
        np.testing.assert_array_equal(x, scaler.transform(train_x))
        np.testing.assert_array_equal(y, train_y)
    assert report["selected_alpha"] == alphas[best]
    np.testing.assert_allclose([v["mse"] for v in report["validation_mse"]], errors)
    np.testing.assert_allclose(prediction, expected_models[best].predict(scaler.transform(test_x)))
    json.dumps(report, allow_nan=False)


def test_probe_determinism_affine_intercept_and_singular_features():
    x = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    y = np.column_stack([3 + 2 * x[:, 0], 1 - x[:, 0]])
    args = (x, y, x + 5, y + [10, -5], x + 10)
    prediction, report = scoring.fit_probe(*args)
    repeated, repeated_report = scoring.fit_probe(*args)
    np.testing.assert_array_equal(prediction, repeated)
    assert report == repeated_report
    assert report["selected_alpha"] == 0.
    np.testing.assert_allclose(prediction, y + [20, -10], atol=1e-12)


def test_exact_ties_keep_first_alpha():
    x = np.arange(8.).reshape(4, 2)
    y = np.zeros((4, 2))
    _, report = scoring.fit_probe(x, y, x, y, x, alphas=[10., 0., 1.])
    assert report["selected_alpha"] == 10.


@pytest.mark.parametrize("change", [
    {"train_x": [[np.nan, 1.], [0., 1.]]},
    {"train_y": [[1., 2.]]},
    {"val_x": [[1.], [2.]]},
    {"val_y": [[1.], [2.]]},
    {"test_x": np.empty((0, 2))},
    {"alphas": []}, {"alphas": [-1.]}, {"alphas": [np.inf]}, {"alphas": [[1.]]},
])
def test_probe_rejects_invalid_inputs(change):
    arrays = {name: np.ones((2, 2))
              for name in ("train_x", "train_y", "val_x", "val_y", "test_x")}
    arrays.update(change)
    with pytest.raises(ValueError):
        scoring.fit_probe(**arrays)
