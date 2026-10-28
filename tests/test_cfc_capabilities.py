import numpy as np
import pytest

from scripts.experiments.cfc_operator_capabilities import generate, specifications, summarize


@pytest.mark.parametrize('shape', ['single', 'double', 'mixed'])
def test_synthetic_truth_recovers_in_independent_numpy_regression(shape):
    spec = dict(seconds=10, source=2, target=2, depth=.4, snr_db=5., shape=shape)
    e = generate(123, spec)
    phi = e['phi']
    design = np.array([np.cos(phi), np.sin(phi), np.cos(2*phi), np.sin(2*phi)]).T
    beta = np.linalg.lstsq(design, e['modulation'], rcond=None)[0]
    truth = [e['d1']*np.cos(e['theta']), e['d1']*np.sin(e['theta']),
             e['d2']*np.cos(e['theta2']), e['d2']*np.sin(e['theta2'])]
    np.testing.assert_allclose(beta, truth, atol=1e-12)
    assert abs(e['target_band_snr_db']-5.) < 1e-10


def test_grid_and_null_truth_cover_declared_domain():
    grid = specifications('grid')
    assert len(grid) == 180
    assert {r['seconds'] for r in grid} == {5, 10, 20}
    assert len({(r['source'], r['target']) for r in grid}) == 30
    for spec in specifications('null'):
        e = generate(321, spec)
        assert np.max(np.abs(e['modulation'])) == 0


def test_invalid_positive_is_retained_in_unconditional_metric():
    row = dict(variant='curve', group='test', valid=True, curve_rmse=.1,
               phase1_error_deg=2., depth1_error=.03, top1_correct=True,
               source_hz_error=.01, record_score=.5)
    result = summarize([row, row | dict(valid=False, curve_rmse=.4)])['curve/test']
    assert result['n'] == 2
    assert result['valid_fraction'] == .5
    assert result['curve_rmse_all'] == .25
    assert result['curve_rmse_valid'] == .1


def test_nonlinear_generator_reports_nonzero_fourier_truth():
    e = generate(4, dict(seconds=10, source=2, target=2, depth=.5,
                         snr_db=10., shape='nonlinear'))
    assert .48 < e['d1'] < .49
    assert .05 < e['d2'] < .07
    assert e['theta2'] == 2*e['theta']


def test_report_pairs_records_by_identity_and_keeps_finite_rate_uncertainty():
    from scripts.experiments.cfc_operator_capabilities_report import paired_difference, rate_interval
    rows = [dict(group='a', seed=0, variant='old', curve_rmse=.1),
            dict(group='a', seed=1, variant='old', curve_rmse=.8),
            dict(group='a', seed=1, variant='new', curve_rmse=.7),
            dict(group='a', seed=0, variant='new', curve_rmse=.05)]
    assert paired_difference(rows, 'old', 'new')['mean_reduction'] == pytest.approx(.075)
    lo,hi=rate_interval(5,100)
    assert lo == pytest.approx(.02154368, abs=1e-7)
    assert hi == pytest.approx(.11175047, abs=1e-7)
