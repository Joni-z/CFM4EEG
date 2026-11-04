"""Checks for split leakage and pairing shortcuts in the frozen legacy probe."""
import numpy as np

from scripts.experiments.legacy_rotation_probe import pairing_data, wrong_donors, paired_accuracy_interval


def test_quadruplets_are_indivisible_and_spectrally_matched():
    data = pairing_data([32, 16, 32], 1105)
    for group in np.unique(data['group']):
        ids = np.flatnonzero(data['group'] == group)
        assert len(ids) == 4
        assert len(np.unique(data['split'][ids])) == 1
        assert data['labels'][ids].sum() == 2
        spectra = abs(np.fft.rfft(data['signals'][ids, 0].astype(float)))
        np.testing.assert_allclose(spectra, np.repeat(spectra[:1], 4, axis=0),
                                   rtol=1e-4, atol=2e-4)
    assert len(np.unique(data['sample_id'])) == len(data['sample_id'])


def test_donors_cannot_cross_splits_or_use_other_members_of_same_quadruplet():
    data = pairing_data([32, 16, 32], 1105)
    donor = wrong_donors(data['split'], data['group'], 2405)
    assert len(np.unique(donor)) == len(donor)
    assert np.all(data['split'][donor] == data['split'])
    assert np.all(data['group'][donor] != data['group'])
    # Independent within-group permutation must not preserve the fixed label order.
    assert np.any(data['labels'][donor] != data['labels'])


def test_interval_resamples_whole_groups():
    labels = np.tile([1, 1, 0, 0], 8)
    groups = np.repeat(np.arange(8), 4)
    exact = paired_accuracy_interval(labels, labels, labels, groups)
    assert exact['difference'] == 0 and exact['ci95'] == [0, 0]
    extreme = paired_accuracy_interval(labels, 1-labels, labels, groups)
    assert extreme['groups'] == 8
    assert extreme['difference'] == 1 and extreme['ci95'] == [1, 1]
