import gzip
import json

import pytest

from scripts.experiments.cfc_operator_reproduce import detection, paired_errors, records, summarize, ARMS


def row(seed, score, positive=False, valid=True):
    return dict(condition='single10' if positive else 'independent',seed=seed,
                d1=.5 if positive else 0.,d2=0.,valid=valid,curve_rmse=.1,
                predictive_score=score,pair_predictive_score=score)


def test_paired_comparison_rejects_different_records():
    with pytest.raises(ValueError,match='paired'):
        paired_errors([row(1,.1,True)],[row(2,.1,True)])


def test_unavailable_positive_measurements_are_penalized():
    a=[row(1,.1,True,False)];b=[row(1,.1,True)]
    ea,eb,groups=paired_errors(a,b)
    assert ea.tolist()==[1.] and eb.tolist()==[.1]
    assert groups.tolist()==['single10']


def test_calibration_threshold_does_not_use_test_scores():
    calibration=[row(i,s) for i,s in enumerate([.1,.2,.3,.4])]
    # Both stress categories are retained separately from independent nulls.
    stress=[dict(row(10,.1),condition='transient'),dict(row(11,.1),condition='nonsinusoidal')]
    one=detection(calibration,[row(20,.5),row(21,.8,True),*stress])
    two=detection(calibration,[row(20,.9),row(21,.99,True),*stress])
    assert one['threshold']==two['threshold']==.4
    assert one['false_positive_rate']==1. and one['sensitivity']==1.


def test_compressed_evidence_records_are_read_without_modification(tmp_path):
    p=tmp_path/'arms'/'example';p.mkdir(parents=True)
    expected=[row(1,.5,True)]
    with gzip.open(p/'records.json.gz','wt') as handle:
        json.dump(expected,handle)
    assert records(tmp_path,'arms/example')==expected


def test_summary_rejects_mislabeled_operator_settings(tmp_path):
    for name,flags in ARMS.items():
        folder=tmp_path/'arms'/name;folder.mkdir(parents=True)
        settings=dict(harmonics=int(flags[1]),phase_cycles=float(flags[3]),
                      refine_phase='--refine-phase' in flags,control_amplitude=False,local=False)
        if name=='fixed_h2':settings['control_amplitude']=True
        (folder/'summary.json').write_text(json.dumps(dict(args=settings)))
    with pytest.raises(ValueError,match='fixed_h2'):
        summarize(tmp_path)
