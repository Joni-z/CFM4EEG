# Dataset contracts

The comparison matrix contains TUEV, IIIC, TUSZ, CHB-MIT, TUAB, ISRUC, Sleep-EDF and ADFD. Additional clinical readers are kept for existing data, independently of the current matrix.

Shared arrays use `train/val/test_signals.npy`, matching labels, and `manifest.json`. `ACCEPTED.json` binds each manifest and array inventory. Corrected CHB-MIT/TUAB builds also require `READY.json`. Manifests define subject membership, sampling rate, channels, windowing, normalization and labels; directory names alone do not identify a protocol.

| Input | Canonical preparation / origin | Contract that must be retained |
|---|---|---|
| CHB-MIT | `preprocessing.rebuild_clinical --dataset chbmit` | Group chb01/chb21; seizure augmentation only in train; natural-grid validation/test; overlap labels; float64 inverse-Q95 sidecars |
| TUAB | `preprocessing.rebuild_clinical --dataset tuab` | Assign each patient once across normal/abnormal directories; disjoint validation; physical µV/100 arrays |
| TUEV shared16 | Existing accepted shared bipolar cache; reader `preprocessing.tuev` |16bipolar channels,200Hz,5s,storedµV/100; retain its exact existing subjects/windows |
| TUEV native23 | Pinned original `make_TUEV.py`; preprocessing source and every pickle verified |23monopolar channels,200Hz,5s,physicalµV; native engine divides100 once; different test cohort from shared16 |
| IIIC | Accepted released preprocessed signal cohort; `preprocessing.iiic` | Retain upstream release filtering and current patient split; storedµV/100 |
| TUSZ | Accepted shared cohort; `preprocessing.tusz` | Retain bipolar montage, overlap rule, patient split andµV/100 |
| ISRUC | Accepted sequence cache; `preprocessing.isruc` | Sequence/epoch layout and flattening must match the host; never silently reorder labels |
| Sleep-EDF | Accepted cache; `preprocessing.sleepedf` |100Hz input; model-required resampling is explicit; do not apply a75Hz passband at100Hz |
| ADFD | Retained Medformer-derived cohort; `preprocessing.medformer_npy` |19channels,200Hz,10s,µV/100; complete raw-EDF lineage and channel-order reconstruction are not established by array acceptance |

CAUEEG and TUAR storage copies are separate from the accepted eight-task comparison inputs; a legacy directory is not automatically an accepted new version. No dataset is regenerated merely to tidy the repository.

The old CHB/TUAB standalone generators are retired. CHB’s annotation/channel readers remain because the corrected rebuild calls them. Generate a new version into a new directory, using an allocated CPU job and the raw-recording reference hashes; never overwrite a completed cache.

## Units and cohort boundaries

- Q95 normalization is reversed using the saved denominator when an author loader expects physicalµV. An arbitrary rescale is not equivalent.
- Inputs already stored asµV/100 must not be divided again before an engine that assumes physicalµV.
-23monopolar signals cannot be reconstructed uniquely from16bipolar differences. Native23 and shared16 results cannot be silently combined as one test set.
- Shared-cohort preprocessing and an author’s native data protocol are named separately, even when training code is copied verbatim.
- REVE’s16/19-channel legacy coordinate defaults remain a documented input convention; channel count alone does not verify anatomical identity.

`data_reference/` contains the raw cohort/hash references used by corrected builds. Acceptance scripts under `scripts/` check data only; they do not train, tune or change test labels.
