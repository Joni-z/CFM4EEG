# Dataset contracts

The comparison inputs are TUEV, IIIC, TUSZ, CHB-MIT, TUAB, ISRUC, Sleep-EDF and
ADFD. Other clinical readers remain available independently of this matrix.

Shared caches contain `train/val/test_signals.npy`, matching label files and
`manifest.json`. `ACCEPTED.json` binds the manifest and array inventory; corrected
CHB-MIT/TUAB builds also require `READY.json`. The manifest defines subjects,
sampling rate, channels, windows, normalization and labels. A directory name alone
does not identify a protocol.

| Input | Preparation / origin | Retained contract |
|---|---|---|
| CHB-MIT | `preprocessing.rebuild_clinical --dataset chbmit` | Group chb01/chb21; augment seizures only in training; natural-grid validation/test; overlap labels; float64 inverse-Q95 sidecars |
| TUAB | `preprocessing.rebuild_clinical --dataset tuab` | Assign each patient once across normal/abnormal directories; disjoint validation; physical µV/100 arrays |
| TUEV shared16 | Accepted shared bipolar cache; reader `preprocessing.tuev` | 16 bipolar channels, 200 Hz, 5 seconds, stored µV/100; retain its exact subjects and windows |
| TUEV native23 | Pinned original `make_TUEV.py`; verified preprocessing source and pickles | 23 monopolar channels, 200 Hz, 5 seconds, physical µV; native engine divides by 100 once; different cohort from shared16 |
| IIIC | Accepted preprocessed release; `preprocessing.iiic` | Retain release filtering and current patient split; 16 channels, six classes, 10 seconds at 200 Hz, stored µV/100 |
| TUSZ | Accepted shared cohort; `preprocessing.tusz` | Retain bipolar montage, overlap rule, patient split and µV/100 |
| ISRUC | Accepted sequence cache; `preprocessing.isruc` | Host-compatible sequence/epoch layout and flattening; preserve label order |
| Sleep-EDF | Accepted cache; `preprocessing.sleepedf` | 100 Hz input; declare model-specific resampling; a 75 Hz passband cannot be applied at 100 Hz |
| ADFD | Medformer-derived cohort; `preprocessing.medformer_npy` | 19 channels, 200 Hz, 10 seconds, µV/100; acceptance does not establish full raw-EDF lineage or channel-order reconstruction |

Shared16 TUEV contains 68,445 / 15,487 / 29,421 training/validation/test windows;
IIIC contains 73,637 / 15,778 / 15,780. Use the exact manifest for other split
counts. CAUEEG and TUAR storage copies are separate from these comparison inputs;
a legacy copy is not automatically an accepted new version.

## Units and host interfaces

- Reverse Q95 with its saved denominator when a loader expects physical µV;
  an arbitrary rescale is not equivalent.
- If an engine divides physical µV by 100, restore that physical input first
  when using a cache already stored as µV/100. Avoid a second division.
- Sixteen bipolar differences do not uniquely reconstruct 23 monopolar signals.
  Shared16 and native23 results must remain visibly separate.
- Declare shared preprocessing separately from an author's data protocol, even
  when using that author's training code verbatim.
- REVE's legacy 16/19-channel coordinate defaults are an input convention;
  channel count alone does not verify anatomical identity. See
  [reve_positions.py](../scripts/rebuild/reve_positions.py).

Model-specific unit bridges are listed in the [baseline guide](UPSTREAM.md).

## Dataset preparation and acceptance

[data_reference/](../data_reference/) holds raw-cohort/hash references for
corrected builds. Acceptance scripts under [scripts/](../scripts/) check data;
they do not train, tune or change labels. Preprocessing/acceptance runs in the
corresponding CPU allocation.

Write regenerated data into a new directory and retain its manifest and build
receipts. Do not overwrite completed caches to tidy the repository. The retired
CHB/TUAB standalone generators are not alternate build paths; CHB annotation and
channel readers remain because the corrected rebuild imports them.
