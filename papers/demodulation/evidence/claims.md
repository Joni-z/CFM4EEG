# Claim registry for "Demodulate, Then Attend" (draft 2026-11-05)

One claim per line. Each line is the numeric or factual statement as it appears in the manuscript, bound to evidence IDs from source_manifest.json. The audit script hashes each line; edit the manuscript and this file together.

## Abstract and introduction

One learned band instead of eight costs 0.42 AUC-PR on CHB-MIT and 0.11 on TUSZ over three seeds. [claim:C001] [evidence:E001,E002]
A flat encoder instead of the three-axis encoder costs 0.53 AUC-PR on CHB-MIT and 0.22 on TUSZ over three seeds. [claim:C002] [evidence:E001,E002]
On TUEV the envelope view alone matches the full 16-band model (0.714 against 0.715 at seed 2). [claim:C003] [evidence:E001,E002,E003]
Removing the waveform view from the full 16-band three-axis model costs 0.167 AUC-PR on CHB-MIT at seed 2. [claim:C004] [evidence:E001,E002,E003]
Rotated versus unrotated content differs by 0.003 on average over eight paired runs. [claim:C005] [evidence:E001,E002,E009]
The operator that measures coupling accurately lowers TUEV kappa from 0.693 to 0.660 at seed 0. [claim:C006] [evidence:E005,E008]
The model with the rotation off reaches TUEV kappa 0.729 and 0.711 at seeds 0 and 2. [claim:C007] [evidence:E001,E009]
The rotation-on 16-band model reaches TUEV kappa 0.693 plus or minus 0.036 over three seeds. [claim:C008] [evidence:E001,E016]
TUSZ AUC-PR is 0.671 plus or minus 0.032 with raw band tokens and 0.680 plus or minus 0.048 with the two-axis duplex model, three seeds each. [claim:C009] [evidence:E001,E019]
IIIC kappa is 0.497 plus or minus 0.007 over three seeds for the 16-band model and 0.524 at seed 0 for the range variant. [claim:C010] [evidence:E001,E016]
The strongest reproduced baselines are CBraMod 0.575 on TUEV (native code, seed 0), FFCL 0.545 on TUSZ and REVE 0.436 on IIIC. [claim:C011] [evidence:E012,E018,E019]
The earlier CHB-MIT test set contained 162 extra seizure-centred windows; the same predictions score 0.697 on it and 0.486 on the natural grid. [claim:C012] [evidence:E005]
A linear probe could not recover the coupling curve from the trained tokens (R squared below 0.01). [claim:C013] [evidence:E006,E021]
An operator that recovered the coupling curve with R squared 0.98 lowered the clinical score when it replaced the rotation. [claim:C014] [evidence:E007,E008]
The 2026 benchmarks report that small models trained from scratch tie or beat 25M to 69M parameter pretrained models on clinical corpora. [claim:C015] [evidence:E108,E112]
The control that replaces physical bands by 16 learned linear rows trails the band model by 0.135 kappa on TUEV at seed 2. [claim:C016] [evidence:E010,E009]
Rotating by a band's own phase costs 0.03 to 0.24 relative to the measured rotation. [claim:C017] [evidence:E001,E002]
Inside the BIOT host the rotation changes TUEV kappa by about plus 0.20 and CHB-MIT AUC-PR by about minus 0.17 over three seeds. [claim:C018] [evidence:E001]

## Background

CBraMod's stem attached to our two-axis encoder reaches TUEV kappa 0.738 at seed 0, while CBraMod's own encoder trained from scratch with the same stem on the same arrays reaches 0.575. [claim:C019] [evidence:E011,E012]

## Experiments: setup and archive

The archive holds 2,681 result files; 748 canonical runs of our model family form 443 cells and 425 single-field contrasts. [claim:C020] [evidence:E001,E002]
TUEV has 68,445 / 15,487 / 29,421 windows and 3,243 distinct test signals. [claim:C021] [evidence:E013,E012]
TUSZ has 46,091 test windows and IIIC has 15,780 test rows. [claim:C022] [evidence:E012]

## Controlled signals

Untrained raw patch tokens give R squared 0.031 linear and 0.469 quadratic for log band power; demodulated tokens give 0.065 and 0.907. [claim:C023] [evidence:E015]
The envelope-token model trained on 2,160, 6,720, 20,000 and all 68,445 TUEV windows reaches kappa 0.652, 0.710, 0.677 and 0.708. [claim:C024] [evidence:E001]
SPaRCNet trained on 2,160, 6,720, 20,000 and all 68,445 TUEV windows reaches kappa 0.492, 0.553, 0.364 and 0.489. [claim:C025] [evidence:E001,E018]
Five rotation-direction settings exist on TUEV for the 16-band three-axis model: identity 0.729 (s0), measured 0.714 (s0) and 0.716 (s2), unaligned weights 0.643 (s2), different-window direction 0.601 (s0), own phase 0.532 (s2). [claim:C026] [evidence:E001]

## Attribution

Raw waveform tokens with one band give CHB-MIT 0.245 and TUSZ 0.558; with eight bands 0.667 and 0.671; paired seed differences are +0.39, +0.44, +0.43 and +0.19, +0.12, +0.03. [claim:C027] [evidence:E001,E002]
The flat encoder gives CHB-MIT 0.136 and TUSZ 0.448 against 0.667 and 0.671 for three-axis; paired differences +0.59, +0.45, +0.56 and +0.33, +0.18, +0.17. [claim:C028] [evidence:E001,E002]
Removing the band attention axis on an earlier frontend cost 0.078 kappa on TUEV, 0.014 AUROC on TUAB and 0.010 AUC-PR on TUSZ; removing electrode coordinates cost 0.080 on TUEV and 0.039 on TUSZ. [claim:C029] [evidence:E017]
Moving from 1 s to 0.25 s patches with 8-band envelope tokens raises TUEV kappa from 0.627 to 0.708 over three seeds; at fixed 20 tokens the moment window gives 0.708, 0.649, 0.611 for 0.25, 0.5, 1 s. [claim:C030] [evidence:E015]
At 16 bands the moment window gives 0.714, 0.739 and 0.737 for 0.25, 1 and 5 s at seed 0. [claim:C031] [evidence:E001,E003]
Token content under the three-axis reader with 8 bands, three seeds: waveform only 0.536 / 0.667, envelope only rotated 0.733 / 0.506, both rows 0.690 / 0.699 (TUEV / CHB-MIT). [claim:C032] [evidence:E001,E019]
Removing the waveform view costs 0.062 in the two-axis reader on CHB-MIT at seed 2. [claim:C033] [evidence:E001,E003]
Per-class TUEV F1 from waveform tokens to envelope-bearing tokens: GPED 0.55 to 0.82, PLED 0.52 to 0.61, ARTF 0.40 to 0.56, BCKG 0.88 to 0.92, EYEM 0.57 to 0.31, SPSW 0.16 to 0.00. [claim:C034] [evidence:E015]
The learned-row three-axis control reached TUEV kappa 0.576 and balanced accuracy 0.552 at seed 2. [claim:C035] [evidence:E010]
The one-row raw-patch control read by the two-axis encoder reaches TUEV 0.638. [claim:C036] [evidence:E012]
Measured against identity: three-axis 0.714/0.729, 0.716/0.711, 0.697/0.671, 0.710/0.699; two-axis 0.685/0.616, 0.713/0.699, 0.605/0.665, 0.624/0.650 (TUEV s0, s2, CHB s0, s2). [claim:C037] [evidence:E001,E009]
Own-phase rotation costs 0.03 to 0.24 relative to measured in 15 paired runs, 14 of 15 negative, mean minus 0.076. [claim:C038] [evidence:E002]
The weights-without-alignment variant costs 0.03 to 0.07 in three of three paired runs. [claim:C039] [evidence:E002]
BIOT host, measured against identity: TUEV 0.596, 0.583, 0.653 against 0.400, 0.385, 0.443; CHB-MIT 0.474, 0.454, 0.465 against 0.685, 0.646, 0.555; IIIC gain about 0.12. [claim:C040] [evidence:E001]
Three-axis against two-axis at 16 bands: CHB-MIT +0.089 (two seeds), CAUEEG +0.062, IIIC +0.057, TUEP +0.049, TUEV +0.016 (two seeds), TUSZ +0.016, TUAR +0.009, ADFD minus 0.017. [claim:C041] [evidence:E001,E003]
Product versus unit rotation +0.002 over 28 seeds; coupling-strength feature minus 0.004 over 28; own-band coupling +0.005 over 9; 8 versus 16 bands +0.024; augmentation +0.006 over 17; weight decay +0.007 over 7. [claim:C042] [evidence:E002]
Pretraining on amplitude reconstruction helped 3 of 12 corpora and hurt 9. [claim:C043] [evidence:E015]

## Main comparison

TUEV from-scratch baselines with native code at seed 0: CBraMod 0.575, EEG-Conformer 0.503, FFCL 0.459, BIOT 0.418, REVE 0.423. [claim:C044] [evidence:E012]
Weighting each distinct TUEV test signal once gives 0.614 for our two-axis model, 0.576 for a raw-patch control and 0.547 for CBraMod. [claim:C045] [evidence:E013]
Pretrained baselines under the earlier protocol: REVE TUEV 0.685 plus or minus 0.039, EEGPT TUSZ 0.545 plus or minus 0.070, REVE IIIC 0.436 plus or minus 0.003, LaBraM TUEP 0.778 plus or minus 0.019, CBraMod Sleep-EDF 0.672 plus or minus 0.005, EEGPT CAUEEG 0.547 plus or minus 0.021, REVE ADFD 0.501 plus or minus 0.049, CBraMod TUAR 0.715 (1). [claim:C046] [evidence:E018]
From-scratch baselines under the earlier protocol: FFCL TUSZ 0.545 plus or minus 0.030, FFCL IIIC 0.423 plus or minus 0.020, EEGPT TUEP 0.786 plus or minus 0.021, ContraWR Sleep-EDF 0.692 plus or minus 0.015, CNN-Transformer TUAR 0.712 (1), BIOT CAUEEG 0.561 plus or minus 0.012, BIOT ADFD 0.525 plus or minus 0.021. [claim:C047] [evidence:E018]
Rotation-on single-seed cells: TUEP 0.811, Sleep-EDF 0.700, TUAR 0.724, CAUEEG 0.543, ADFD 0.460; TUSZ three-axis 0.674; 8-band duplex TUEP 0.810 plus or minus 0.004. [claim:C048] [evidence:E001]
CHB-MIT natural grid with native code: CBraMod 0.570 and EEGPT 0.379 at seed 0. [claim:C049] [evidence:E012]
The first run of our model on the corrected CHB-MIT split stopped at validation AUC-PR 0.416, against 0.564 on the enriched split. [claim:C050] [evidence:E009,E005]
Legacy CHB-MIT validation and test sets had 21,184 / 17,386 windows with 150 / 276 positives; the natural grid has 21,094 / 17,224 with 60 / 114. [claim:C051] [evidence:E005]
TUEV has 144 identical-signal groups with conflicting labels and no identical signal across splits. [claim:C052] [evidence:E013]
The new coupling operator paired on TUEV gives measured 0.694 against identity 0.683 at seed 0. [claim:C053] [evidence:E020]
