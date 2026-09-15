# CHBMIT CBraMod+Quad16 classifier collapse

Two completed validation passes: AUROC0.5, balanced accuracy0.5, trapezoidal PR-AUC0.5035404. The PR number is the integration artifact for constant scores with prevalence150/21184, not useful discrimination.
CPU inspection of saved best.pt on145, using one positive and one negative TRAIN window: logits exactly[1.02268648,-1.33267391] for both, probability0.086640626. Quad token mean absolute sample difference1.07326; encoder difference5.09598. Second classifier Linear output range[-255.83,-55.82], its ELU output is exactly-1 for every channel/sample. Classifier is saturated, not merely predicting negative at the threshold.
Wrote STOP to only this source run to release GPU7 and retain checkpoint. No test evaluated for the stopped run. Original Quad CHBseed1/2 remain live; seed0 also had valPR~0.006-0.028 during first2800steps, so their current earlylowPR cannot support rejection.
Next engineering question: scale/optimization compatibility at the unchanged native CBraMod classifier. Any calibration/restart must have a new run identity and retain this failed admission history; no silent rewriting of oldresults.
