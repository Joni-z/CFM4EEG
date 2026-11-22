# Statistical details

The independent unit is a complete RNG realization. Four depths and four preferred phases share each positive realization; four independent-envelope depths share each null realization. Noise conditions share underlying draws and are reported separately. No patches are counted as independent replicates.

The primary scope retains all 20 native patches. A secondary interior scope retains patch indices 4–15, specified before outcomes. Patch errors are averaged within each recording before means and intervals. Pooled-vector angle is explicitly secondary.

Phase MAE, localization and FPR intervals use 2,000 realization-cluster bootstrap draws. Additional strength Spearman and detection AUC intervals use 500 cluster bootstrap draws. These are Monte Carlo uncertainty over synthetic realizations of one frozen checkpoint, not training-seed uncertainty or clinical patient uncertainty. No hypothesis-significance claims or multiple-task superiority claims are made.

The threshold is calibrated separately for each known frequency/noise condition on 200 held-out null realizations. It is a controlled benchmark threshold, not a deployment-ready operating threshold. AUC uses independent-envelope nulls matched by modulation-depth mixture; constant-envelope nulls are retained as separate controls. Calibration records never enter reported test AUC.

The offline 5 s reference was added after inspecting the initial native assay only to validate the synthetic distinction; it is not a preregistered competing model. Its paired AUC difference is descriptive.

All exact point estimates and primary intervals are in summary.json. Additional intervals and paired reference-minus-native AUC differences are in additional-ci.json. No test-truth gain calibration or score clipping was performed.
