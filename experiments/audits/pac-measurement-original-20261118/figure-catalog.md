# Figure catalog

- measurement-main.png/pdf: top row shows injected versus measured normalized depth; dashed line is identity, and colored lines are means across 100 realizations and four preferred phases per depth. Bottom row shows patch preferred-phase MAE; dashed 90° is the uniform-angle error reference. The figure displays means only; intervals are in summary.json and additional-ci.json. It shows monotonic strength response with substantial scale bias, good clean theta/alpha phase recovery, and noise-related deterioration.
- measurement-controls.png/pdf: native coupling-pair localization, coupling-versus-independent-envelope AUC, and calibrated FPR across all five frequency families. These are original 0.25 s outputs only. Low FPR must be read alongside the low TPR in summary.json. This figure shows that strength sensitivity does not establish PAC specificity.
- frozen-filter-responses.png: actual frozen FIR magnitude responses normalized separately by each filter peak. It depicts overlap and band coverage, not learned coupling. Exact carrier/sideband gains and response-based truth masks are in protocol.json. It prevents attributing all error to the statistic while ignoring the fixed filter bank.

No graph contains token/readout or downstream clinical results. No result has been inserted into the paper or presentation.
