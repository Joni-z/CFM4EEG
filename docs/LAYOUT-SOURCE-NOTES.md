# CfFT figure/table layout references

Downloaded arXiv TeX sources on 2026-10-16:
- JiT, 2511.13720v2: https://arxiv.org/src/2511.13720v2
- MeanFlow, 2505.13447v1: https://arxiv.org/src/2505.13447v1
- MAR, 2406.11838v3: https://arxiv.org/src/2406.11838v3

Read the actual source, not only PDF screenshots.

## Source findings and adaptations
- JiT experiments.tex uses small tables for one architectural/scaling question, grouped conditions and short headers; its preamble centralizes table spacing.
- MeanFlow tables/ablation.tex arranges six small minipage sub-tables (one factor each), marks defaults in gray, and states the common training setting once. tables/imgnet-256.tex distinguishes the primary matched family from contextual comparisons.
- MAR experiments.tex uses neighboring minipages for compact figures or tables and their captions, including temperature and sampling-step sweeps.
- CfFT keeps all main-table payloads, uses consistent 8 pt type and grouped dataset rules, and adds two adjacent compact analyses with separate captions: moment-window sensitivity and fixed-predictor reliance. No oversized extension beyond the CVPR text block or copied scientific claims.

## Evidence contract for the new plots
- Window plot: six disk-verified seed-0 result.json records; 0.25/1/5 s estimation context; token patch fixed to 0.25 s; earlier band-mixer encoder. Displays both primary metric and balanced accuracy; no invented uncertainty.
- Fixed-predictor plot: all six recorded interventions, same two-axis checkpoint and same 29421 test windows; no retraining. This is not a comparison of optimally trained alternatives.
- Full raw records retained with this revision. Compact plotting input and a reproducible pubfig/matplotlib script are under figures/src.
