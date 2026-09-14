# D-axis screen: is augmentation admissible in the final model? (2026-09-14, agent A)

## Why
Augmentation is the only lever with measured gains on the corpora we lose. On the folded d192 encoder
(`cf2_v1d192`, seed 0) the bundle `flip0.5,jitter0.05,mask0.2,channel0.2,frequency0.2` moved
CAUEEG 0.518 -> 0.575 (baseline 0.561, a loss turned into a win), ISRUC 0.697 -> 0.733,
Sleep-EDF 0.633 -> 0.661, TUAR 0.591 -> 0.622 — but cost TUEV 0.655 -> 0.587 (-0.068).
On the factorized stack the same bundle cost TUEV only -0.0115 while gaining TUAR +0.0383, so the
cost is architecture-dependent and has never been measured on the tri-axial d128 stack, which is
where both leading candidates (rot2 coupling-only, duplex) and Codex's local_residual work live.

Second question, asked in the same screen: is the TUEV cost attributable to the two transforms that
attack the coupling statistic rather than to augmentation per se? `flip` is a time reversal, which
inverts the phase progression whose geometry Z_ij encodes; `frequency` zeroes FFT bins and destroys
band content outright. `jitter` (additive noise) and `channel` (whole-channel dropout) leave
within-channel cross-frequency structure intact.

## Arms — 4 runs, seed 0, mi2101x single-GPU nodes
Base config `configs/coupling_only/<ds>_paclock_rot2.yaml` (tri-axial, d128, depth 6, 8 bands,
patch 50, pac_interaction + rotation, freq attention), changed in exactly one field:
| arm | augmentations |
|---|---|
| augfull | flip0.5, jitter0.05, mask0.2, channel0.2, frequency0.2 |
| augsafe | jitter0.05, channel0.2 |
Corpora: TUEV (the corpus augmentation cost most) and CAUEEG (the corpus it gained most).

## Controls — already on disk at 3 seeds, not re-run
TUEV `paclock_rot2` 0.7328 +- 0.0161 (3) ; CAUEEG `paclock_rot2` 0.4900 +- 0.0344 (3).
Best reproduced baselines: TUEV reve_pretrained 0.685 +- 0.039 ; CAUEEG biot_scratch 0.561 +- 0.012.

## Predeclared decision, written before the runs land
- ADMIT GLOBALLY if an arm holds TUEV within 0.02 of 0.7328 AND lifts CAUEEG by >= +0.04.
  That arm's augmentation goes into every subsequent candidate on this stack, by both agents.
- ADMIT IN SAFE FORM ONLY if augsafe passes the TUEV clause and augfull does not — i.e. the cost is
  the coupling-hostile transforms, not augmentation.
- REJECT AS A GLOBAL SETTING if both arms cost TUEV more than 0.05. Augmentation is then a per-corpus
  knob, cannot appear in a single shipped model, and is reported as such.
- A seed-0 screen promotes at most to a paired seed-1/2 replication on the passing arm. It never
  promotes straight to a 12-corpus campaign, and a failure closes this configuration, not the family.

## Budget
4 runs, one mi2101x node each (16 cores, 1 GPU — about half the billing of a packed mi2104x node),
max_hours 4.5, wall limit 6 h. No pretraining, no Torch jobs, no b2.
