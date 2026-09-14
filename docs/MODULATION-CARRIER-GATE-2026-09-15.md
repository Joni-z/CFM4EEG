# F: preserve waveform information inside PAC modulation

User steering 2026-09-15: structural information preservation is the priority;
band count and regularization alone do not solve the representation tradeoff.
This replaces the prepared NB16-FOLLOWUP screen BEFORE any full training.
Hardware smoke 419224 was completed; none of its four candidate configs trained.

## Structural hypotheses

Original rotation uses h = a * u, where a is a real analytic-amplitude feature
and u is the normalized PAC-aligned phase. Own-band waveform phase is not
explicitly retained. Arbitrary h + waveform can interfere in the same coordinates;
duplex doubles rows; the attempted late routing has not rescued performance.
These observations motivate testing the carrier itself, not another pooling head.

Two new operators use exactly 16 rows and D128 throughout:

1. **Quadrature modulation:** h = (a + i w) * u. The existing amplitude feature
   and a learned real waveform projection occupy orthogonal coordinates before
   the same PAC rotation. The waveform map starts at zero; base parameters and
   initial logits are identical to n5. For unit u, the two contributions are
   orthogonal and |h|^2 = a^2 + w^2. No extra token rows or free additive bypass.
2. **Waveform carrier modulation:** h = (r + i w) * u. Both r and w project
   the filtered waveform (128 real coordinates for a 50-sample patch). PAC
   modulates these coordinates rather than replacing the band's phase with
   only a slow-phase representation. The initial combined projection is checked
   for rank 50; training can change its rank. The existing amplitude projection
   parameters carry r in this mode and remain trainable.

For fixed u away from the epsilon guard, conjugate rotation recovers the two
carrier coordinates and preserves their norm. This is NOT a proof that the
complete input-dependent tokenizer is invertible, that the learned waveform
map remains full rank, or that coupling and morphology remain independently
readable by the encoder. Carrier phase can still interact with coupling phase.
The paired task experiment is required. No performance guarantee is claimed.

Both add 3,200 trainable waveform-projection weights, preserve the base encoder
and its initial RNG stream, and use the SAME n5 16-band regularization recipe.
No changes to optimizer, loss, cadence, batch 32, 20-epoch cosine schedule,
training data or checkpoint-selection metric.

## Four runs, seed 0

- TUEV quadrature and carrier.
- CHB-MIT quadrature and carrier.

Existing TUEV n5 (one seed, val kappa .633040061) and CHB duplex (three seeds;
seed-0 val PR .678727686) are performance references, not repeated filler jobs.
The CHB reference differs in recipe/geometry, so this screen cannot isolate
an incremental benefit over a same-recipe CHB n5 baseline. Attribution needs
that control if a strong candidate emerges. These four runs compare two
structural operators under one recipe and test whether either retains the
needed task performance; they do not establish a universal 16-band benefit.

The available CHB n1 16-band run has val PR .670557898 after six epochs and
18.8 hours; old duplex also took six epochs and 18.7 hours. A single-card
12-hour cap risks censoring the informative stage. Site submission testing
confirmed 24h is accepted on mi2104x. Use one full four-card allocation,
max_hours 22 per CHB trainer and 7 per TUEV trainer. Report actual observed
patience/budget termination. Do not relabel censored runs as complete failures.

## GPU admission and gate

Before training, independently smoke each actual config on a GPU, exercising
all augmentation branches before timing and excluding initial warmup steps.
Check shared initialization, initial quadrature/base logit equality, coordinate
orthogonality, conditional inverse, flat-input finiteness, nonzero gradients
through PAC and waveform maps, rank of the initial full waveform map, and
strict saved-checkpoint roundtrip. Review peak memory and measured runtime.

Select checkpoints on validation only. Pilot target for each operator:
- TUEV val kappa >= .623040061 (within .01 of historical n5 seed 0), with
  validation balanced accuracy at that checkpoint no more than .02 below
  historical n5's selected-validation value. Preserve per-class recall.
- CHB val PR >= .658727686 (within .02 of historical duplex seed 0).

These are operational margins, not significance or statistical noninferiority
claims. If the historical selected-validation balanced accuracy cannot be
recovered, record that clause as unresolved; never substitute test accuracy.
No promotion until normally completed comparisons and both task requirements
are reviewed. A pilot pass enables a replication proposal only. A failure
closes this particular setting provisionally, not the whole architecture family.
No pretraining or external-backbone campaign in this screen.

## Historical selected-validation reference recovered before training

`PACLock/logs/DSN_01-416484-tuev_crofremo_n5-s0.out`, epoch 8, records
validation balanced accuracy **.5245** and kappa .6330. The full result records
kappa .633040061 and peak index 8. Only four-decimal balanced accuracy is
available, so use a four-decimal comparison floor **.5045** for that clause;
do not imply greater precision or substitute the .6926 test balanced accuracy.
