# CroFreMo manuscript figure plan — discussion draft, 2026-09-15

Title: **Cross-Frequency Modulation for EEG Foundation Models**.
Core message: measured PAC geometry determines token construction; retaining
within-band information is the structural requirement for broad usefulness.
The current manuscript describes the evaluated amplitude-carrier/duplex
reference. The new quadrature and waveform-carrier pilots are not established
final models and must not be illustrated as validated winners.

## Main figures: four questions, four figures

### Figure 1 — What does cross-frequency modulation add, and how does it construct a token?

A full-width conceptual/method figure, read left to right, about half a page.
- **(a) Signal relationship.** Draw a low-frequency phase trace, high-frequency
  oscillation and its envelope, aligned on one time axis. Shade recurring slow
  phases and show the fast envelope preferentially increasing there. A small
  counterexample can match per-band energy while changing the cross-band
  relationship. If synthetic, label it schematic and verify any claimed power
  matching; never claim identical full spectra from an illustrative drawing.
- **(b) Token construction (largest panel).** Learned bands -> analytic phase
  and envelope -> centered complex Z -> preferred-phase alignment -> carrier
  modulation -> token. Give the PAC operator most of the visual area; use a
  complex-plane inset to show alignment/rotation, with at most the Z, u and h
  equations. Color phase and amplitude consistently. Keep the carrier block
  modular until the structural candidate is selected. For the current reference,
  show amplitude-carrier tokens and their complementary waveform pathway.
- **(c) Representation use.** A small token grid enters an encoder, with clinical
  tasks as outputs. Use generic backbone blocks; avoid a large Transformer
  diagram that makes the novelty look like another attention architecture.
  Pretraining may be shown as the evaluation route, not as an already proven gain.

Caption claim: PAC supplies the geometry of token construction. Do not imply
that the short-window statistic proves physiological coupling, that waveform
information is globally lossless, or that an untested carrier is the final model.

### Figure 2 — Does one fixed tokenizer remain strong across clinical tasks?

A task-grouped, twelve-dataset dot/interval plot (small multiples if needed).
- One chosen model throughout. Show individual training seeds and means;
  use reported primary metrics, clearly labeled per task. No pooled average
  of kappa, AUC-PR and AUROC; no radar-chart area as evidence.
- Compare against a small predeclared set of strong, protocol-compatible
  references. Separate released-pretrained and scratch settings and distinguish
  local adapter results from published references with different cohorts.
- Missing/censored runs stay visibly missing/flagged. Do not draw intervals
  for single-seed cells or replace each dataset with its best historical variant.
- Keep full metrics and baseline tables available in the appendix once the
  figure replaces their main-text space. Generate all data through
  scripts/gen_tables.py and preserve source receipts.

Current status: the historical reference can be plotted now as a draft;
final plotting must wait for a fixed candidate and completed seed coverage.

### Figure 3 — Is the measured modulation responsible, and is waveform content retained?

Two clearly separated evidence panels, about half a page.
- **(a) Retrained controls:** selected candidate versus matched own-phase,
  amplitude-only/magnitude and waveform controls, with paired seed dots.
  TUEV + CHB-MIT are the primary complementary tasks; include a third task
  such as TUAR where the phase-intervention effect differs. Label any row-count
  or capacity mismatch rather than calling all controls strictly matched.
- **(b) Validation interventions:** perturb the preferred-phase pairing or
  remove a content pathway on a fixed checkpoint; plot paired changes, not
  only an attractive coupling heatmap. Preserve the distinction between
  independent training seeds and repeated perturbation draws.

Available historical receipts: TUEV/TUAR content knockout and preferred-phase
interventions in results/audits/*20260913.json. These describe the old factorized
model; their heatmaps/bars must not be attributed to the new candidates.
A PAC heatmap is supportive visualization, not sufficient causal evidence.

### Figure 4 — Does the tokenizer transfer beyond its own encoder?

Two task-specific panels; add a third pretraining panel only when supported.
- **(a) Host interventions:** CBraMod/LaBraM/REVE, native versus replacement;
  use a separate strip for CBraMod native + waveform versus native + modulation
  (the more specific content contrast). Mark scratch training prominently.
  Whole-frontend replacement also changes decomposition/token layout.
- **(b) Breadth of transfer:** show TUEV alongside CHB-MIT/TUSZ where available;
  keep negative effects. Avoid presenting TUEV-only improvements as universal.
- **(c), conditional:** scratch versus the SAME candidate after pretraining,
  low-label curves and/or unseen-corpus transfer. Do not populate with
  checkpoints from different architectures or schematic rising accuracy curves.
  If unavailable, keep Figure 4 about host portability and put the pretraining
  question in the limitations/experimental plan, not a fabricated panel.

## Appendix figures

1. Training/selection behavior: compare 8/16/24 bands by epochs or examples
   processed (never raw evaluation index across corpora), using validation
   only; mark best-checkpoint selection and budget stops. Late peaks alone
   are not performance wins, and n5 bundles band count with regularization.
2. TUEV per-class recall/confusion and CHB precision-recall curves for the
   validation-selected checkpoints; show seed variation and class support.
3. Efficiency: real full-step timing after warmup, parameters, token count
   and memory on the same hardware/precision/batch protocol.

## Visual style and production

White background, flat vector graphics, consistent restrained palette:
waveform navy, phase teal, amplitude orange, modulation purple; baselines gray.
Use labels/shapes as well as color. No 3-D brain decoration, giant module
catalogues, misleading smooth learning curves or screenshots of slides.

Architecture/concept art should be editable SVG/PDF; quantitative panels
should be generated with plotting code and include CSV/JSON provenance.
Check font legibility at final paper width. The old talk-specific TikZ files
modes.tex/tokenizer.tex and their PDFs are retired from the working folder.

Draw Figure 1 first: its PAC motivation and processing flow survive candidate
changes. Then choose exact plot panels/data for Figures 2–4. The carrier inset
is finalized only after deciding which structural design the paper reports.
