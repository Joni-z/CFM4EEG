# CFM4EEG: naming and research story

- Repository: **CFM4EEG**.
- Model: **CroFreMo**, expanded as **Cross-Frequency Modulation**.
- Paper title: **Cross-Frequency Modulation for EEG Foundation Models**.
- PACLock is a legacy implementation name. Preserve existing imports,
  environment variables, checkpoint/config identifiers, and cluster paths.

## Core contribution

PAC-guided cross-frequency modulation constructs the tokens themselves.
A learned filterbank yields analytic band signals. Centered patch-level
phase-amplitude covariation determines preferred-phase alignment and
relative weights. A normalized complex rotation modulates amplitude
features using this alignment. This is the proposed token-construction
prior; PAC as a neuroscience concept is not claimed as new.

Duplex retains complementary waveform information because the modulation
representation need not preserve all within-band morphology. Its role is
to support useful modulation-based representations across tasks.

## Complete research program

1. Establish one fixed candidate with strong results across more than ten
   datasets, compared with external strong baselines under aligned protocols.
2. Attribute improvements to measured modulation using matched own-phase,
   amplitude-only, learned-transformation, and surrogate-pairing controls.
   Use multiple seeds and multiple corpora for conclusions.
3. Show that waveform preservation retains morphology-sensitive performance
   without sacrificing the advantages of modulation tokens.
4. Test tokenization in other backbones, separating whole-frontend replacement
   from controls that specifically isolate the alignment operator.
5. Establish consistent pretraining and downstream transfer. Any proposed
   modulation-aware objective needs implementation and evaluation before
   it can appear as a completed method or result.

These are goals and evidence requirements, not claims that all have been met.
Current family-level tradeoffs and mixed transfer/pretraining outcomes remain
visible in the paper. PAC-inspired predictive gains alone do not demonstrate
physiological coupling. Tables must be generated from runs using
`scripts/gen_tables.py`; never edit numerical cells by hand.

The canonical manuscript is on mbp at
`/Users/zzz/Desktop/figure/ICLR2027-Paper`; publish using
`./push_overleaf.sh "message"`, which compiles before pushing to Overleaf.
