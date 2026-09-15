# CroFreMo naming, evidence figures and monitor — 2026-09-16

Public model name: CroFreMo. Tokenizer structure name: Duplex. The augmentation-only recipe is CroFreMo (+Aug). Preserve all internal A128 / a128_r1 IDs, configuration paths, checkpoints and seeds; renaming the manuscript does not rename or relabel any run.

The reference and current R1 use three actual attention axes: temporal, electrodes, and band/token rows. Configuration uses freq_mixer=attention and the default space_over_bands=false. The historical folded d192 branch uses two axes by folding electrode×row into spatial attention and removing the separate frequency sublayer. Duplex names two token routes, not encoder axes. PAC is explicitly encoded in token construction; the frequency-row mixer is ordinary attention.

Paper: main title remains Cross-Frequency Modulation for EEG Foundation Models. The two restored main-text data figures follow TFM's result ordering: Figure 3 host reuse at page 8 top; Figure 4 token-content and alignment controls at page 9 top. Main text ends on page 9; references begin on page 10. SciPilot profile/layout/grayscale checks and full-size rendered-page review completed. Source data and generation code are in the paper repository; all numerical displays generated from recorded results. Public model labels use CroFreMo; the appendix retains one explicit internal-ID mapping.

New completed host evidence:
- CroFreMo–BIOT TUEV seed0 test kappa 0.5692214915226723. Historical native seed0: 0.5038691031613584 (use exact result.json for reporting). Corresponding Quad seed0: 0.5559698284900686.
- Historical Quad–CBraMod IIIC seed0 test kappa 0.4159895348526409; native seed0 0.3978562666698662.
- Host comparison uses declared loader/root/recipe and matching class counts, but historical BIOT lacks manifest creation metadata. Do not claim verified runtime identity or a fresh paired rerun.

Figure 4 rotation-only controls use paclock_rot2 raw files: 3 TUEV seeds, 1 CHB-MIT, 1 TUSZ, no IIIC. Do not fill those gaps with a rounded frozen-table aggregate from a different coupling family. Waveform and Duplex have 3 seeds each. Alignment differences come from folded d192 measured-vs-own controls, not matched final-model attribution. Negative seed effects and unsuccessful host arms remain visible.

Early augmentation results (validation only; same-seed reference prefix matched by evaluation count, NOT test comparison):
TUEV kappa 0.6353965 vs original 0.609325; IIIC kappa 0.513693 vs 0.471718; CHB-MIT PR 0.035959 vs 0.037207; TUSZ PR 0.223268 vs 0.247381; Siena PR 0.463455 vs 0.448661; ADFD BA 0.466807 vs 0.483021; SleepEDF kappa 0.621434 vs 0.600384. These are early trajectories, not completed final scores. CHB-MIT is still in epoch0 and is not a dead line on this evidence. No new run canceled based on this inspection.

B2 pretrain chain 46083824/25/26 remains queued (Priority/Dependency), no pretraining compute yet. Shared 145 CroFreMo CBraMod–CHB-MIT seed0 remains guarded for a truly vacant GPU. Seed1/2 of new R1 remain deferred per the user's latest scheduling instruction. No new architecture or seed was launched in this manuscript turn.
