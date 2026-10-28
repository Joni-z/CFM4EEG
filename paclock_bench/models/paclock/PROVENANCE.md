# CfFT implementation

This package contains the project’s learned filterbank, analytic-signal features,
relation-conditioned tokenizer and Transformer encoder. The `paclock` module and
class names are retained for checkpoint compatibility; the project/model name is
CfFT.

The [model guide](../../../docs/MODEL.md) describes current configurations,
the two-axis encoder, waveform controls and optional measurement/copula frontends.
Filename `triaxial.py` also contains older axis options; the selected
configuration determines the actual encoder.

The standalone [CFC curve operator](../../../docs/CFC_OPERATOR.md) adds
signal-derived phase-frequency refinement and first/second-harmonic regression.
It does not replace any tokenizer frontend or historical checkpoint behavior.

`tokenizer_mode: measured_pac` is a separate optional frontend. It uses fixed
physical analysis anchors and first-harmonic phase-amplitude regression, then
conditions the existing content tokens. Its source-dependent statistical
context differs from the reference's short local moments.
`measurement_mode: identity` disables its complete guidance branch. Historical
local-moment checkpoints retain their original identity.

The [component validation record](../../../review/PAC_COMPONENT_VALIDATION_20261028.json)
pins clinical training to `b8ddd17` and exact full-test checkpoint replay to
`e2cf787`. The latter preserves measurement settings when loading checkpoints;
both selected TUEV checkpoints reproduce their original logits. Result and
checkpoint payloads remain in the external evidence archives.

The 2026-10-22 repository cleanup did not change the tokenizer/encoder computations or
checkpoint tensor keys. Historical `_prev` and `_reference` copies are preserved
outside the active source tree. The pre-cleanup source snapshot and a fixed-input
forward/state comparison are recorded in the external maintenance directory.

Baseline reproduction uses [original-author entrypoints](../../../docs/UPSTREAM.md),
not this model package. See [dataset contracts](../../../docs/DATA.md) and
[audit records](../../../docs/AUDIT.md) for input and execution provenance.
