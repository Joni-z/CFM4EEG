# CfFT implementation

This package contains the project’s learned filterbank, analytic-signal features,
relation-conditioned tokenizer and Transformer encoder. The `paclock` module and
class names are retained for checkpoint compatibility; the project/model name is
CfFT.

The [model guide](../../../docs/MODEL.md) describes current configurations,
the two-axis encoder, waveform controls and the optional copula frontend.
Filename `triaxial.py` also contains older axis options; the selected
configuration determines the actual encoder.

The repository cleanup did not change these tokenizer/encoder computations or
checkpoint tensor keys. Historical `_prev` and `_reference` copies are preserved
outside the active source tree. The pre-cleanup source snapshot and a fixed-input
forward/state comparison are recorded in the external maintenance directory.

Baseline reproduction uses [original-author entrypoints](../../../docs/UPSTREAM.md),
not this model package. See [dataset contracts](../../../docs/DATA.md) and
[audit records](../../../docs/AUDIT.md) for input and execution provenance.
